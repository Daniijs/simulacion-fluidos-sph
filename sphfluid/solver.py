"""Position Based Fluids solver (Macklin & Müller, 2013) written in Taichi.

One step:

1. predict positions with external forces,
2. sort particles into the neighbour grid and reorder memory,
3. iteratively project the density constraint ``rho_i / rho_0 - 1 <= 0``,
4. derive velocities from the corrected positions,
5. apply XSPH smoothing and, optionally, vorticity confinement.

Solid walls are static boundary particles whose contribution is weighted by
their volume (Akinci et al., 2012), so arbitrary meshes can be used as obstacles.
"""

import math

import numpy as np
import taichi as ti

from .colliders import ARCH_PARAMS, BOX_PARAMS, pack_colliders
from .config import FluidParams, SolverParams
from .geometry import BOUNDARY_SPACING_RATIO
from .grid import UniformGrid
from .kernels import (
    boundary_volume_scale,
    lattice_offsets,
    poly6,
    poly6_coefficient,
    spiky_gradient,
    spiky_gradient_coefficient,
)
from .scene import SceneGeometry

NEIGHBOUR_CELLS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]


@ti.data_oriented
class PBFSolver:
    def __init__(self, fluid: FluidParams, params: SolverParams, scene: SceneGeometry):
        self.params = params
        self.time = 0.0
        self._pending_gates = sorted(enumerate(scene.gates, start=1), key=lambda g: g[1].open_time)

        # --- constants (compile-time inside kernels) ---
        dx = fluid.particle_spacing
        h = fluid.support_radius
        self.h, self.h2 = h, h * h
        self.rho0 = fluid.rest_density
        self.poly6_k = poly6_coefficient(h)
        self.spiky_k = spiky_gradient_coefficient(h)
        self.w0 = self.poly6_k * h**6

        offsets = lattice_offsets(dx, h)
        self.mass = self.rho0 / float(poly6(np.linalg.norm(offsets, axis=1), h).sum())
        grad = spiky_gradient(offsets, h) * self.mass / self.rho0
        reference = float((grad**2).sum())  # sum |grad C|^2 for a full neighbourhood
        self.epsilon = params.relaxation * reference
        self.tensile_scale = params.tensile_k / reference
        self.inv_w_dq = 1.0 / float(poly6(params.tensile_dq * h, h))
        self.gravity = ti.Vector(list(params.gravity))
        self.lower = ti.Vector([float(c) + 0.5 * dx for c in scene.domain_lower])
        self.upper = ti.Vector([float(c) - 0.5 * dx for c in scene.domain_upper])
        self.sink_height = float(max(scene.sink_height, -1e30))
        self.max_correction = params.max_correction * dx
        self.boundary_scale = boundary_volume_scale(dx, h, dx * BOUNDARY_SPACING_RATIO, params.boundary_layers)

        # --- fluid state ---
        n = len(scene.fluid)
        self.num_particles = n
        self.x = ti.Vector.field(3, ti.f32, n)
        self.x_old = ti.Vector.field(3, ti.f32, n)
        self.v = ti.Vector.field(3, ti.f32, n)
        self.delta = ti.Vector.field(3, ti.f32, n)
        self.scratch = ti.Vector.field(3, ti.f32, n)
        self.omega = ti.Vector.field(3, ti.f32, n)
        self.lam = ti.field(ti.f32, n)
        self.rho = ti.field(ti.f32, n)
        self.x.from_numpy(scene.fluid.astype(np.float32))
        velocity = scene.fluid_velocity if scene.fluid_velocity is not None else np.zeros((n, 3))
        self.v.from_numpy(velocity.astype(np.float32))

        lower = np.asarray(scene.domain_lower) - h
        upper = np.asarray(scene.domain_upper) + h
        self.grid = UniformGrid(lower, upper, h, n)
        self.num_active = n

        # --- boundary particles (static, sorted once) ---
        boundary = [scene.boundary] + [g.particles for _, g in self._pending_gates]
        groups = [np.zeros(len(scene.boundary), np.int32)] + [
            np.full(len(g.particles), gid, np.int32) for gid, g in self._pending_gates
        ]
        b_pos = np.concatenate(boundary).astype(np.float32) if boundary else np.zeros((0, 3), np.float32)
        b_group = np.concatenate(groups)
        m = max(len(b_pos), 1)
        self.num_boundary = len(b_pos)
        self.bx = ti.Vector.field(3, ti.f32, m)
        self.b_scratch = ti.Vector.field(3, ti.f32, m)
        self.b_group = ti.field(ti.i32, m)
        self.b_group_scratch = ti.field(ti.i32, m)
        self.psi = ti.field(ti.f32, m)
        self.bgrid = UniformGrid(lower, upper, h, m)
        if self.num_boundary:
            self.bx.from_numpy(b_pos)
            self.b_group.from_numpy(b_group)
            self.bgrid.build(self.bx, -1e30, self.num_boundary)
            self.bgrid.reorder(self.bx, self.b_scratch, self.num_boundary)
            self.bgrid.reorder(self.b_group, self.b_group_scratch, self.num_boundary)
            self._compute_boundary_volumes(self.num_boundary)

        self.terrain_enabled = scene.terrain is not None
        self._init_terrain(scene)
        self._init_colliders(scene, dx)

    # ------------------------------------------------------------------ public API

    def advance(self, frame_dt: float) -> int:
        """Advance by ``frame_dt`` seconds with CFL-limited sub-steps. Returns the sub-step count."""
        vmax = max(self._max_speed(self.num_active), 1e-3)
        dt_cfl = self.params.cfl * self.h / vmax
        substeps = max(1, math.ceil(frame_dt / min(dt_cfl, self.params.max_dt)))
        dt = frame_dt / substeps
        for _ in range(substeps):
            self._open_due_gates()
            self.step(dt)
        return substeps

    def step(self, dt: float) -> None:
        self._predict(dt, self.num_active)
        self.num_active = self.grid.build(self.x, self.sink_height, self.num_particles)
        for field in (self.x, self.x_old, self.v):
            self.grid.reorder(field, self.scratch, self.num_particles)
        for _ in range(self.params.iterations):
            self._compute_lambda(self.num_active)
            self._compute_delta(self.num_active)
            self._apply_delta(self.num_active)
        self._update_velocity(dt, self.num_active)
        if self.time < self.params.settle_time:
            # Deep water starts from a uniform lattice, i.e. without its hydrostatic
            # compression; damping lets it settle without rebounding.
            self._damp(1.0 - self.params.settle_damping, self.num_active)
        self._xsph(self.num_active)
        if self.params.vorticity > 0.0:
            self._vorticity(dt, self.num_active)
        self.time += dt

    def measure_density(self) -> np.ndarray:
        """Density of every active particle at the current positions, without moving them."""
        self.num_active = self.grid.build(self.x, self.sink_height, self.num_particles)
        for field in (self.x, self.x_old, self.v):
            self.grid.reorder(field, self.scratch, self.num_particles)
        self._compute_lambda(self.num_active)
        return self.densities()

    def positions(self) -> np.ndarray:
        return self.x.to_numpy()[: self.num_active]

    def velocities(self) -> np.ndarray:
        return self.v.to_numpy()[: self.num_active]

    def densities(self) -> np.ndarray:
        return self.rho.to_numpy()[: self.num_active]

    # ------------------------------------------------------------------ helpers

    def _open_due_gates(self) -> None:
        while self._pending_gates and self._pending_gates[0][1].open_time <= self.time:
            gid, _ = self._pending_gates.pop(0)
            self._disable_group(gid, self.num_boundary)

    def _init_colliders(self, scene: SceneGeometry, dx: float) -> None:
        colliders = list(scene.colliders)
        for gid, gate in self._pending_gates:
            if any(c.group != gid for c in gate.colliders):
                raise ValueError(f"Colliders of gate '{gate.name}' must use group {gid}")
            colliders += list(gate.colliders)
        boxes, arches = pack_colliders(colliders)
        self.num_boxes, self.num_arches = len(boxes), len(arches)
        self.boxes = ti.field(ti.f32, (max(self.num_boxes, 1), BOX_PARAMS))
        self.arches = ti.field(ti.f32, (max(self.num_arches, 1), ARCH_PARAMS))
        if self.num_boxes:
            self.boxes.from_numpy(boxes)
        if self.num_arches:
            self.arches.from_numpy(arches)
        self.group_enabled = ti.field(ti.i32, len(scene.gates) + 1)
        self.group_enabled.fill(1)
        self.collider_margin = 0.35 * dx

    def _init_terrain(self, scene: SceneGeometry) -> None:
        if scene.terrain is None:
            self.heights = ti.field(ti.f32, (2, 2))
            self.terrain_origin = ti.Vector([0.0, 0.0])
            self.terrain_inv_cell = 1.0
            self.terrain_res = ti.Vector([2, 2])
            return
        t = scene.terrain
        self.heights = ti.field(ti.f32, t.heights.shape)
        self.heights.from_numpy(t.heights.astype(np.float32))
        self.terrain_origin = ti.Vector([t.origin_x, t.origin_z])
        self.terrain_inv_cell = 1.0 / t.cell
        self.terrain_res = ti.Vector(list(t.heights.shape))

    # ------------------------------------------------------------------ kernel functions

    @ti.func
    def _w(self, r2):
        return self.poly6_k * (self.h2 - r2) ** 3

    @ti.func
    def _grad_w(self, r, r2):
        length = ti.sqrt(r2)
        return self.spiky_k * (self.h - length) ** 2 / length * r

    @ti.func
    def _terrain_height(self, p):
        f = (ti.Vector([p[0], p[2]]) - self.terrain_origin) * self.terrain_inv_cell
        height = -1e9  # no terrain outside the heightfield
        if (f >= 0.0).all() and (f <= self.terrain_res - 1.0).all():
            f = ti.min(f, self.terrain_res - 1.001)
            i = ti.cast(ti.floor(f), ti.i32)
            t = f - i
            h00 = self.heights[i[0], i[1]]
            h10 = self.heights[i[0] + 1, i[1]]
            h01 = self.heights[i[0], i[1] + 1]
            h11 = self.heights[i[0] + 1, i[1] + 1]
            height = (h00 * (1 - t[0]) + h10 * t[0]) * (1 - t[1]) + (h01 * (1 - t[0]) + h11 * t[0]) * t[1]
        return height

    @ti.func
    def _confine(self, p):
        q = ti.max(ti.min(p, self.upper), self.lower)
        if ti.static(self.terrain_enabled):
            depth = self._terrain_height(q) - q[1]
            if depth > 0.0:
                # Push out along the surface normal, not vertically: on steep slopes a
                # vertical projection turns a tiny overlap into a large jump.
                e = 0.5 / self.terrain_inv_cell
                dhdx = (
                    self._terrain_height(q + ti.Vector([e, 0.0, 0.0]))
                    - self._terrain_height(q - ti.Vector([e, 0.0, 0.0]))
                ) / (2 * e)
                dhdz = (
                    self._terrain_height(q + ti.Vector([0.0, 0.0, e]))
                    - self._terrain_height(q - ti.Vector([0.0, 0.0, e]))
                ) / (2 * e)
                n = ti.Vector([-dhdx, 1.0, -dhdz]).normalized()
                q += n * depth * n[1]
        for k in range(self.num_boxes):
            q = self._push_out_of_box(q, k)
        for k in range(self.num_arches):
            q = self._push_out_of_arch(q, k)
        return q

    @ti.func
    def _push_out_of_box(self, p, k):
        q = p
        if self.group_enabled[ti.cast(self.boxes[k, 7], ti.i32)]:
            c = ti.Vector([self.boxes[k, 0], self.boxes[k, 1], self.boxes[k, 2]])
            half = ti.Vector([self.boxes[k, 3], self.boxes[k, 4], self.boxes[k, 5]]) + self.collider_margin
            cs, sn = ti.cos(self.boxes[k, 6]), ti.sin(self.boxes[k, 6])
            d = p - c
            local = ti.Vector([cs * d[0] - sn * d[2], d[1], sn * d[0] + cs * d[2]])
            gap = half - ti.abs(local)
            if gap.min() > 0.0:  # inside: leave through the nearest face
                axis = 0
                if gap[1] < gap[axis]:
                    axis = 1
                if gap[2] < gap[axis]:
                    axis = 2
                local[axis] = ti.math.sign(local[axis]) * half[axis]
                q = c + ti.Vector([cs * local[0] + sn * local[2], local[1], -sn * local[0] + cs * local[2]])
        return q

    @ti.func
    def _push_out_of_arch(self, p, k):
        q = p
        if self.group_enabled[ti.cast(self.arches[k, 10], ti.i32)]:
            cx, cz, radius = self.arches[k, 0], self.arches[k, 1], self.arches[k, 2]
            th0, th1, y0, y1 = self.arches[k, 3], self.arches[k, 4], self.arches[k, 5], self.arches[k, 6]
            t_top, t_base, crest = self.arches[k, 7], self.arches[k, 8], self.arches[k, 9]
            m = self.collider_margin
            dx, dz = cx - p[0], p[2] - cz
            rho = ti.sqrt(dx * dx + dz * dz)
            theta = ti.atan2(dz, dx)
            t = t_top + (t_base - t_top) * ti.min(ti.max((crest - p[1]) / crest, 0.0), 1.0)
            inner, outer = radius - 0.5 * t - m, radius + 0.5 * t + m
            a0, a1 = th0 - m / rho, th1 + m / rho
            if inner < rho < outer and a0 < theta < a1 and y0 < p[1] < y1 + m:
                pen = ti.Vector([rho - inner, outer - rho, y1 + m - p[1], rho * (theta - a0), rho * (a1 - theta)])
                best = 0
                for i in ti.static(range(1, 5)):
                    if pen[i] < pen[best]:
                        best = i
                new_rho, new_theta = rho, theta
                if best == 0:
                    new_rho = inner
                elif best == 1:
                    new_rho = outer
                elif best == 2:
                    q[1] = y1 + m
                elif best == 3:
                    new_theta = a0
                else:
                    new_theta = a1
                q[0] = cx - new_rho * ti.cos(new_theta)
                q[2] = cz + new_rho * ti.sin(new_theta)
        return q

    # ------------------------------------------------------------------ kernels

    @ti.kernel
    def _compute_boundary_volumes(self, m: ti.i32):
        for b in range(m):
            xb = self.bx[b]
            total = self.w0
            cb = self.bgrid.cell_of(xb)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = cb + ti.Vector(off)
                if self.bgrid.contains(c):
                    start, end = self.bgrid.cell_range(c)
                    for k in range(start, end):
                        if k != b:
                            r = xb - self.bx[k]
                            r2 = r.dot(r)
                            if r2 < self.h2:
                                total += self._w(r2)
            self.psi[b] = self.boundary_scale * self.rho0 / total

    @ti.kernel
    def _disable_group(self, group: ti.i32, m: ti.i32):
        self.group_enabled[group] = 0
        for b in range(m):
            if self.b_group[b] == group:
                self.psi[b] = 0.0

    @ti.kernel
    def _predict(self, dt: ti.f32, n: ti.i32):
        for i in range(n):
            self.x_old[i] = self.x[i]
            self.v[i] += dt * self.gravity
            self.x[i] = self._confine(self.x[i] + dt * self.v[i])

    @ti.kernel
    def _compute_lambda(self, n: ti.i32):
        for i in range(n):
            xi = self.x[i]
            rho = self.mass * self.w0
            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_grad2 = 0.0
            ci = self.grid.cell_of(xi)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = ci + ti.Vector(off)
                if self.grid.contains(c):
                    start, end = self.grid.cell_range(c)
                    for j in range(start, ti.min(end, n)):
                        r = xi - self.x[j]
                        r2 = r.dot(r)
                        if j != i and r2 < self.h2 and r2 > 1e-12:
                            rho += self.mass * self._w(r2)
                            g = (self.mass / self.rho0) * self._grad_w(r, r2)
                            grad_i += g
                            sum_grad2 += g.dot(g)
                    start_b, end_b = self.bgrid.cell_range(c)
                    for b in range(start_b, end_b):
                        r = xi - self.bx[b]
                        r2 = r.dot(r)
                        if r2 < self.h2 and r2 > 1e-12:
                            rho += self.psi[b] * self._w(r2)
                            grad_i += (self.psi[b] / self.rho0) * self._grad_w(r, r2)
            self.rho[i] = rho
            constraint = ti.max(rho / self.rho0 - 1.0, 0.0)
            self.lam[i] = -constraint / (sum_grad2 + grad_i.dot(grad_i) + self.epsilon)

    @ti.kernel
    def _compute_delta(self, n: ti.i32):
        for i in range(n):
            xi = self.x[i]
            lam_i = self.lam[i]
            fluid_term = ti.Vector([0.0, 0.0, 0.0])
            solid_term = ti.Vector([0.0, 0.0, 0.0])
            ci = self.grid.cell_of(xi)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = ci + ti.Vector(off)
                if self.grid.contains(c):
                    start, end = self.grid.cell_range(c)
                    for j in range(start, ti.min(end, n)):
                        r = xi - self.x[j]
                        r2 = r.dot(r)
                        if j != i and r2 < self.h2 and r2 > 1e-12:
                            s_corr = -self.tensile_scale * (self._w(r2) * self.inv_w_dq) ** self.params.tensile_n
                            fluid_term += (lam_i + self.lam[j] + s_corr) * self._grad_w(r, r2)
                    start_b, end_b = self.bgrid.cell_range(c)
                    for b in range(start_b, end_b):
                        r = xi - self.bx[b]
                        r2 = r.dot(r)
                        if r2 < self.h2 and r2 > 1e-12:
                            solid_term += self.psi[b] * self._grad_w(r, r2)
            self.delta[i] = (self.mass * fluid_term + lam_i * solid_term) / self.rho0

    @ti.kernel
    def _apply_delta(self, n: ti.i32):
        # Where the density gradient nearly cancels (e.g. a particle squeezed between
        # two boundary layers) the correction can blow up; cap it to keep the solve stable.
        for i in range(n):
            d = self.delta[i]
            length = d.norm()
            if length > self.max_correction:
                d *= self.max_correction / length
            self.x[i] = self._confine(self.x[i] + d)

    @ti.kernel
    def _update_velocity(self, dt: ti.f32, n: ti.i32):
        for i in range(n):
            v = (self.x[i] - self.x_old[i]) / dt
            speed = v.norm()
            if speed > self.params.max_speed:
                v *= self.params.max_speed / speed
            self.v[i] = v

    @ti.kernel
    def _damp(self, factor: ti.f32, n: ti.i32):
        for i in range(n):
            self.v[i] *= factor

    @ti.kernel
    def _xsph(self, n: ti.i32):
        for i in range(n):
            xi, vi = self.x[i], self.v[i]
            correction = ti.Vector([0.0, 0.0, 0.0])
            ci = self.grid.cell_of(xi)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = ci + ti.Vector(off)
                if self.grid.contains(c):
                    start, end = self.grid.cell_range(c)
                    for j in range(start, ti.min(end, n)):
                        r = xi - self.x[j]
                        r2 = r.dot(r)
                        if j != i and r2 < self.h2:
                            correction += (self.mass / self.rho[j]) * (self.v[j] - vi) * self._w(r2)
            self.scratch[i] = vi + self.params.xsph * correction
        for i in range(n):
            self.v[i] = self.scratch[i]

    @ti.kernel
    def _vorticity(self, dt: ti.f32, n: ti.i32):
        # omega_i = sum_j m/rho_j (v_j - v_i) x gradW_ij
        for i in range(n):
            xi, vi = self.x[i], self.v[i]
            w = ti.Vector([0.0, 0.0, 0.0])
            ci = self.grid.cell_of(xi)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = ci + ti.Vector(off)
                if self.grid.contains(c):
                    start, end = self.grid.cell_range(c)
                    for j in range(start, ti.min(end, n)):
                        r = xi - self.x[j]
                        r2 = r.dot(r)
                        if j != i and r2 < self.h2 and r2 > 1e-12:
                            w += (self.mass / self.rho[j]) * (self.v[j] - vi).cross(self._grad_w(r, r2))
            self.omega[i] = w
        # f_i = eps (N x omega_i), N = normalised gradient of |omega|
        for i in range(n):
            xi = self.x[i]
            eta = ti.Vector([0.0, 0.0, 0.0])
            ci = self.grid.cell_of(xi)
            for off in ti.static(NEIGHBOUR_CELLS):
                c = ci + ti.Vector(off)
                if self.grid.contains(c):
                    start, end = self.grid.cell_range(c)
                    for j in range(start, ti.min(end, n)):
                        r = xi - self.x[j]
                        r2 = r.dot(r)
                        if j != i and r2 < self.h2 and r2 > 1e-12:
                            eta += (self.mass / self.rho[j]) * self.omega[j].norm() * self._grad_w(r, r2)
            if eta.norm() > 1e-6:
                self.v[i] += dt * self.params.vorticity * eta.normalized().cross(self.omega[i])

    @ti.kernel
    def _max_speed(self, n: ti.i32) -> ti.f32:
        result = 0.0
        for i in range(n):
            ti.atomic_max(result, self.v[i].norm())
        return result
