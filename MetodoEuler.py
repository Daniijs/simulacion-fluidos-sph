import numpy as np

class MetodoEuler:
    @staticmethod
    def integra(sistema, dt, gravedad_vec):
        num = sistema.num_activas
        if num == 0: return
        
        pos = sistema.pos
        vel = sistema.vel
        fuerza = sistema.fuerza
        masa = sistema.masa
        
        # Gravedad como array de numpy
        g = np.array([gravedad_vec.x, gravedad_vec.y, gravedad_vec.z], dtype=np.float32)
        
        # Integración de Euler vectorizada (Súper rápida)
        # v = v + (F/m + g) * dt
        # p = p + v * dt
        
        # masa es 1D, necesitamos expandirla para división de matriz Nx3
        m_inv = 1.0 / (masa[:num, np.newaxis] + 1e-9)
        
        acel = (fuerza[:num] * m_inv) + g
        vel[:num] += acel * dt
        pos[:num] += vel[:num] * dt

