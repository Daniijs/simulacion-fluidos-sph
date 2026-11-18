import numpy as np

class Dinamica:
    def __init__(self, sistema, h, densidad_reposo, viscosidad, k_presion):
        self.sistema = sistema
        self.h = h
        self.h2 = h * h
        self.densidad_reposo = densidad_reposo
        self.viscosidad = viscosidad
        self.k_presion = k_presion
        
        # Precalculamos constantes de los kernels para ahorrar tiempo
        self.poly6_const = 315.0 / (64.0 * np.pi * (h**9))
        self.spiky_grad_const = -45.0 / (np.pi * (h**6))
        self.visc_lap_const = 45.0 / (np.pi * (h**6))

    def calculaFuerzaSPH(self, hashing):
        num = self.sistema.num_activas
        if num == 0: return
        
        pos = self.sistema.pos[:num]
        vel = self.sistema.vel[:num]
        dens = self.sistema.densidad[:num]
        pres = self.sistema.presion[:num]
        fuerza = self.sistema.fuerza[:num]
        masa = self.sistema.masa[:num]

        # Limpiar fuerzas previas
        fuerza.fill(0)

        # 1. Obtener vecinos agrupados por celda (Optimización extrema)
        celdas_vecinos = hashing.get_celdas_vecinos()
        particle_keys = hashing.particle_keys
        
        # Estructura para guardar caché de distancias del paso 1 al paso 2
        # Guardaremos: (v_idx, r_vecs, r2s)
        cache_vecindades = [None] * num

        # PASO 1: Calcular Densidad y Presión
        for i in range(num):
            key = (particle_keys[i, 0], particle_keys[i, 1], particle_keys[i, 2])
            idx_vecinos = celdas_vecinos.get(key, None)
            
            if idx_vecinos is None or len(idx_vecinos) <= 1:
                # Sin vecinos o solo sí mismo
                d_i = masa[i] * self.poly6_const * (self.h2)**3
                dens[i] = d_i
                pres[i] = np.maximum(0, self.k_presion * (d_i - self.densidad_reposo))
                continue
            
            vecinos = np.array(idx_vecinos)
            r_vecs = pos[i] - pos[vecinos]
            r2s = np.sum(r_vecs**2, axis=1)
            
            # Filtrar por radio de suavizado h
            mask = r2s < self.h2
            
            # Excluir a sí mismo (r2 = 0) y posibles distancias nulas
            valid_mask = mask & (vecinos != i) & (r2s > 1e-9)
            v_idx = vecinos[valid_mask]
            
            if len(v_idx) == 0:
                # Solo tiene autodespreciable
                d_i = masa[i] * self.poly6_const * (self.h2)**3
                dens[i] = d_i
                pres[i] = np.maximum(0, self.k_presion * (d_i - self.densidad_reposo))
                continue
                
            r2s_filtered = r2s[valid_mask]
            r_vecs_filtered = r_vecs[valid_mask]
            
            # Calcular densidad: vecinos + autoinfluencia
            d_i = np.sum(masa[v_idx] * self.poly6_const * (self.h2 - r2s_filtered)**3)
            d_i += masa[i] * self.poly6_const * (self.h2)**3
            
            dens[i] = d_i
            pres[i] = np.maximum(0, self.k_presion * (d_i - self.densidad_reposo))
            
            # Guardamos en caché toda la info de distancias y vectores filtrados
            cache_vecindades[i] = (v_idx, r_vecs_filtered, r2s_filtered)

        # PASO 2: Calcular Fuerzas SPH (Presión + Viscosidad + Cohesión)
        # Cero lookups de celdas, cero cálculos redundantes de distancias o restas de vectores!
        for i in range(num):
            cached = cache_vecindades[i]
            if cached is None: continue
            
            v_idx, r_vecs, r2s = cached
            rs = np.sqrt(r2s)
            
            # Preparar términos para vectorización rápida
            rho_i = max(dens[i], self.densidad_reposo)
            vol_i = masa[i] / rho_i
            
            rho_j = np.maximum(dens[v_idx], self.densidad_reposo)
            vol_j = masa[v_idx] / rho_j
            
            # 1. Fuerza de Presión
            p_term = (pres[i] + pres[v_idx]) / 2.0
            grad_w = (self.spiky_grad_const * (self.h - rs)**2)[:, np.newaxis] * (r_vecs / rs[:, np.newaxis])
            f_presion = -np.sum((vol_i * vol_j * p_term)[:, np.newaxis] * grad_w, axis=0)
            
            # 2. Fuerza de Viscosidad
            v_diff = vel[v_idx] - vel[i]
            lap_w = (self.visc_lap_const * (self.h - rs))[:, np.newaxis]
            f_viscosidad = np.sum(self.viscosidad * (vol_i * vol_j)[:, np.newaxis] * v_diff * lap_w, axis=0)

            # 3. Fuerza de Cohesión (Tensión superficial simple para evitar "dispersiones espurias")
            cohesion_coeff = 0.5
            f_cohesion = np.sum((masa[v_idx] * cohesion_coeff)[:, np.newaxis] * (-r_vecs) * ((self.h - rs)**3)[:, np.newaxis], axis=0)
            
            f_total = f_presion + f_viscosidad + f_cohesion
            
            # LIMITADOR DE FUERZA (Anti-Explosión)
            limite = 100.0 
            fuerza[i] = np.clip(f_total, -limite, limite)
