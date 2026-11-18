import numpy as np

class GestorColision:
    @staticmethod
    def colisionLimites(LimitesX, LimitesY, LimitesZ, espesor, sistema, tasaRozamiento=0.9, tasaRebote=0.3):
        num = sistema.num_activas
        if num == 0: return
        pos = sistema.pos[:num]
        vel = sistema.vel[:num]
        
        # Mascaras booleanas para cada limite
        # Limites X
        low_x = pos[:, 0] < LimitesX[0] + espesor
        high_x = pos[:, 0] > LimitesX[1] - espesor
        
        pos[low_x, 0] = LimitesX[0] + espesor
        vel[low_x, 0] *= -tasaRebote
        vel[low_x, 1] *= tasaRozamiento
        vel[low_x, 2] *= tasaRozamiento
        
        pos[high_x, 0] = LimitesX[1] - espesor
        vel[high_x, 0] *= -tasaRebote
        vel[high_x, 1] *= tasaRozamiento
        vel[high_x, 2] *= tasaRozamiento

        # Limites Y
        low_y = pos[:, 1] < LimitesY[0] + espesor
        high_y = pos[:, 1] > LimitesY[1] - espesor
        
        pos[low_y, 1] = LimitesY[0] + espesor
        vel[low_y, 1] *= -tasaRebote
        vel[low_y, 0] *= tasaRozamiento
        vel[low_y, 2] *= tasaRozamiento
        
        pos[high_y, 1] = LimitesY[1] - espesor
        vel[high_y, 1] *= -tasaRebote
        vel[high_y, 0] *= tasaRozamiento
        vel[high_y, 2] *= tasaRozamiento

        # Limites Z
        low_z = pos[:, 2] < LimitesZ[0] + espesor
        high_z = pos[:, 2] > LimitesZ[1] - espesor
        
        pos[low_z, 2] = LimitesZ[0] + espesor
        vel[low_z, 2] *= -tasaRebote
        vel[low_z, 0] *= tasaRozamiento
        vel[low_z, 1] *= tasaRozamiento
        
        pos[high_z, 2] = LimitesZ[1] - espesor
        vel[high_z, 2] *= -tasaRebote
        vel[high_z, 0] *= tasaRozamiento
        vel[high_z, 1] *= tasaRozamiento


    @staticmethod
    def colisionEsfera(centro_vpy, radio, sistema, tasaRozamiento=0.995, tasaRebote=0.1):
        num = sistema.num_activas
        if num == 0: return
        pos = sistema.pos[:num]
        vel = sistema.vel[:num]
        centro = np.array([centro_vpy.x, centro_vpy.y, centro_vpy.z], dtype=np.float32)
        
        rel_pos = pos - centro
        dist2 = np.sum(rel_pos**2, axis=1)
        dist = np.sqrt(dist2)
        
        mask = dist < radio
        if np.any(mask):
            normal = rel_pos[mask] / (dist[mask, np.newaxis] + 1e-9)
            # Añadimos un pequeño offset de 1e-4 para evitar colisiones continuas
            pos[mask] = centro + normal * (radio + 1e-4)
            
            # Velocidad relativa normal: v_normal = dot(v, normal)
            v_dot_n = np.sum(vel[mask] * normal, axis=1)
            
            # Solo rebotar si se mueve hacia adentro
            rebound_mask = v_dot_n < 0
            if np.any(rebound_mask):
                idx = np.where(mask)[0][rebound_mask]
                n = normal[rebound_mask]
                v = vel[idx]
                vn = v_dot_n[rebound_mask, np.newaxis] * n
                vt = v - vn
                vel[idx] = vt * tasaRozamiento - vn * tasaRebote

