import numpy as np

class SistemaParticulas:
    def __init__(self, capacidad_inicial=10000):
        self.capacidad = capacidad_inicial
        self.num_activas = 0
        
        # Estructuras de datos NumPy para máxima velocidad
        self.pos = np.zeros((self.capacidad, 3), dtype=np.float32)
        self.vel = np.zeros((self.capacidad, 3), dtype=np.float32)
        self.fuerza = np.zeros((self.capacidad, 3), dtype=np.float32)
        self.densidad = np.zeros(self.capacidad, dtype=np.float32)
        self.presion = np.zeros(self.capacidad, dtype=np.float32)
        self.masa = np.zeros(self.capacidad, dtype=np.float32)
        
        self.particulas = []

    def addParticula(self, p):
        if self.num_activas >= self.capacidad:
            self._expandir()
            
        idx = self.num_activas
        self.pos[idx] = [p.pos.x, p.pos.y, p.pos.z]
        self.vel[idx] = [p.vel.x, p.vel.y, p.vel.z]
        self.masa[idx] = p.masa
        self.particulas.append(p)
        self.num_activas += 1

    def _expandir(self):
        nueva_capacidad = self.capacidad * 2
        self.pos = np.resize(self.pos, (nueva_capacidad, 3))
        self.vel = np.resize(self.vel, (nueva_capacidad, 3))
        self.fuerza = np.resize(self.fuerza, (nueva_capacidad, 3))
        self.densidad = np.resize(self.densidad, (nueva_capacidad,))
        self.presion = np.resize(self.presion, (nueva_capacidad,))
        self.masa = np.resize(self.masa, (nueva_capacidad,))
        self.capacidad = nueva_capacidad

    def getListaParticulas(self):
        # Actualizamos las posiciones y datos físicos de los objetos por si se usan en exportadores
        for i in range(self.num_activas):
            p = self.particulas[i]
            p.pos.x, p.pos.y, p.pos.z = self.pos[i]
            p.vel.x, p.vel.y, p.vel.z = self.vel[i]
            p.densidad = float(self.densidad[i])
            p.presion = float(self.presion[i])
        return self.particulas
    
    def getNumParticulas(self):
        return self.num_activas
