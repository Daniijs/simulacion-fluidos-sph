import numpy as np

class SpatialHashing:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
        self.particle_keys = None

    def _get_key(self, pos):
        return (int(pos[0] / self.cell_size),
                int(pos[1] / self.cell_size),
                int(pos[2] / self.cell_size))

    def update(self, pos_array, num_activas):
        self.grid = {}
        self.particle_keys = None
        if num_activas == 0: return
        
        # Cálculo de keys vectorizado
        self.particle_keys = (pos_array[:num_activas] / self.cell_size).astype(np.int32)
        
        for i in range(num_activas):
            key = (self.particle_keys[i, 0], self.particle_keys[i, 1], self.particle_keys[i, 2])
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(i)

    def get_celdas_vecinos(self):
        # Optimización masiva: Calcula la vecindad de 27 celdas una sola vez por celda activa,
        # en lugar de hacerlo 27 veces por cada partícula individual.
        celdas_vecinos = {}
        for key in self.grid.keys():
            vecinos = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        if neighbor_key in self.grid:
                            vecinos.extend(self.grid[neighbor_key])
            celdas_vecinos[key] = vecinos
        return celdas_vecinos

    def get_vecinos_indices(self, pos_i, h):
        # Mantenemos este método para compatibilidad si otros módulos lo usan
        key_i = self._get_key(pos_i)
        vecinos = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    key = (key_i[0] + dx, key_i[1] + dy, key_i[2] + dz)
                    if key in self.grid:
                        vecinos.extend(self.grid[key])
        return vecinos
