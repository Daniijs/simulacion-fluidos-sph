from vpython import *
from Particula import Particula
from SistemaParticulas import SistemaParticulas
import math

class Fuente:
    @staticmethod
    def generaColumna(posicionInicial, masa, numParticulasX, numParticulasY, numParticulasZ, separacion, radio_particula=None):
        sistema = SistemaParticulas()
        if radio_particula is None:
            radio_particula = separacion / 4.0
            
        for i in range(numParticulasX):
            for j in range(numParticulasY):
                for k in range(numParticulasZ):
                    pos = posicionInicial + vector(i * separacion, j * separacion, k * separacion)
                    p = Particula(pos=pos, velocity=vector(0,0,0), radius=radio_particula, masa=masa, color=color.cyan)
                    sistema.addParticula(p)
                    
        return sistema

    @staticmethod
    def generaCapaCircular(posicionInicial, masa, radio_columna, separacion, radio_particula, theta_offset=0.0, velocidadInicial=vector(0,0,0), altura_relativa=0.0, visual=True):
        # Genera una única capa circular para inserción progresiva (capa a capa)
        sistema = SistemaParticulas()
        num_particulas_radio = int(radio_columna / separacion)
        
        for r_idx in range(num_particulas_radio + 1):
            r = r_idx * separacion
            if r == 0:
                pos = posicionInicial + vector(0, altura_relativa, 0)
                p = Particula(pos=pos, vel=velocidadInicial, radius=radio_particula, masa=masa, visual=visual)
                sistema.addParticula(p)
            else:
                num_angular = int((2 * math.pi * r) / separacion)
                for a in range(num_angular):
                    theta = a * (2 * math.pi / num_angular) + theta_offset
                    x = r * math.cos(theta)
                    z = r * math.sin(theta)
                    pos = posicionInicial + vector(x, altura_relativa, z)
                    p = Particula(pos=pos, vel=velocidadInicial, radius=radio_particula, masa=masa, visual=visual)
                    sistema.addParticula(p)
        return sistema

    @staticmethod
    def calculaNumParticulasColumnaCircular(capas, radio_columna, separacion):
        num_particulas_radio = int(radio_columna / separacion)
        particulas_por_capa = 0
        for r_idx in range(num_particulas_radio + 1):
            r = r_idx * separacion
            if r == 0:
                particulas_por_capa += 1
            else:
                num_angular = int((2 * math.pi * r) / separacion)
                particulas_por_capa += num_angular
        return particulas_por_capa * capas

    @staticmethod
    def generaColumnaCircularCapaCapa(posicionInicial, masa, capas, radio_columna, separacion, radio_particula, theta_step=math.pi/10, velocidadInicial=vector(0,0,0)):
        # Genera una columna completa aplicando una rotación progresiva a cada capa
        sistema = SistemaParticulas()
        for j in range(capas):
            offset = j * theta_step
            capa = Fuente.generaCapaCircular(posicionInicial, masa, radio_columna, separacion, radio_particula, offset, velocidadInicial, j * separacion)
            for p in capa.getListaParticulas():
                sistema.addParticula(p)
        return sistema
