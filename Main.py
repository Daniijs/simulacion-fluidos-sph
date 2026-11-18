from vpython import *
from Particula import Particula
from SistemaParticulas import SistemaParticulas
from Fuente import Fuente
from SpatialHashing import SpatialHashing
from Dinamica import Dinamica
from GestorColision import GestorColision
from MetodoEuler import MetodoEuler
from ExportadorPOV import ExportadorPOV
from ExportadorRF import ExportadorRF
import os
import math
import ctypes
import numpy as np

# ================= Optimización de Sistema (Windows Priority) =================
if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)

# ================= Parámetros de Simulación OPTIMIZADA (~3000 Partículas) =================
separacion = 0.02
masa = 0.008 
densidad_reposo = 1000.0
viscosidad = 0.4 
k_presion = 5000.0 
h = 0.06 # Radio optimizado para velocidad
dt = 0.0005
sub_steps = 10 # Pasos optimizados para simular 0.005s por frame (10 * 0.0005s) y llegar a estabilidad en 600 frames

radio_particula = 0.012 
gravedad = vector(0, -9.81, 0)

# ================= Entorno Virtual =================
LimitesX = [-1.2, 3.2]
LimitesY = [0.0, 1.8]
LimitesZ = [-0.7, 0.7]
espesor = 0.02
esfera_virtual = {'pos': vector(-0.4, 0.6, 0), 'radius': 0.4}

# Posición de la fuente: Recta y centrada encima de la esfera en el lado izquierdo
posicion_grifo = vector(-0.4, 1.6, 0)
velocidad_grifo = vector(0, -1.2, 0)

# ================= Inicialización =================
sistema = SistemaParticulas(capacidad_inicial=10000)
hashing = SpatialHashing(cell_size=h)
dinamica = Dinamica(sistema, h=h, densidad_reposo=densidad_reposo, viscosidad=viscosidad, k_presion=k_presion)

capa_actual = 0
capas_totales = 25
intervalo_capa = (separacion / mag(velocidad_grifo)) * 0.95 
tiempo_capa = intervalo_capa 
theta_step = math.pi / 8 
particulas_estimadas = Fuente.calculaNumParticulasColumnaCircular(capas_totales, 0.14, separacion)

print(f"--- MOTOR SPH VECTORIZADO (NUMPY) INICIADO ---")
print(f"Objetivo: {particulas_estimadas} partículas | Aceleración: Hardware Vectorized")
print(f"Posición de la fuente: {posicion_grifo} | Velocidad calculada: {velocidad_grifo}")
frame_count = 0
tiempo_total = 0.0
export_dir = "POV_Frames"

# Bucle principal
while frame_count < 2000:
    for _ in range(sub_steps):
        # Generación
        if capa_actual < capas_totales and tiempo_capa >= intervalo_capa:
            orientacion = capa_actual * theta_step
            nueva_capa = Fuente.generaCapaCircular(
                posicionInicial=posicion_grifo,
                masa=masa,
                radio_columna=0.14,
                separacion=separacion,
                radio_particula=radio_particula,
                theta_offset=orientacion,
                velocidadInicial=velocidad_grifo,
                visual=False 
            )
            for p in nueva_capa.getListaParticulas():
                sistema.addParticula(p)
            capa_actual += 1
            tiempo_capa = 0
            
        if sistema.num_activas > 0:
            # SPH Vectorizado
            hashing.update(sistema.pos, sistema.num_activas)
            dinamica.calculaFuerzaSPH(hashing)
            MetodoEuler.integra(sistema, dt, gravedad)
            GestorColision.colisionLimites(LimitesX, LimitesY, LimitesZ, espesor, sistema)
            GestorColision.colisionEsfera(esfera_virtual['pos'], esfera_virtual['radius'], sistema)
                
        tiempo_total += dt
        tiempo_capa += dt
        
    # Exportación Pro
    limites_escena = {'min': vector(LimitesX[0], LimitesY[0], LimitesZ[0]), 'max': vector(LimitesX[1], LimitesY[1], LimitesZ[1])}
    
    class MockObstaculo:
        def __init__(self, d):
            self.pos = d['pos']
            self.radius = d['radius']
    
    ExportadorPOV.exportar(export_dir, frame_count, sistema, radio=radio_particula, limites_escena=limites_escena, obstaculo=MockObstaculo(esfera_virtual))
    
    # Exportación RealFlow (Para importación y renderizado en RealFlow/Maxwell para 0.5 puntos extra)
    particulas_list = sistema.getListaParticulas()
    ExportadorRF.exportar("RF_Frames", frame_count, particulas_list, radio_particula)
    
    if frame_count % 10 == 0:
        print(f"Progress: Frame {frame_count}/2000 | Particles: {sistema.num_activas} | CPU Load: High")
        
    frame_count += 1

print("--- SIMULACIÓN FINALIZADA ---")
