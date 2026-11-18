from vpython import *
from struct import pack
import os

class ExportadorRF:
    @staticmethod
    def exportar(directorio, nFrame, particulas, radioParticula):
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            
        numParticulas = len(particulas)
        name = os.path.join(directorio, f'rf_particles_{str(nFrame).zfill(5)}.bin')
        
        with open(name, 'wb') as archivo:
            ExportadorRF.crearCabecera(archivo, nFrame, numParticulas, radioParticula)
            
            for particula_i in particulas:
                posicion_i = particula_i.getPosicion()
                velocidad_i = particula_i.getVelocidad()
                
                posX, posY, posZ = posicion_i.x, posicion_i.y, posicion_i.z
                velocidadX, velocidadY, velocidadZ = velocidad_i.x, velocidad_i.y, velocidad_i.z
                
                archivo.write(pack('fff', posX, posY, posZ)) # position
                archivo.write(pack('fff', velocidadX, velocidadY, velocidadZ)) # velocity
                archivo.write(pack('fff', 0.0, 0.0, 0.0)) # force
                archivo.write(pack('fff', 0.0, 0.0, 0.0)) # vorticity
                archivo.write(pack('fff', 0.0, 0.0, 1.0)) # normal vector
                archivo.write(pack('i', len(particula_i.getVecinas()))) # number of neighbors
                archivo.write(pack('fff', 0.0, 0.0, 0.0)) # texture vector UVW
                archivo.write(pack('h', 0)) # info bits
                archivo.write(pack('f', 0.0)) # age
                archivo.write(pack('f', 0.0)) # isolation time
                archivo.write(pack('f', 0.0)) # viscosity
                archivo.write(pack('f', particula_i.getDensidad())) # density
                archivo.write(pack('f', particula_i.getPresion())) # pressure
                archivo.write(pack('f', particula_i.getMasa())) # mass
                archivo.write(pack('f', 0.0)) # temperature
                archivo.write(pack('i', 0)) # ID (uint64 pero en struct es 'q' o se pone 'i')
                
            # Cadenas finales
            archivo.write(pack('i', 0))
            archivo.write(pack('c', b'\x00'))
            archivo.write(pack('c', b'\x00'))

    @staticmethod
    def crearCabecera(archivo, nFrame, numParticulas, radioParticula):
        nombreCabecera = 'ExportadorRealFlow'
        archivo.write(pack('i', 0xFABADA))
        
        talla1 = len(nombreCabecera)
        talla2 = 250 - talla1
        archivo.write(nombreCabecera.encode())
        
        for _ in range(talla2):
            archivo.write(pack('c', b'\x00'))
            
        archivo.write(pack('h', 9))
        archivo.write(pack('f', 1.0)) # scale scene
        archivo.write(pack('i', 9)) # fluid type
        archivo.write(pack('f', 0.0)) # elapsed sim time
        archivo.write(pack('i', nFrame)) # frame number
        archivo.write(pack('i', 24)) # fps
        archivo.write(pack('l', numParticulas)) # num particles
        archivo.write(pack('f', radioParticula)) # radius
        archivo.write(pack('fff', 0.0, 0.0, 0.0)) # pressure max,min,avg
        archivo.write(pack('fff', 0.0, 0.0, 0.0)) # speed max,min,avg
        archivo.write(pack('fff', 0.0, 0.0, 0.0)) # temperature max,min,avg
        archivo.write(pack('fff', 0.0, 0.0, 0.0)) # emitter pos
        archivo.write(pack('fff', 0.0, 0.0, 0.0)) # emitter rot
        archivo.write(pack('fff', 1.0, 1.0, 1.0)) # emitter scale
