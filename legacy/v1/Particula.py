from vpython import *

class Particula:
    def __init__(self, pos, vel, masa, radius, visual=True):
        self.pos = pos
        self.vel = vel
        self.masa = masa
        self.radius = radius
        self.fuerza = vector(0, 0, 0)
        self.densidad = 0.0
        self.presion = 0.0
        self.vecinas = []
        
        self.visual = visual
        if self.visual:
            self.sphere = sphere(pos=pos, radius=radius, color=color.cyan, opacity=0.8)
        
    def actualizaVisual(self):
        if self.visual:
            self.sphere.pos = self.pos
        
    def getPosicion(self):
        return self.pos
        
    def setPosicion(self, pos):
        self.pos = pos
        
    def getVelocidad(self):
        return self.vel
        
    def setVelocidad(self, vel):
        self.vel = vel
        
    def getRadio(self):
        return self.radius
        
    def getMasa(self):
        return self.masa
        
    def getFuerza(self):
        return self.fuerza
        
    def setFuerza(self, fuerza):
        self.fuerza = fuerza
        
    def resetFuerza(self):
        self.fuerza = vector(0,0,0)
        
    def addFuerza(self, fuerza):
        self.fuerza = self.fuerza + fuerza
        
    def getDensidad(self):
        return self.densidad
        
    def setDensidad(self, densidad):
        self.densidad = densidad
        
    def getPresion(self):
        return self.presion
        
    def setPresion(self, presion):
        self.presion = presion
        
    def getVecinas(self):
        return self.vecinas
        
    def setVecinas(self, vecinas):
        self.vecinas = vecinas
        
    def clearVecinas(self):
        self.vecinas = []
