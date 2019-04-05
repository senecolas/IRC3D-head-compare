"""
BoundingBox.py
BoundingBox class
"""

class BoundingBox():
  def __init__(self):
    """
    Constructor of the BoundingBox
    """
    self.A = [0,0,0]
    self.B = [0,0,0]
    self.C = [0,0,0]
    self.D = [0,0,0]
    self.E = [0,0,0]
    self.F = [0,0,0]
    self.G = [0,0,0]
    self.H = [0,0,0]

  #################################
  ### ====      INIT       ==== ###
  #################################

  def setVertexes(self, xMin, xMax, yMin, yMax, zMin, zMax):
    self.A = [xMin,yMax,zMax]
    self.B = [xMax,yMax,zMax]
    self.C = [xMax,yMax,zMin]
    self.D = [xMin,yMax,zMin]
    self.E = [xMin,yMin,zMax]
    self.F = [xMax,yMin,zMax]
    self.G = [xMax,yMin,zMin]
    self.H = [xMin,yMin,zMin]