"""
MeshManager.py
Mesh manager class
"""

from Face import Face
from FaceDetector import FaceDetector
from PyQt5 import QtGui
import ctypes
import cv2
import json
import os
import pyglet
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront import visualization
from pywavefront import Wavefront
from pyglet.gl.gl_info import GLInfo

class MeshManager():
  def __init__(self, meshPath=""):
    """
    Constructor of the MeshManager
    """
    self.meshPath = meshPath
    self.mesh = None
    self.isModelLoaded = False
    self.viewWidth = 0
    self.viewHeight = 0
    self.glWindow = None
    self.lightfv = None
    self.fovy = 20
    self.rx = 0
    self.ry = 0
    self.rz = 0

    if meshPath != "":
      self.load(meshPath)

  #################################
  ### ====     LOADING     ==== ###
  #################################

  def load(self, filename):
    self.mesh = Wavefront(filename)
    
  def initGL(self):
    self.isModelLoaded = True
    self.glWindow = pyglet.window.Window(self.viewWidth, self.viewHeight, caption='Mesh orientation', resizable=True)
    self.lightfv = ctypes.c_float * 4

    self.glWindow.set_visible(False)
    glMatrixMode(GL_PROJECTION)
    
    # OpenGL INFO
    info = GLInfo()
    info.set_active_context()

    print("-- OpenGL use the " + str(info.get_renderer()))

  #################################
  ### ====      TESTS      ==== ###
  #################################  

  def isLoaded(self):
    """ Returns true if a video is loaded """
    if self.mesh == None:
      return False
    return True
    
  #################################
  ### ====     ACTIONS     ==== ###
  #################################

  # Meshes positionning on the 3d scene
  # FOV management (slide bar)

  def frame(self, faces):

    if self.isModelLoaded == False:
      return
    glMatrixMode(GL_PROJECTION)
    self.glWindow.clear()
    glLoadIdentity()
    gluPerspective(self.fovy, self.viewWidth / float(self.viewHeight), 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

    glLightfv(GL_LIGHT0, GL_POSITION, self.lightfv(-20.0, 50.0, 25.0, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, self.lightfv(0.5, 0.5, 0.5, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, self.lightfv(0.8, 0.8, 0.8, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_MODELVIEW)

    mesh_cpt = 0
    if len(faces) == 0:
      self.drawMesh(0, 0, 0, 0, 1)
    else:
      for face in faces:
        self.drawMesh(face.yaw, face.pitch, face.roll, mesh_cpt, len(faces))
        mesh_cpt += 1

    # To check color buffer (then compare it with the pixmap after conversion) #
    # pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')

    rgbImage = pyglet.image.get_buffer_manager().get_color_buffer()

    # Convert ColorBuffer to Pixmap
    convertToQtFormat = QtGui.QImage(rgbImage.get_image_data().data, rgbImage.get_image_data().width, rgbImage.get_image_data().height, QtGui.QImage.Format_RGBA8888_Premultiplied)
    # convertToQtFormat.save('qImage_screenshot.png')
    convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)

    pixmap = QtGui.QPixmap(convertToQtFormat)

    # Mirror the pixmap (because of an gorizontal "mirror effect" during convert)
    sm = QtGui.QTransform()
    sm.scale(1, -1)
    pixmap = pixmap.transformed(sm)

    return pixmap
      

  #################################
  ### ====     DRAWING     ==== ###
  #################################

  def drawMesh(self, yaw, pitch, roll, mesh_number, nb_meshes):
    if self.mesh != None:
      x_start = -((nb_meshes - 1) * 0.125)
      x, y, z = (x_start + 0.25 * mesh_number, -2, 0)

      # Transforms : comparisons between face bounding boxes( on the video) and the bounding box of the face on OpenGL render --> y position : we move the mesh back until bounding boxes are "almost even"
      # then we move the mesh on x and z axes to make the bounding box at the same pos
      # BUT --> this involve a bounding box on the pixmap frame wich follow the face

      # Reset previous matrix transformations
      glLoadIdentity()

      # Rotations for sphere on axis - useful
      glTranslated(x, z, y)
      glRotatef(-pitch - self.rx, 1, 0, 0) # sounds like pitch on x axis -> red on the schema
      glRotatef(-yaw - self.rz, 0, 1, 0) # sounds like yaw on z axis -> green on schema
      glRotatef(-roll - self.ry, 0, 0, 1) # sounds like roll on y axis -> blue on schema

      visualization.draw(self.mesh)
    else:
      print ("No mesh loaded : can't draw the mesh")