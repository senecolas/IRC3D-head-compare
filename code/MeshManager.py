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

from PIL import Image
import numpy as np

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
    self.brightness = 0
    self.contrast = 0
    self.hue = 0
    self.saturation = 1
    self.temperature = 0

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

    # Convert to OpenCV Image
    size = rgbImage.get_image_data().width, rgbImage.get_image_data().height
    pilImage = Image.frombuffer('RGBA', size, rgbImage.get_image_data().data, 'raw', 'RGBA', 0, 1)
    open_cv_image = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGBA2RGB)
    open_cv_image = cv2.flip(open_cv_image, 0)

    """ Filters """
    # Saturation
    open_cv_image = self.applySaturation(open_cv_image, self.saturation)
    # Hue
    open_cv_image = self.applyHue(open_cv_image, self.hue)
    # Brightness / contrast
    open_cv_image = self.applyBrightnessContrast(open_cv_image, self.brightness, self.contrast)
    # Temperature
    open_cv_image = self.applytemperature(open_cv_image, self.temperature)
    
    return open_cv_image
      

  #################################
  ### ====     DRAWING     ==== ###
  #################################

  def applyBrightnessContrast(self, input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

  def applyHue(self, input_img, hueshift = 0):
    hsvImg = cv2.cvtColor(input_img,cv2.COLOR_RGB2HSV)
    if hueshift >= 0:
      hsvImg[...,0] += hueshift
    else:
      hsvImg[...,0] -= abs(hueshift)
    return cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

  def applySaturation(self, input_img, sat_factor = 1):
    hsvImg = cv2.cvtColor(input_img,cv2.COLOR_RGB2HSV)
    hsvImg[...,1] *= sat_factor
    return cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

  def applytemperature(self, input_img, temperature = 0):
    labImg = cv2.cvtColor(input_img,cv2.COLOR_RGB2LAB)
    if temperature >= 0:
      labImg[...,2] += temperature
    else :
      labImg[...,2] -= abs(temperature)
    return cv2.cvtColor(labImg,cv2.COLOR_LAB2RGB)

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