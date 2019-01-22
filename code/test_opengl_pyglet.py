import ctypes

import pyglet
from pyglet.gl import *

from pywavefront import visualization
from pywavefront import Wavefront

import os
import math

# Mesh load and OpenGL context init
mesh = Wavefront('../face orientation/visage.obj')
viewport = (1000,1000)
window = pyglet.window.Window(viewport[0],viewport[1], caption='Mesh orientation', resizable=True)
lightfv = ctypes.c_float * 4
rx, ry, rz = (0,0,0)

# Orientation file load (output.txt : NOT TO USE)
# filename = 'output.txt'
# directory = os.path.join(os.getcwd(),'../output')
# fullpath = os.path.join(directory, filename)

# try:
#     file = open(fullpath, "r+")
# except IOError:
#     print("Could not open file !")

# content = file.readlines()
# file.close()

@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(90.0, width/float(height), 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_draw():

    window.clear()

    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-40.0, 200.0, 100.0, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_MODELVIEW)

    # Rotations for sphere on axis - useful
    glTranslated(0, 0, -2)
    glRotatef(ry, 1, 0, 0)
    glRotatef(rx, 0, 1, 0)
    glRotatef(rz, 0, 0, 1)

    visualization.draw(mesh)

def update(dt):
    # In case we have something to change on the mesh (like rotation for example)
    global rx, ry, rz
    return

pyglet.clock.schedule(update)
pyglet.app.run()