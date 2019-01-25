# Face.py
# Face class and functions

import cv2
import json
from math import cos
from math import sin
import numpy as np

# class that manages the information of a face on a given frame
class Face():
  def __init__(self, confidence=1., x_min=-1, x_max=-1, y_min=-1, y_max=-1, yaw=0., pitch=0., roll=0.):
    self.confidence = confidence
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.yaw = float(yaw)
    self.pitch = float(pitch)
    self.roll = float(roll)
  
  # save on a txt file by overwriting the file (for blender test)
  def save(self, path):
    txt_out = open(path, 'w')
    txt_out.write('%f %f %f\n' % (self.yaw, self.pitch, self.roll))
    print("Face orientation save on", path)
  
  
  def getJSONData(self):
    return {
      "confidence": self.confidence,
      "position": {
        "x_min": self.x_min,
        "x_max": self.x_max,
        "y_min": self.y_min,
        "y_max": self.y_max,
      },
      "yaw": self.yaw,
      "pitch": self.pitch,
      "roll": self.roll,
    }
  
  def setJSONData(self, data):
    self.confidence = data['confidence']
    self.x_min = data['position']['x_min']
    self.x_max = data['position']['x_max']
    self.y_min = data['position']['y_min']
    self.y_max = data['position']['y_max']
    self.yaw = data['yaw']
    self.pitch = data['pitch']
    self.roll = data['roll']
    return self
    
  def drawAxis(self, img):
    pitch = self.pitch * np.pi / 180
    yaw = -(self.yaw * np.pi / 180)
    roll = self.roll * np.pi / 180

    tdx = (self.x_min + self.x_max) / 2
    tdy = (self.y_min + self.y_max) / 2
    size = abs(self.y_max - self.y_min) / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
 
    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

  def drawSquare(self, img):
    cv2.rectangle(img, (int(self.x_min), int(self.y_min)), (int(self.x_max), int(self.y_max)), (0, 0, 255), 3)
    return img
    

