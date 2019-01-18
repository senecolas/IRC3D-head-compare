# visage.py
# Visage class and functions

import json

# class that manages the information of a face on a given frame
class Visage():
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
    print("Visage orientation save on", path)
  
  
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

