# visage.py
# Visage class and functions

# class that manages the information of a face on a given frame
class Visage():
  def __init__(self, confidence, x_min, x_max, y_min, y_max, yaw, pitch, roll):
    self.confidence = confidence
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.yaw = yaw
    self.pitch = pitch
    self.roll = roll
    
  def save(self, path):
    txt_out = open(path, 'w')
    txt_out.write('%f %f %f\n' % (self.yaw, self.pitch, self.roll))
    print("Visage orientation save on", path)

