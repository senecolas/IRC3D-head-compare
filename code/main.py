from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import argparse
import cv2
import datasets
import dlib
import dlib.cuda as cuda
from frame import *
from hopenet import *
import json
import os
import sys
import time
import timeit
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from ui import main
import utils
from visage import Visage

# python3 code/main.py --video ./videos/CCTV_1.mp4 --output ./output/output.txt

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
  parser.add_argument('--video', dest='video_path', help='Path of video', default='')
  parser.add_argument('--config', dest='config_path', help='Path of the configuraiton JSON file', default='../config.json')
  parser.add_argument('--output', dest='output', help='Path and name to output file', default="../output/output.txt")
  args = parser.parse_args()
  return args

class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent=parent)
    self.setupUi(self)
    self.isPlaying = False
    self.isVideoLoaded = False
    self.isLoadedData = False
    self.ret = True
    self.videoFPS = 25
    self.actualFrame = 0
    self.isDragging = False
    self.lastUpdate = 0
    self.conf_threshold = 0.75
    self.maxWidth = self.VideoWidget.geometry().width()
    self.maxHeight = self.VideoWidget.geometry().height()
    self.centerX = self.maxWidth / 2. # center of the video
    self.centerY = self.maxHeight / 2.
    self.zoom = 1.
    self.mousePos = QtCore.QPointF(0, 0)
    self.output_path = "output.txt"
    self.read_PB.clicked.connect(lambda: self.play())
    self.pause_PB.clicked.connect(lambda: self.pause())
    self.nextFrame_PB.clicked.connect(lambda: self.moveFrame(1))
    self.lastFrame_PB.clicked.connect(lambda: self.moveFrame(-1))
    self.headPosition_PB.clicked.connect(lambda: self.getHeadPosition())
    self.actionOpen_video.triggered.connect(self.selectVideo)
    self.videoSlider.actionTriggered.connect(lambda: self.sliderChanged())
    self.VideoWidget.wheelEvent = self.wheelEvent
    self.VideoWidget.mousePressEvent = self.mousePressEvent
    self.VideoWidget.mouseReleaseEvent = self.mouseReleaseEvent
    self.VideoWidget.mouseMoveEvent = self.mouseMoveEvent
    
  def mousePressEvent(self, event):
    self.mousePos = event.pos()
    self.isDragging = True
  
  def mouseReleaseEvent(self, event):
    self.isDragging = False
    
  def mouseMoveEvent(self, event):
    if (self.isDragging):
      self.centerX -= (event.pos().x() - self.mousePos.x()) 
      self.centerY -= (event.pos().y() - self.mousePos.y())
      self.drawFrame() #we redraw the frame
    self.mousePos = event.pos()
  
  def sliderChanged(self):
    self.isPlaying = False
    if(self.videoSlider.value() + 1 == self.actualFrame):
      return
    self.setFrame(self.videoSlider.value() + 1)
    
  def wheelEvent(self, event):
    self.zoom = utils.clamp(1., self.zoom + event.angleDelta().y() * 0.002, 20.)
    if(self.zoom == 1):
      self.centerX = self.maxWidth / 2. # center of the video
      self.centerY = self.maxHeight / 2.
    else:
      self.centerX = event.pos().x() * self.zoom
      self.centerY = event.pos().y() * self.zoom
    self.drawFrame() #we redraw the frame
  
  def getHeadPosition(self):
    if(self.isVideoLoaded == False or self.isLoadedData == False):
      return
    visages = getFrameVisages(self.frame, self.hopenetModel, self.cnn_face_detector, self.transformations, self.conf_threshold, self.gpu_id)
    for vis in visages:
      print("VISAGE : ", float(vis.yaw), float(vis.pitch), float(vis.roll)) 
      vis.save(self.output_path)
    
  def play(self):
    if(self.isVideoLoaded == False):
      return
    self.isPlaying = True
    #if the video is end, we restart
    if(self.ret == False):
      self.video.set(1, 0)
      
  def pause(self):
    self.isPlaying = False
    
  def selectVideo(self):
    fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Video')[0]
    self.loadVideo(str(fileName))
    
  def loadVideo(self, fileName):
    self.videoPath = fileName
    self.video = cv2.VideoCapture(self.videoPath)
    if(not self.video.isOpened()):
      print("ERROR : Can't load", self.videoPath)
      return 
    self.videoFPS = int(self.video.get(cv2.CAP_PROP_FPS))
    self.frameCount = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
    self.videoSlider.setMaximum(self.frameCount - 1) # we set the slider
    self.isVideoLoaded = True
    self.setFrame(0) # we draw the first frame
    self.cachePath = self.videoPath + self.cache_string + ".json"
    self.loadCacheFile() #we load or create the cache file
    
  def moveFrame(self, num):
    if(self.isVideoLoaded == False):
      return
    self.isPlaying = False
    self.actualFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.setFrame(self.actualFrame + num - 1)
    
  def setFrame(self, frameNum):
    if(self.isVideoLoaded == False):
      return
    self.video.set(1, frameNum)
    self.actualFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.videoSlider.setValue(self.actualFrame - 1)
    self.drawNextFrame()
  
  def drawNextFrame(self):
    if(self.isVideoLoaded == False):
      return
    self.ret, frame = self.video.read()
    if (self.ret == True):
      # Draw frame
      self.frame = frame
      self.drawFrame()
      # Update slider
      self.actualFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
      self.videoSlider.setValue(self.actualFrame-1)

    else:
      self.isPlaying = False
  
  # Draw the actual frame on the screen
  def drawFrame(self):
    if(self.isVideoLoaded == False):
      return
    
    # convert cv2 video to QPixmap
    rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QtGui.QImage.Format_RGB888)
    convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
    pixmap = QtGui.QPixmap(convertToQtFormat)
    
    # Resize with zoom
    pixmap = pixmap.scaled(self.maxWidth * self.zoom, self.maxHeight * self.zoom, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    
    # Add image
    scene = QtWidgets.QGraphicsScene() 
    scene.addItem(QtWidgets.QGraphicsPixmapItem(pixmap))
    self.VideoWidget.setScene(scene) 
    
    #Center on the appropriate position
    self.VideoWidget.centerOn(self.centerX, self.centerY)
    

  def update(self):
    # PLAY THE VIDEO
    if(self.isPlaying):
      self.drawNextFrame()
      
      # FPS CONTROLLER
      elapsedTime = timeit.default_timer() - self.lastUpdate
      if (elapsedTime < 1. / self.videoFPS): 
        time.sleep(1. / self.videoFPS - elapsedTime)

    self.lastUpdate = timeit.default_timer()
    
  def loadConfig(self, jsonFile):
    if not os.path.exists(jsonFile):
      sys.exit('ERROR : Configuration file does not exist')
    with open(jsonFile) as json_file:
      data = json.load(json_file)
      self.snapshot = data['snapshot']
      self.face_model = data['face_model']
      self.gpu_id = data['gpu_id']
      self.conf_threshold = data['conf_threshold']
      self.cache_string = data['cache_string']
      
  def loadCacheFile(self):
    if not os.path.exists(self.cachePath):
      self.initCacheFile()
    else:
      with open(self.cachePath) as json_cacheFile:
        self.cacheData = json.load(json_cacheFile)
        print(self.cacheData)
    
  def initCacheFile(self):
    self.cacheData = {"data": []}
    for i in range(int(self.frameCount)):
      self.cacheData["data"].append({"isLoaded": False})
    self.saveCacheFile()
      
  def saveCacheFile(self):
    with open(self.cachePath, 'w', encoding='utf-8') as outfile:
      json.dump(self.cacheData, outfile)

  def loadData(self):
    cudnn.enabled = True
    
    print(torch.cuda.get_device_name(self.gpu_id))
    
    # ResNet50 structure
    self.hopenetModel = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    dlib.DLIB_USE_CUDA = 1

    # Dlib face detection model
    self.cnn_face_detector = dlib.cnn_face_detection_model_v1(self.face_model)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(self.snapshot)
    self.hopenetModel.load_state_dict(saved_state_dict)

    print ('Loading data.')

    self.transformations = transforms.Compose([transforms.Scale(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    
    self.hopenetModel.cuda(self.gpu_id)

    print ('Ready to test network.')
    self.isLoadedData = True

    # Test the Model
    self.hopenetModel.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    #torch.no_grad()


if __name__ == '__main__':
  args = parse_args()
 
  
  app = QtWidgets.QApplication(sys.argv)
  window = MainWindow()
  window.show()
  
  window.loadConfig(args.config_path)
  window.loadData()
  window.output_path = args.output
  
  if(args.video_path != ""): 
    window.loadVideo(args.video_path);
  
  timer = QtCore.QTimer()
  timer.timeout.connect(window.update)
  timer.start(10)

  sys.exit(app.exec_())