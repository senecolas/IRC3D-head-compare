from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import argparse
import cv2
import datasets
import dlib
import dlib.cuda as cuda
from face import Face
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

# python3 code/main.py --video ./videos/CCTV_1.mp4 --output ./output/output.txt

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
  parser.add_argument('--video', dest='video_path', help='Path of video', default='')
  parser.add_argument('--config', dest='config_path', help='Path of the configuraiton JSON file', default='../config.json')
  parser.add_argument('--output', dest='output', help='Path and name to output file', default="../output/output.txt")
  args = parser.parse_args()
  return args

class MainWindow(main.Ui_MainWindow, QtWidgets.QMainWindow):
  def __init__(self, parent=None):
    # INIT QT WINDOWS
    super(MainWindow, self).__init__(parent=parent)
    self.setupUi(self)
    
    # DEFAULT VARIABLES
    self.isPlaying = False
    self.isVideoLoaded = False
    self.isLoadedData = False
    self.ifDrawAxis = False
    self.ifDrawSquare = False
    self.ret = True
    self.videoFPS = 25
    self.currentFrame = 0
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
    
    # PUSH BUTTONS EVENTS
    self.read_PB.clicked.connect(lambda: self.play())
    self.pause_PB.clicked.connect(lambda: self.pause())
    self.nextFrame_PB.clicked.connect(lambda: self.moveFrame(1))
    self.lastFrame_PB.clicked.connect(lambda: self.moveFrame(-1))
    self.next10Frame_PB.clicked.connect(lambda: self.moveFrame(10))
    self.last10Frame_PB.clicked.connect(lambda: self.moveFrame(-10))
    self.headPosition_PB.clicked.connect(lambda: self.getHeadPosition())
    self.coordinates_PB.clicked.connect(lambda: self.drawAxisEvent())
    self.square_BT.clicked.connect(lambda: self.drawSquareEvent())
    
    # MENU BAR EVENTS
    self.actionOpen_video.triggered.connect(self.selectVideo)
    self.actionOpen_model.triggered.connect(self.selectModel)
    
    # SLIDERS EVENTS
    self.videoSlider.actionTriggered.connect(lambda: self.sliderChanged())
    self.confidenceSlider.actionTriggered.connect(lambda: self.confidenceChanged())
    
    # MOUSE EVENTS
    self.VideoWidget.wheelEvent = self.wheelEvent
    self.VideoWidget.mousePressEvent = self.mousePressEvent
    self.VideoWidget.mouseReleaseEvent = self.mouseReleaseEvent
    self.VideoWidget.mouseMoveEvent = self.mouseMoveEvent
    self.VideoWidget.resizeEvent = self.resizeEvent


  
  #################################
  ### ==== EVENTS ON FRAME ==== ###
  #################################
 
  def mousePressEvent(self, event):
    """ Event call at each click on the VideoWidget. Get the mouse pos and active dragging """
    self.mousePos = event.pos()
    self.isDragging = True
  
  
  def mouseReleaseEvent(self, event):
    """ Event call at each realse click on the VideoWidget. Disable dragging """
    self.isDragging = False


  def mouseMoveEvent(self, event):
    """ Event call at each mouse movement on VideoWidget. Change the center with the dragging """
    if (self.isDragging):
      self.centerX -= (event.pos().x() - self.mousePos.x()) 
      self.centerY -= (event.pos().y() - self.mousePos.y())
      self.drawFrame() #we redraw the frame
    self.mousePos = event.pos()
    
    
  def wheelEvent(self, event):
    """ Event call at each mouse wheel action on videoSlider. Zoom, change the center and redraw the frame """
    self.zoom = utils.clamp(1., self.zoom + event.angleDelta().y() * 0.002, 20.)
    if(self.zoom == 1):
      self.centerX = self.maxWidth / 2. # center of the video
      self.centerY = self.maxHeight / 2.
    else:
      self.centerX = event.pos().x() * self.zoom
      self.centerY = event.pos().y() * self.zoom
    self.drawFrame() #we redraw the frame



  #################################
  ### ==== EVENTS ON MENU  ==== ###
  #################################
  
  def resizeEvent(self, event):
    """ Event call at each resize. Resize the window and update the VideoWidget """
    self.maxWidth = self.VideoWidget.geometry().width()
    self.maxHeight = self.VideoWidget.geometry().height()
    self.drawFrame() #we redraw the frame
 
 
  def drawAxisEvent(self):
    """ Event call with the push of coordinates_PB. Active or disable the draw of axis and redraw the frame """
    if(self.ifDrawAxis):
      self.ifDrawAxis = False
    else:
      self.ifDrawAxis = True
    self.drawFrame() #we redraw the frame
    
    
  def drawSquareEvent(self):
    """ Event call with the push of square_BT. Active or disable the draw of square and redraw the frame """
    if(self.ifDrawSquare):
      self.ifDrawSquare = False
    else:
      self.ifDrawSquare = True
    self.drawFrame() #we redraw the frame
  
  
  def sliderChanged(self):
    """ Event call at each videoSlider changements. Set the new frame """
    self.isPlaying = False
    if(self.videoSlider.value() + 1 == self.currentFrame): # if the value is the same, we not reset the frame
      return
    self.setFrame(self.videoSlider.value() + 1)
    
    
  def confidenceChanged(self):
    """ Event call at each confidenceSlider changements. Change the conf_threshold and redraw the frame """
    self.conf_threshold = self.confidenceSlider.value() / 100.
    self.confidenceInfo.setText("{0:.2f}".format(self.conf_threshold))
    self.drawFrame() #we redraw the frame
    
    
  def selectVideo(self):
    """ Event call at each click on actionOpen_video. Opens a video selection window and load the selected video """
    fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Video')[0]
    self.loadVideo(str(fileName))
  
  
  def selectModel(self):
    """ Event call at each click on actionOpen_model. Opens a mesh selection window and load the selected mesh """
    fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Select 3D Model')[0]
    self.loadModel(str(fileName))



  #################################
  ### ====     ACTIONS     ==== ###
  #################################
      
  def play(self):
    """ Start the video if it's loaded. If the video is end, restart the video """  
    if(self.isVideoLoaded == False):
      return
    self.isPlaying = True
    #if the video is end, we restart
    if(self.ret == False):
      self.video.set(1, 0)
      
      
  def pause(self):
    """ Pause the video """  
    self.isPlaying = False   
    
    
  def moveFrame(self, num):
    """ Moves the current frame by 'num' frames """  
    if(self.isVideoLoaded == False):
      return
    self.isPlaying = False
    self.currentFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.setFrame(self.currentFrame + num - 1)
    
    
  def setFrame(self, frameNum):
    """ Set the current frame on 'frameNum' """ 
    if(self.isVideoLoaded == False):
      return
    self.video.set(1, frameNum)
    self.currentFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.videoSlider.setValue(self.currentFrame - 1)
    self.drawNextFrame()


  def getHeadPosition(self):
    """ Launch the calculation to get the faces of the current frame and saves it in the cache file """
    
    if(self.isVideoLoaded == False or self.isLoadedData == False):
      return self.getCurrentCacheData()
    if(self.getCurrentCacheData()['isLoaded']):
      print("Face already loaded")
      return self.getCurrentCacheData()['faces']
    
    # Get faces
    faces = getFrameFaces(self.frame, self.hopenetModel, self.cnn_face_detector, self.transformations, self.gpu_id)
    for vis in faces:
      self.getCurrentCacheData()['faces'].append(vis.getJSONData())
      vis.save(self.output_path)
    
    # Updated cache, info and processTable
    self.getCurrentCacheData()['isLoaded'] = True
    self.saveCacheFile()
    self.updateInfo()
    self.updateProcessTable(self.currentFrame-1)
    
    # Redraw the frame
    self.drawFrame()
    
    # return 
    return self.getCurrentCacheData()['faces']


  def saveCacheFile(self):
    """ Update the cache file """
    with open(self.cachePath, 'w', encoding='utf-8') as outfile:
      json.dump(self.cacheData, outfile)



  #################################
  ### ====     GETTERS     ==== ###
  #################################

  def getCurrentCacheData(self):
    """ Return the data of the current frame """
    return self.getCacheData(int(self.currentFrame - 1))
  
  
  def getCacheData(self, index):
    """ Return the data of the frame corresponding to the 'index' (be careful: the index starts at 0 and the frame at 1) """
    return self.cacheData['data'][index]
    
    
  def getCurrentFaces(self):
    """ Return an array of current faces on the frame taking into account the conf_threshold (empty if not process or if no face has been detected with this conf_threshold) """
    res = []
    if(self.getCurrentCacheData()['isLoaded'] == False):
      return res
    for face in self.getCurrentCacheData()['faces']:
      face = Face().setJSONData(face)
      if(face.confidence > self.conf_threshold):
        res.append(face)
    return res
  
  
  
  #################################
  ### ====     LOADING     ==== ###
  #################################
  
  def loadConfig(self, jsonFile):
    """ Loads the configuration from a JSON file """
    if not os.path.exists(jsonFile):
      sys.exit('ERROR : Configuration file does not exist')
    with open(jsonFile) as json_file:
      data = json.load(json_file)
      self.snapshot = data['snapshot']
      self.face_model = data['face_model']
      self.gpu_id = data['gpu_id']
      self.conf_threshold = data['conf_threshold']
      self.cache_string = data['cache_string']
    self.confidenceSlider.setValue(self.conf_threshold * 100)
    self.confidenceChanged()


  def loadModel(self, fileName):
    """ Load a 3D model """
    self.ModelPath = fileName
    self.modelInfo.setText("Model: " + self.ModelPath)
    
    
  def loadVideo(self, fileName):
    """ Load a video """
    self.videoPath = fileName
    self.videoInfo.setText("Video: " + self.videoPath)
    self.video = cv2.VideoCapture(self.videoPath)
    if(not self.video.isOpened()):
      print("ERROR : Can't load", self.videoPath)
      return 
    self.videoFPS = int(self.video.get(cv2.CAP_PROP_FPS))
    self.frameCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # we set the slider
    self.videoSlider.setMaximum(self.frameCount - 1) 
    
    # we load the cache
    self.cachePath = self.videoPath + self.cache_string + ".json"
    self.isVideoLoaded = True
    self.loadCacheFile() #we load or create the cache file
    
    # and we load the process table view
    self.initProcessTable()
    
    # we draw the first frame
    self.setFrame(0) 


  def initProcessTable(self):
    """ Create the process table according to the video (the colored table that shows the loaded frames below the video) """
    self.videoProcessTable.setColumnCount(self.frameCount)
    self.videoProcessTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    for i in range(self.frameCount):
      item = QtWidgets.QTableWidgetItem()
      self.videoProcessTable.setItem(0, i, item)
      self.updateProcessTable(i)
        
    
  def loadCacheFile(self):
    """ Load the cache file of the video or create it if it does not exist """
    if not os.path.exists(self.cachePath):
      self.initCacheFile()
    else:
      with open(self.cachePath) as json_cacheFile:
        self.cacheData = json.load(json_cacheFile)
    
    
  def initCacheFile(self):
    """ Create the cache file of the video """
    self.cacheData = {"data": []}
    for i in range(int(self.frameCount)):
      self.cacheData["data"].append({"isLoaded": False,
                                    "faces": []})
    self.saveCacheFile()
      
  def loadData(self):
    """ Load deep learning data (DLIB and Hopenet) """
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

  
  
  #################################
  ### ====     DRAWING     ==== ###
  #################################

  def drawFrame(self):
    """ Draw the current frame on the screens (the current frame must be initialized before) """
    if(self.isVideoLoaded == False):
      return
    
    frame = self.frame.copy()

    # add axis
    if(self.ifDrawAxis):
      frame = self.drawAxis(frame)
      
    # add square
    if(self.ifDrawSquare):
      frame = self.drawSquare(frame)
    
    # convert cv2 video to QPixmap
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


  def drawNextFrame(self):
    """ Draw the next frame """
    if(self.isVideoLoaded == False):
      return
    self.ret, frame = self.video.read()
    if (self.ret == True):
      
      # Draw frame
      self.frame = frame
      self.currentFrame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
      self.drawFrame()
      
      # Update slider
      self.videoSlider.setValue(self.currentFrame-1)
      
      # Update info
      self.updateInfo()

    else: # If the video is end
      self.isPlaying = False


  def drawAxis(self, frame):
    """ Draw the current faces orientation on the frame 'frame'. Does not overwrite the data of the current frame """
    for face in self.getCurrentFaces():
      face.drawAxis(frame)
    return frame
  
  
  def drawSquare(self, frame):
    """ Draw the current faces square on the frame 'frame'. Does not overwrite the data of the current frame """
    for face in self.getCurrentFaces():
      face.drawSquare(frame)
    return frame

  
  
  #################################
  ### ====     UPDATE      ==== ###
  #################################


  def updateInfo(self):
    """ Update the frame informations displayed on the screen """
    frameInfo = "Frame: " + str(int(self.currentFrame)) + "/" + str(int(self.frameCount))
    self.frameInfo.setText(frameInfo)
    if(self.getCurrentCacheData()['isLoaded']):
      faceNumber = self.getCurrentCacheData()['faces'].__len__()
      if(faceNumber > 0):
        facesInfo = str(faceNumber) + " face(s) detected"
        self.facesInfo.setStyleSheet("color: rgb(0, 153, 0);")
      else:
        facesInfo = "No faces were detected"
        self.facesInfo.setStyleSheet("color: rgb(255, 102, 0);")
    else:
      facesInfo = "Frame not process"
      self.facesInfo.setStyleSheet("color: rgb(255, 0, 0);")
    self.facesInfo.setText(facesInfo)
   

  def updateProcessTable(self, index):
    """ Update the frame informations displayed on the screen """
    bg = QtGui.QColor(153, 153, 153) #default background (gray)
    # if a frame is loas
    if(self.getCacheData(index)['isLoaded']):
      # Frame with faces
      if(self.getCacheData(index)['faces'].__len__() > 0):
        bg = QtGui.QColor(0, 153, 0)
      # Frame without faces
      else:
        bg = QtGui.QColor(255, 102, 0)
    self.videoProcessTable.item(0, index).setBackground(bg)
    

  def update(self):
    """ Update function to call in loop """
    # PLAY THE VIDEO
    if(self.isPlaying):
      self.drawNextFrame()
      
      # FPS CONTROLLER
      elapsedTime = timeit.default_timer() - self.lastUpdate
      if (elapsedTime < 1. / self.videoFPS): 
        time.sleep(1. / self.videoFPS - elapsedTime)

    self.lastUpdate = timeit.default_timer()

  


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