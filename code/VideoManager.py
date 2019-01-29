"""
VideoManager.py
Video manager class with cache management
"""

from Face import Face
from FaceDetector import FaceDetector
from PyQt5 import QtGui
import cv2
import json
import os

class VideoManager():
  def __init__(self, videoPath="", cacheString="-cache", faceDetector=None):
    """
    Constructor of the VideoManager
    """
    self.videoPath = videoPath
    self.cacheString = cacheString
    self.video = None 
    self.currentFrame = []
    self.currentFrameNumber = 0
    self.frameCount = 0 #total frame number
    self.fps = 25
    self.ret = True #if the video is over
    self.isPlaying = False
    self.ifDrawAxis = False
    self.ifDrawSquare = False
    self.cacheData = {}
    
    if videoPath != "":
      self.load()
      
      
      
  #################################
  ### ====     LOADING     ==== ###
  #################################
    
  def load(self, fileName=None, cacheString=None):
    """ Load the video """
    if fileName != None:
      self.videoPath = fileName
    if cacheString != None:
      self.cacheString = cacheString
    
    if not os.path.exists(self.videoPath):
      raise ValueError("ERROR : %s don't exist" % (self.videoPath))
    
    self.video = cv2.VideoCapture(self.videoPath)
    
    if(not self.video.isOpened()):
      self.video = None
      raise ValueError("ERROR : Can't load %s" % (self.videoPath))
 
    self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
    self.frameCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # we load the cache
    self.cachePath = self.videoPath + self.cacheString + ".json"
    self.loadCacheFile() #we load or create the cache file
    
    # we draw the first frame
    self.setFrame(0)
  
  
  
  #################################
  ### ====      TESTS      ==== ###
  #################################  

  def isLoaded(self):
    """ Returns true if a video is loaded """
    if self.video == None:
      return False
    return True
  
  
  def hasCurrentFrame(self):
    """ Returns true if the current frame is set """
    if self.currentFrame == []:
      return False
    return True
  
  
  def isFacesLoaded(self):
    """ Returns true if the face detector has already calculated this frame """
    return self.getCurrentCacheData()['isLoaded']
  
  
  
  #################################
  ### ====  FACE DETECTOR  ==== ###
  #################################
  
  def getHeadPosition(self, faceDetectorCallback=None):
    """ 
    Launch the calculation to get the faces of the current frame and saves it in the cache file. 
    Call the faceDetectorCallback(float, string) function with percentage and progress message at each state change
    """
    
    if(self.isLoaded() == False):
      raise ValueError("ERROR : no video was loaded")
    if(self.hasCurrentFrame() == False):
      raise ValueError("ERROR : no frame has been defined")
    if(self.faceDetector == None or self.faceDetector.isLoaded() == False):
      raise ValueError("ERROR : FaceDetector was not loaded")
    if(self.isFacesLoaded()):
      return self.getCurrentCacheData()
    
    # Get faces
    try:
      faces = self.faceDetector.getFrameFaces(self.currentFrame, faceDetectorCallback)
      for vis in faces:
        self.getCurrentCacheData()['faces'].append(vis.getJSONData())

      # Updated cache
      self.getCurrentCacheData()['isLoaded'] = True
      self.saveCacheFile()
    except:
      print("stopped getHeadPosition")

    # return 
    return self.getCurrentCacheData()


  def getAllHeadPosition(self, faceDetectorCallback=None, callback=None):
    """ 
    Launch the getHeadPosition function for all frame from the current frame
    Call the faceDetectorCallback(float, string) function with percentage and progress message at each state change
    Call the callback() function after each frame processing 
    """
    for i in range(self.currentFrameNumber, self.frameCount + 1):
      if self.isFacesLoaded() == False:
        self.getHeadPosition(faceDetectorCallback)
        if self.faceDetector.isStopped():
          return
      if callback != None:
        callback()
      self.read()
        
    
  #################################
  ### ====     ACTIONS     ==== ###
  #################################
      
  def play(self):
    """ Start the video if it's loaded. If the video is end, restart the video """  
    if(self.isLoaded() == False):
      return
    self.isPlaying = True
    #if the video is end, we restart
    if(self.ret == False):
      self.video.set(1, 0)
      
      
  def pause(self):
    """ Pause the video """  
    self.isPlaying = False
    
      
  def moveFrame(self, num, callback=None):
    """ Moves the current frame by 'num' frames. Call the callback function at the end if defined (e.g. to update an interface) """  
    if(self.isLoaded() == False):
      return
    self.isPlaying = False
    self.currentFrameNumber = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.setFrame(self.currentFrameNumber + num - 1, callback)
    
    
  def setFrame(self, frameNum, callback=None):
    """ Set the current frame on 'frameNum'. Call the callback function at the end if defined (e.g. to update an interface)  """ 
    if(self.isLoaded() == False):
      return
    self.video.set(1, frameNum)
    self.currentFrameNumber = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.read()
    if callback != None:
      callback()
    
  
  #################################
  ### === CACHE  MANAGEMENT === ###
  #################################
    
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
    
  
  def saveCacheFile(self):
    """ Update the cache file """
    with open(self.cachePath, 'w', encoding='utf-8') as outfile:
      json.dump(self.cacheData, outfile)


  def getCurrentCacheData(self):
    """ Return the data of the current frame """
    return self.getCacheData(int(self.currentFrameNumber - 1))
  
  
  def getCacheData(self, index):
    """ Return the data of the frame corresponding to the 'index' (be careful: the index starts at 0 and the frame at 1) """
    return self.cacheData['data'][index]
  
  
  
  #################################
  ### ===      GETTERS      === ###
  #################################
    
  def faces(self):
    """ Return an array of current faces on the frame taking into account the 'confThreshold' (empty if not process or if no face has been detected with this conf_threshold) """
    res = []
    if(self.isFacesLoaded() == False):
      return res
    for face in self.getCurrentCacheData()['faces']:
      face = Face().setJSONData(face)
      if(face.confidence > self.faceDetector.confThreshold):
        res.append(face)
    return res
  


  #################################
  ### ====     DRAWING     ==== ###
  #################################

  def frame(self, ifDrawAxis=None, ifDrawSquare=None):
    """ 
    Get the current frame in QPixmap format for drawing on the screen (the current frame must be initialized before)
    If ifDrawAxis and ifDrawSquare are not defined, they will be replaced by the values of the class
    Return the current frame in QPixmap format and it's number
    """
    if self.isLoaded() == False:
      raise ValueError("ERROR : no video was loaded")
    
    if self.hasCurrentFrame() == False:
      raise ValueError("ERROR : no frame has been defined")
    
    frame = self.currentFrame.copy()

    # add axis
    if ifDrawAxis or (self.ifDrawAxis and ifDrawAxis == None):
      frame = self.drawAxis(frame)
      
    # add square
    if ifDrawSquare or (self.ifDrawSquare and ifDrawSquare == None):
      frame = self.drawSquare(frame)
    
    # convert cv2 video to QPixmap
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QtGui.QImage.Format_RGB888)
    convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
    pixmap = QtGui.QPixmap(convertToQtFormat)
    
    return pixmap, self.currentFrameNumber
  
  
  def read(self, ifDrawAxis=None, ifDrawSquare=None):
    """ 
    Get the next frame in the QPixmap format for drawing on the screen
    If ifDrawAxis and ifDrawSquare are not defined, they will be replaced by the values of the class 
    """
    if(self.isLoaded() == False):
      return
    self.ret, frame = self.video.read()
    if (self.ret == True):
      
      # Draw frame
      self.currentFrame = frame
      self.currentFrameNumber = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
  
    else: # If the video is end
      self.isPlaying = False
      
    return self.frame(ifDrawAxis, ifDrawSquare)
  
  
  def drawAxis(self, frame):
    """ Draw the current faces orientation on the frame 'frame'. Does not overwrite the data of the current frame """
    for face in self.faces():
      face.drawAxis(frame)
    return frame
  
  
  def drawSquare(self, frame):
    """ Draw the current faces square on the frame 'frame'. Does not overwrite the data of the current frame """
    for face in self.faces():
      face.drawSquare(frame)
    return frame