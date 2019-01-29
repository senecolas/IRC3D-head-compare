"""
FaceDetector.py
Faces detection management class
""" 

from Face import Face
from PIL import Image
import cv2
import datasets
import dlib
import dlib.cuda as cuda
import gc
from hopenet import *
import time
import timeit
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms


class FaceDetector():
  def __init__(self, faceModelPath, snapshotPath, gpuID=None):
    """ 
    Constructor of the FaceDetector
      - 'faceModelPath' is path of the dlib face detections model (.dat)
      - 'snapshotPath' is the path of the Hopenet trained model to use to calculate the orientation of the face (.pkl)
      - 'gpuID' is the identifier of the GPU to use by default (if no gpu is detected, the cpu will be used)
    """
    self.gpuId = gpuID
    self.deviceType = 'cpu'
    self.faceModelPath = faceModelPath
    self.snapshotPath = snapshotPath
    self.confThreshold = 0.75
    self.isLoadedData = False
    self.hopenetModel = None
    self.cnnFaceDetector = None
    self.transformation = None
    self.device = None
    self.isStop = False


  def setTorchConfig(self):
    """ Set the CPU or GPU configuration """
    
    if torch.cuda.is_available() and self.gpuId >= 0: # CUDA is available 
      self.deviceType = 'cuda'
      if self.gpuId >= torch.cuda.device_count(): # invalid GPU ID, we take the current GPU
        self.gpuId = torch.cuda.current_device()
      self.device = torch.device('cuda:%d' % (self.gpuId))
      cudnn.enabled = True

      # CUDA INFO
      print("-- FaceDetector use the", torch.cuda.get_device_name(self.gpuId))
      if dlib.DLIB_USE_CUDA == False:
        print("Your installation of dlib does not use cuda, the face detection will use the CPU. To reinstall dlib with cuda, refer to the installation instructions.")
      
    else: # CUDA is not available or gpuId = -1 (force CPU), use CPU
      self.deviceType = 'cpu'
      self.device = torch.device('cpu') 
      print("-- FaceDetector use the CPU")


  def load(self, callback=None):
    """ Load deep learning data (DLIB and Hopenet). Call the callback(float, string) function with percentage and progress message at each state change """
    if callback != None:
      callback(0.0, "Get GPU/CPU config...")
    
    #Get GPU/CPU config
    self.setTorchConfig()
    
    if callback != None:
      callback(0.22, "Set Hopenet and Dlib Model...")
      
    # Set hopenet model with ResNet50 structure
    self.hopenetModel = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # Dlib face detection model
    self.cnnFaceDetector = dlib.cnn_face_detection_model_v1(self.faceModelPath)

    if callback != None:
      callback(0.33, "Loading snapshot (Hopenet pkl model)...")  

    # Load snapshot
    if self.deviceType == 'cuda':
      saved_state_dict = torch.load(self.snapshotPath)
    else:
      saved_state_dict = torch.load(self.snapshotPath, map_location=lambda storage, loc: storage)

    self.hopenetModel.load_state_dict(saved_state_dict)
    
    if callback != None:
      callback(0.97, "Loading data...") 

    self.transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
    
    if self.deviceType == 'cuda':
      self.hopenetModel.cuda(self.device)
    else:
      self.hopenetModel.cpu()
  
    if callback != None:
      callback(1.0, "Ready to test network...") 
      
    self.isLoadedData = True

    # Test the Model
    self.hopenetModel.eval()  # Change model to 'eval' mode (BN uses moving mean/var).


  def stop(self):
    """ Stop the frame calculation """
    self.isStop = True


  def isStopped(self):
    return self.isStop


  def isLoaded(self):
    """ Return true if deep learning data is loaded """
    return self.isLoadedData
  
  
  def getFrameFaces(self, frame, callback=None):
    """ Get faces of the frame. Call the callback(float, string) function with percentage and progress message at each state change. Raise exception if stopProcessing is called  """
    res = []; # Result data
    self.isStop = False
    startTime = timeit.default_timer()

    # We read the frame
    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if self.isStop: 
      raise
    if callback != None:
      callback(0.01, "Dlib face detection (this may be long)") 

    # Dlib face detection
    try:
      dets = self.cnnFaceDetector(cv2_frame, 1) # 1 is the number of times it should upsample the image (helps to detect smaller faces)
    except:
      dets = self.cnnFaceDetector(cv2_frame, 0) # when upsample does not work
      
    if self.isStop: 
      raise
    if callback != None:
      callback(0.7, "Calculate tensor") 

    # we calculate tensor
    idx_tensor = [idx for idx in range(66)]

    if self.deviceType == 'cuda':
      idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.device)
    else:
      idx_tensor = torch.FloatTensor(idx_tensor).cpu()
    
    facesNumber = len(list(enumerate(dets)))
    faceCount = 0

    # For each detected faces
    for idx, det in enumerate(dets):
      faceCount += 1
      
      if self.isStop: 
        raise
      if callback != None:
        callback(0.75 + 0.25 * faceCount / facesNumber, "Calculate face orientation (%d/%d)" % (faceCount, facesNumber)) 
      
      # Get x_min, y_min, x_max, y_max, conf
      x_min = det.rect.left()
      y_min = det.rect.top()
      x_max = det.rect.right()
      y_max = det.rect.bottom()
      conf = det.confidence

      # coordinate of the face
      bbox_width = abs(x_max - x_min)
      bbox_height = abs(y_max - y_min)
      x_min -= 2 * bbox_width / 4
      x_max += 2 * bbox_width / 4
      y_min -= 3 * bbox_height / 4
      y_max += bbox_height / 4
      x_min = max(x_min, 0); y_min = max(y_min, 0)
      x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)

      # Crop the face image
      img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
      img = Image.fromarray(img)

      # Transform the face image
      img = self.transformations(img)
      img_shape = img.size()
      img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
      
      if self.deviceType == 'cuda':
        img = Variable(img).cuda(self.device)
      else:
        img = Variable(img).cpu()

      # get 3d orientation of the face
      yaw, pitch, roll = self.hopenetModel(img)

      yaw_predicted = F.softmax(yaw, dim=1)
      pitch_predicted = F.softmax(pitch, dim=1)
      roll_predicted = F.softmax(roll, dim=1)

      # Get continuous predictions in degrees.
      yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
      pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
      roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

      # save data
      res.append(Face(conf, x_min, x_max, y_min, y_max, yaw_predicted, pitch_predicted, roll_predicted))

    print("Face Detector Time:", timeit.default_timer() - startTime)
    return res