from PIL import Image
import cv2
import datasets
import dlib
import dlib.cuda as cuda
from face import Face
from hopenet import *
import timeit
import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

"""
FaceDetector.py
Faces detection management class
""" 

class FaceDetector():
  def __init__(self, faceModelPath, snapshotPath, gpuID=None):
    self.gpuId = gpuID
    self.faceModelPath = faceModelPath
    self.snapshotPath = snapshotPath
    self.isLoadedData = False
    self.hopenetModel = None
    self.cnnFaceDetector = None
    self.transformation = None
    self.ifStop = False


  def load(self, callback=None):
    """ Load deep learning data (DLIB and Hopenet). Call the callback(float, string) function with percentage and progress message at each state change """
    if callback != None:
      callback(0.0, "Get GPU/CPU config...")
    
    #Get GPU/CPU config ==> TO DO
    cudnn.enabled = True
    print(torch.cuda.get_device_name(self.gpuId))
    dlib.DLIB_USE_CUDA = 1
    
    if callback != None:
      callback(0.22, "Set Hopenet and Dlib Model...")
      
    # Set hopenet model with ResNet50 structure
    self.hopenetModel = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # Dlib face detection model
    self.cnnFaceDetector = dlib.cnn_face_detection_model_v1(self.faceModelPath)
    
    if callback != None:
      callback(0.33, "Loading snapshot (Hopenet pkl model)...")  

    # Load snapshot
    saved_state_dict = torch.load(self.snapshotPath)
    self.hopenetModel.load_state_dict(saved_state_dict)
    
    if callback != None:
      callback(0.97, "Loading data...") 

    self.transformations = transforms.Compose([transforms.Scale(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
    self.hopenetModel.cuda(self.gpuId)
  
    if callback != None:
      callback(1.0, "Ready to test network...") 
      
    self.isLoadedData = True

    # Test the Model
    self.hopenetModel.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    #torch.no_grad()


  def stop(self):
    """ Stop the frame calculation """
    self.ifStop = True


  def isLoaded(self):
    """ Return true if deep learning data is loaded """
    return self.isLoadedData
  
  def getFrameFaces(self, frame, callback=None):
    """ Get faces of the frame. Call the callback(float, string) function with percentage and progress message at each state change. Raise exception if stopProcessing is called  """
    res = []; # Result data
    self.ifStop = False

    # We read the frame
    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if self.ifStop : 
      raise
    if callback != None:
      callback(0.01, "Dlib face detection (this may be long)") 

    # Dlib face detection
    dets = self.cnnFaceDetector(cv2_frame, self.gpuId)

    if self.ifStop : 
      raise
    if callback != None:
      callback(0.7, "Calculate tensor") 

    # we calculate tensor
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpuId)
    
    facesNumber = len(list(enumerate(dets)))
    faceCount = 0

    # For each detected faces
    for idx, det in enumerate(dets):
      faceCount += 1
      
      if self.ifStop : 
        raise
      if callback != None:
        callback(0.75 + 0.25 * faceCount/facesNumber, "Calculate face orientation (%d/%d)" % (faceCount, facesNumber)) 
      
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
      img = Variable(img).cuda(self.gpuId)

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

    if callback != None:
      callback(1.0, "End of the faces calculation") 

    return res
  
  
  
# All frame functions

# Get the actual frame data
#   frame = the cv2 frame
#   model = hopenet load model
#   cnn_face_detector = Dlib face detection model
#   gpu = the gpu id
def getFrameFaces(frame, model, cnn_face_detector, transformations, gpu):
  # Result data
  res = [];

  # We read the frame
  cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Dlib face detection
  startDLIB = timeit.default_timer()
  print("-- Dlib Face detector")
  dets = cnn_face_detector(cv2_frame, gpu)
  print("   Time : " + str(timeit.default_timer() - startDLIB))
  
  # we calculate tensor
  idx_tensor = [idx for idx in range(66)]
  idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

  # For each detected faces
  for idx, det in enumerate(dets):
    # Get x_min, y_min, x_max, y_max, conf
    x_min = det.rect.left()
    y_min = det.rect.top()
    x_max = det.rect.right()
    y_max = det.rect.bottom()
    conf = det.confidence

    print("Possible face (", x_min, ",", y_min, ",", x_max, ",", y_max, ") with", conf, "confidence ")

    print("-- Face detected")
    print("-- Calculation of orientation")
    startFace = timeit.default_timer()

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
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(gpu)

    # get 3d orientation of the face
    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)

    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    # save data
    print("-- Result : ", yaw_predicted, pitch_predicted, roll_predicted)
    res.append(Face(conf, x_min, x_max, y_min, y_max, yaw_predicted, pitch_predicted, roll_predicted))

    # end of loop
    print("   Time : " + str(timeit.default_timer() - startFace))

  return res
      
