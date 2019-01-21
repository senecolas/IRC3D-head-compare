from PIL import Image
from visage import Visage
import cv2
import datasets
import dlib
from hopenet import *
import timeit
import torch
import timeit
from torchvision import transforms

# frame.py
# All frame functions

# Get the actual frame data
#   frame = the cv2 frame
#   model = hopenet load model
#   cnn_face_detector = Dlib face detection model
#   gpu = the gpu id
def getFrameVisages(frame, model, cnn_face_detector, transformations, gpu):
  # Result data
  res = [];

  # We read the frame
  cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Dlib visage detection
  startDLIB = timeit.default_timer()
  print("-- Dlib Face detector")
  dets = cnn_face_detector(cv2_frame, gpu)
  print("   Time : " + str(timeit.default_timer() - startDLIB))
  
  # we calculate tensor
  idx_tensor = [idx for idx in range(66)]
  idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

  # For each detected visages
  for idx, det in enumerate(dets):
    # Get x_min, y_min, x_max, y_max, conf
    x_min = det.rect.left()
    y_min = det.rect.top()
    x_max = det.rect.right()
    y_max = det.rect.bottom()
    conf = det.confidence

    print("Possible face (", x_min, ",", y_min, ",", x_max, ",", y_max, ") with", conf, "confidence ")

    print("-- Visage detected")
    print("-- Calculation of orientation")
    startVisage = timeit.default_timer()

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
    res.append(Visage(conf, x_min, x_max, y_min, y_max, yaw_predicted, pitch_predicted, roll_predicted))

    # end of loop
    print("   Time : " + str(timeit.default_timer() - startVisage))

  return res
      
