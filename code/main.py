from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets
import argparse
import cv2
import datasets
import dlib
from frame import *
from hopenet import *
import os
import sys
import timeit
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from ui import main
import utils

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                      default=0, type=int)
  parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                      default='', type=str)
  parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                      default='', type=str)
  parser.add_argument('--video', dest='video_path', help='Path of video')
  parser.add_argument('--output_string', dest='output_string', help='String appended to output file', default="")
  parser.add_argument('--frame', dest='frame', help='The frame to calculate', type=int, default=1)
  parser.add_argument('--conf_threshold', dest='conf_threshold', help='The face detection threshold', type=float, default=0.75)
  args = parser.parse_args()
  return args


class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent=parent)
    self.setupUi(self)
    self.play = False
  
  # Draw the actual frame on the screen
  def drawFrame(self, frame):
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QtGui.QImage.Format_RGB888)
    convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
    pixmap = QPixmap(convertToQtFormat)
    resizeImage = pixmap.scaled(661, 351, QtCore.Qt.KeepAspectRatio)
    #QApplication.processEvents()
    self.VideoWidget.setPixmap(resizeImage)



if __name__ == '__main__':
  args = parse_args()
  
  app = QtWidgets.QApplication(sys.argv)
  window = MainWindow()
  window.show()
  
  window.video = cv2.VideoCapture(args.video_path);
 
  
  cap = cv2.VideoCapture(args.video_path)
 
  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      # Display the resulting frame
      window.drawFrame(frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else: 
      break
      
  sys.exit(app.exec_())
  
  comment = """

  startTime = timeit.default_timer()
  
  print(torch.cuda.get_device_name(args.gpu_id))

  cudnn.enabled = True

  batch_size = 1
  gpu = args.gpu_id
  snapshot_path = args.snapshot
  out_dir = '../output'
  video_path = args.video_path

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  if not os.path.exists(args.video_path):
    sys.exit('Video does not exist')

  # ResNet50 structure
  model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

  # Dlib face detection model
  cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

  print ('Loading snapshot.')
  # Load snapshot
  saved_state_dict = torch.load(snapshot_path)
  model.load_state_dict(saved_state_dict)

  print ('Loading data.')

  transformations = transforms.Compose([transforms.Scale(224),
                                       transforms.CenterCrop(224), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  model.cuda(gpu)

  print ('Ready to test network.')

  # Test the Model
  model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

  video = cv2.VideoCapture(video_path)


  print ("== Frame " + str(args.frame) + " == ")
  video.set(1, args.frame - 1) #we go to the determined image on the video
  
  # read frame
  visages = getFrameVisages(video, model, cnn_face_detector, transformations, args.conf_threshold, gpu)
  for vis in visages:
    print("VISAGE : ", float(vis.yaw), float(vis.pitch), float(vis.roll)) 
  
  video.release()
  time = timeit.default_timer() - startTime
  print("== THE END ==")
  print('Time : ', time) """