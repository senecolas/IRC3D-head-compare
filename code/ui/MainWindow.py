# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(662, 472)
        self.MainWindows = QtWidgets.QWidget(MainWindow)
        self.MainWindows.setMinimumSize(QtCore.QSize(662, 431))
        self.MainWindows.setMaximumSize(QtCore.QSize(662, 16777215))
        self.MainWindows.setObjectName("MainWindows")
        self.frame = QtWidgets.QFrame(self.MainWindows)
        self.frame.setGeometry(QtCore.QRect(0, 0, 661, 351))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.widget = QtWidgets.QWidget(self.MainWindows)
        self.widget.setGeometry(QtCore.QRect(60, 370, 541, 61))
        self.widget.setObjectName("widget")
        self.menu = QtWidgets.QVBoxLayout(self.widget)
        self.menu.setContentsMargins(0, 0, 0, 0)
        self.menu.setObjectName("menu")
        self.slider = QtWidgets.QSlider(self.widget)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.menu.addWidget(self.slider)
        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setObjectName("buttons")
        self.lastFrame = QtWidgets.QPushButton(self.widget)
        self.lastFrame.setObjectName("lastFrame")
        self.buttons.addWidget(self.lastFrame)
        self.read = QtWidgets.QPushButton(self.widget)
        self.read.setObjectName("read")
        self.buttons.addWidget(self.read)
        self.headPosition = QtWidgets.QPushButton(self.widget)
        self.headPosition.setObjectName("headPosition")
        self.buttons.addWidget(self.headPosition)
        self.pause = QtWidgets.QPushButton(self.widget)
        self.pause.setObjectName("pause")
        self.buttons.addWidget(self.pause)
        self.nextFrame = QtWidgets.QPushButton(self.widget)
        self.nextFrame.setObjectName("nextFrame")
        self.buttons.addWidget(self.nextFrame)
        self.menu.addLayout(self.buttons)
        self.read.raise_()
        self.nextFrame.raise_()
        self.headPosition.raise_()
        self.lastFrame.raise_()
        self.pause.raise_()
        self.frame.raise_()
        self.slider.raise_()
        MainWindow.setCentralWidget(self.MainWindows)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 662, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lastFrame.setText(_translate("MainWindow", "Last Frame"))
        self.read.setText(_translate("MainWindow", "Lecture"))
        self.headPosition.setText(_translate("MainWindow", "Get head position"))
        self.pause.setText(_translate("MainWindow", "Pause"))
        self.nextFrame.setText(_translate("MainWindow", "Next Frame"))

