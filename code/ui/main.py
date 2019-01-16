# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
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
        self.layoutWidget = QtWidgets.QWidget(self.MainWindows)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 370, 541, 61))
        self.layoutWidget.setObjectName("layoutWidget")
        self.menu = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.menu.setContentsMargins(0, 0, 0, 0)
        self.menu.setObjectName("menu")
        self.slider = QtWidgets.QSlider(self.layoutWidget)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.menu.addWidget(self.slider)
        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setObjectName("buttons")
        self.lastFrame_PB = QtWidgets.QPushButton(self.layoutWidget)
        self.lastFrame_PB.setObjectName("lastFrame_PB")
        self.buttons.addWidget(self.lastFrame_PB)
        self.read_PB = QtWidgets.QPushButton(self.layoutWidget)
        self.read_PB.setObjectName("read_PB")
        self.buttons.addWidget(self.read_PB)
        self.headPosition_PB = QtWidgets.QPushButton(self.layoutWidget)
        self.headPosition_PB.setObjectName("headPosition_PB")
        self.buttons.addWidget(self.headPosition_PB)
        self.pause_PB = QtWidgets.QPushButton(self.layoutWidget)
        self.pause_PB.setObjectName("pause_PB")
        self.buttons.addWidget(self.pause_PB)
        self.nextFrame_PB = QtWidgets.QPushButton(self.layoutWidget)
        self.nextFrame_PB.setObjectName("nextFrame_PB")
        self.buttons.addWidget(self.nextFrame_PB)
        self.menu.addLayout(self.buttons)
        self.layoutWidget.raise_()
        self.frame.raise_()
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
        self.lastFrame_PB.setText(_translate("MainWindow", "Last Frame"))
        self.read_PB.setText(_translate("MainWindow", "Lecture"))
        self.headPosition_PB.setText(_translate("MainWindow", "Get head position"))
        self.pause_PB.setText(_translate("MainWindow", "Pause"))
        self.nextFrame_PB.setText(_translate("MainWindow", "Next Frame"))

