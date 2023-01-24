import sys
from PyQt5.QtWidgets import QWidget, QLabel, QFormLayout, QPushButton, QMainWindow, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QAction, QMessageBox, QApplication, QFileDialog, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt  # ?
import cv2
import numpy as np
import math


class VideoCapture(QLabel):
    def __init__(self, filename, parent):
        super(QLabel, self).__init__()
        self.cap = cv2.VideoCapture(str(filename[0]))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) * 10
        # self.codec = self.cap.get(cv2.CAP_PROP_FOURCC)
        # self.video_label = QLabel(self)
        # print('cv2 video width:',int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print('cv2 video height:',int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print('selfwidth',self.width())
        # print('selfheight', self.height())
        # print('labelwidth',self.video_label.width())
        # print('labelheight', self.video_label.height())
        # parent.layout.addWidget(self.video_label)
        # set your label as the window's centralWidget.
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.framesToDenoise = []
        lightSpeed = 299792458  # m/s
        distance = 20000000 * 2  # 20 000 km # multiply by 2 ?
        travelTime = distance / lightSpeed
        objSpeedInPixels = 99
        correction = objSpeedInPixels * travelTime
        print('traveltime:', travelTime, 'correction:', correction)
        # self.nextFrameSlot()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w  # 24bits per pixel
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def nextFrameSlot(self):
        ret, frame = self.cap.read(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        background = frame
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if pos == self.length:  # Loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # frame = self.denoiseFilter(frame)
        frame = cv2.medianBlur(frame, 21)
        # frame = frame[y1:y2, x1:
        h, w = frame.shape
        roiBorderX = 300
        roiBorderY = 100
        frame = frame[roiBorderY:(h - roiBorderY), roiBorderX:(w - roiBorderX)]
        # TODO: put some filters after ROI before Border
        # TODO: mirror the top|bottom (also, beam is not in center)?
        frame = cv2.equalizeHist(frame)
        # print('focus',self.getFocus(frame))
        # image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
        frame = cv2.copyMakeBorder(frame, roiBorderY, roiBorderY, roiBorderX, roiBorderX, cv2.BORDER_CONSTANT)
        frame = self.thresholdFilter(frame)
        # remove holes
        mkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, mkernel)

        pts = self.enclosingTriangle(frame)
        pts = pts.reshape((-1, 3, 2))
        # cv2.polylines(frame,pts,True,(123),1,cv2.LINE_AA)#
        cv2.drawMarker(background, (pts[0, 2, 0], pts[0, 2, 1]), (255), cv2.MARKER_CROSS, 10, 1)
        # TODO: get only coords here
        # kernel = np.ones((3, 3), 'uint8')
        # frame = cv2.erode(frame, kernel, cv2.BORDER_REFLECT, iterations=1)
        # frame = cv2.dilate(frame, kernel, cv2.BORDER_REFLECT, iterations=1)
        # background = self.addHue(background)
        # threshFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # added_image = cv2.addWeighted(threshFrame, 0.6, background, 0.4, 0)
        pix = self.convert_cv_qt(background)
        self.setPixmap(pix)

    def getFocus(self, frame):
        # cv::Laplacian(src_gray, dst, CV_64F);
        #
        # cv::Scalar mu, sigma;
        # cv::meanStdDev(dst, mu, sigma);
        dst = cv2.Laplacian(frame, cv2.CV_64F)
        mu = np.array
        sigma = np.array
        mu = np.zeros(frame.shape[:2], dtype=np.uint8)
        sigma = np.zeros(frame.shape[:2], dtype=np.uint8)
        # Expected Ptr<cv::UMat> for argument 'mean'
        cv2.meanStdDev(dst, mu, sigma)
        return sigma[0] * sigma[0]

    def enclosingTriangle(self, frame):
        # Find only external contours in grayscale image
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find triangle vertices of minimum area enclosing contour
        _, triangle = cv2.minEnclosingTriangle(contours[0])  # FIXME: index out of range
        pts = np.int32(np.squeeze(np.round(triangle)))
        # print('area:', self.triangleArea(pts))
        # TODO: add plot for area

        # cv::drawMarker(myimage, cv::Point(x, y),  cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
        # convert(flatten) to 1d array, that contains 3 arrays, with 2d elements
        # pts = pts.reshape((-1, 1, 2))
        # convert(flatten) to 1d array, that contains 1 array, with 2d elements
        # was 3d array with 2d elements 3,1,2

        return pts

    def distanceCartesian(self, pointOne, pointTwo):
        return math.sqrt(((pointOne[0] - pointTwo[0]) ** 2) + ((pointOne[1] - pointTwo[1]) ** 2))

    def triangleArea(self, pts):
        a = self.distanceCartesian(pts[0], pts[1])
        b = self.distanceCartesian(pts[1], pts[2])
        c = self.distanceCartesian(pts[2], pts[0])
        # calculate the semi-perimeter
        s = (a + b + c) / 2
        # calculate the area
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        print('The area of the triangle is %0.2f' % area)
        return area

    # TODO: get focus value by triangle size?
    # TODO: draw the fucking triangle
    # TODO: handle brightness variations
    # TODO: convert more videos

    def denoiseFilter(self, frame):
        frame = cv2.fastNlMeansDenoising(frame)
        return frame

    def addHue(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w, ch = frame.shape
        red = (0, 0, 255)
        color = tuple(reversed(red))
        redImage = np.zeros((h, w, ch), dtype="uint8")
        redImage[:] = color
        # old api
        # mask = cv2.CreateMat(frame.rows, frame.cols, cv2.CV_8UC3, color)
        frame = cv2.addWeighted(frame, 0.5, redImage, 0.5, cv2.CV_8UC3)
        return frame

    def adaptiveThresholdFilter(self, frame):
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        return frame

    def thresholdFilter(self, frame):
        cv2.threshold(frame, 220, 255, cv2.THRESH_TOZERO, frame)
        return frame

    def start(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000.0 / self.frame_rate))

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        self.framesToDenoise.clear()
        super(QLabel, self).deleteLater()


class VideoControlsWidget(QWidget):
    def __init__(self, parent):
        super(VideoControlsWidget, self).__init__(parent)
        self.layout = QHBoxLayout(self)
        self.startButton = QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QPushButton('Pause', parent)
        self.pauseButton.setFixedWidth(50)
        self.layout.addWidget(self.startButton)
        self.layout.addWidget(self.pauseButton)
        self.setLayout(self.layout)


class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.setGeometry(0, 0, 1024, 868)
        self.setWindowTitle("CV demo 0")

        self.capture = None

        self.isVideoFileLoaded = False

        self.quitAction = QAction("&Exit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.triggered.connect(self.closeApplication)

        self.openVideoFile = QAction("&Open Video File", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)

        self.videoControlsWidget = VideoControlsWidget(self)
        self.videoControlsWidget.sizePolicy().setVerticalPolicy(QSizePolicy().Minimum)

        self.mainWidget = QWidget(self)
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.mainLayout.addWidget(self.videoControlsWidget)

        self.setCentralWidget(self.mainWidget)

    def startCapture(self):
        # FIXME: do not close on file not open
        if not self.capture and self.isVideoFileLoaded:
            self.capture = VideoCapture(self.videoFileName, self.videoControlsWidget)
            self.videoControlsWidget.pauseButton.clicked.connect(self.capture.pause)
            self.mainLayout.addWidget(self.capture)
        self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadVideoFile(self):
        try:
            self.videoFileName = QFileDialog.getOpenFileName(self, 'Select a Video File')
            self.isVideoFileLoaded = True
        except:
            print("Please Select a Video File")

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message', 'Do you really want to exit?', QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
