import os

from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# from inference import *
#from inference import *
import sys


# video_dir_path = "test_videos/"
# evaluateModelOnSplit(video_dir_path=video_dir_path, model_path="run/1_january_weights/C3D-DesktopAssembly_iter-4800.pth.tar")

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Diamond Cutter")
 
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        self.progressBar=QProgressBar()
        self.label = QLabel()
        self.pixmap = QPixmap("C:/Users/usman/Downloads/fyp.jpeg")
        self.label.setPixmap(self.pixmap)
 
        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.runInferenceButton=QPushButton("Run Inference")
        self.runInferenceButton.setIcon(self.style().standardIcon(QStyle.SP_FileDialogToParent))
 
        self.openButton = QPushButton("Open Video")   
        self.openButton.clicked.connect(self.openFile)
        self.runInferenceButton.clicked.connect(self.runInference)

 
        widget = QWidget(self)
        self.setCentralWidget(widget)
        
        layout = QGridLayout()
        layout.addWidget(self.label,0,0,3,3)
        #layout.addWidget(videoWidget,0,0,2,2)
        layout.addWidget(self.openButton,3,0)
        layout.addWidget(self.playButton,3,1)
        layout.addWidget(self.runInferenceButton,3,2)
 
        widget.setLayout(layout)
        self.mediaPlayer.setVideoOutput(videoWidget)
        
        
      
        widget.setLayout(layout)

    def runInference(self):
        print("istinkja")
        print(str(self.inputVideoPath).split("/")[-2])
        # .split("/")[-2] + "/"
        print(os.getcwd())
        from inference import evaluateModelOnSplit
        print(os.getcwd())
        evaluateModelOnSplit(self,video_dir_path=str(self.inputVideoPath),
                             model_path="run/1_january_weights/C3D-DesktopAssembly_iter-4800.pth.tar")

    def openFile(self):
        self.inputVideoPath, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())
        
        print(self.inputVideoPath)
              

 
    def play(self):
        #put output inference video path here 
        if self.inputVideoPath != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self.inputVideoPath)))#put output inference video path here 
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
 
 
app = QApplication(sys.argv)
videoplayer = VideoPlayer()
videoplayer.resize(640, 480)
videoplayer.show()
sys.exit(app.exec_())