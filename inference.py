from enum import Enum

import os
import time
import cv2
import numpy as np
from utilss import *
import torch
from collections import deque
from network import C3D_model

       
from yolov5.detect import *

label_dict = {
    "2":"Balling",
    "1":"Batting",
    "0":"Background"
}

# counter = 0

class Segment_Classifier(object):
	
	def __init__(self, model_path, _resize_height = 112, _resize_width = 112):

		self._resize_height = _resize_width
		self._resize_width  = _resize_height
		self.crop_size = 112
		self.buffer_deque = deque()
		self.segment_length = 16
		self.Model =  C3D_model.C3D(num_classes=23, pretrained=False)
		checkpoint = torch.load('{}'.format(model_path), map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
		self.Model.load_state_dict(checkpoint['state_dict'])
		self.Model.cuda()
		self.Model.eval()
		
	def store_current_frame(self, img):
		
		assert len(self.buffer_deque) <= self.segment_length, "Inconsistent length of Deque"

		if len(self.buffer_deque) == self.segment_length:
            # pop last frame and add current frame
			self.buffer_deque.popleft()
			frame = np.array(cv2.resize(img, (self._resize_width, self._resize_height))).astype(np.float64)
			self.buffer_deque.append(frame)
		
		else:

			frame = np.array(cv2.resize(img, (self._resize_width, self._resize_height))).astype(np.float64)
			self.buffer_deque.append(frame)
	
	def inference(self):
		
		buffer = normalize(np.array(self.buffer_deque))
		buffer = crop(buffer, self.crop_size)
		buffer = to_tensor(buffer)
		buffer = np.expand_dims(buffer, axis = 0)
		buffer = torch.from_numpy(buffer).float().to('cuda')
		output = self.Model(buffer)
		probs = torch.nn.Softmax(dim=1)(output)
		preds = torch.max(probs, 1)[1]
		return preds.item()


def evaluateModelOnSplit(video_player_obj,video_dir_path,model_path):

    # for videos in os.listdir(video_dir_path):

        # cap = cv2.VideoCapture(video_dir_path + videos)

    cap = cv2.VideoCapture(video_dir_path )


    classifier = Segment_Classifier(model_path)
    # Check whether user selected camera is opened successfully.

    if not (cap.isOpened()):
        print('Could not open video device')

    count = 0
    inc = 0
    fps_time = 0
    end_of_video = False

    y_pred = []

    Detect = Detector()
    Detect.load_model()
    Detect.data_loader()
    while (not end_of_video):
        ret, frame = cap.read()

        if (ret):

            classifier.store_current_frame(frame)

            if inc < 16:
                pred = 0
            elif (inc >= 16):

                #detector
                ##
                # main(frame)


                ##
                # print("IN detector")
                result = classifier.inference()
                pred = result
                print("in C3d")
                Detect.inference(frame,pred,video_player_obj)


            # print(pred)
            y_pred.append(pred)

            frame = cv2.resize(frame,(640,480))
            try:

                cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Class: %s" % (label_dict[str(pred)]),(320,20),cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 1)
            except:
                pass
            # cv2.imshow("Cricket Test Videos",frame)
            k = cv2.waitKey(1)
            # if k == ord("q"):
            #     exit()

        else:
            end_of_video = True


        count += 1
        inc += 1
        fps_time = time.time()

    Detect.release_video()
    cap.release()

if __name__ == '__main__':

    video_dir_path = "test_videos/"
    # evaluateModelOnSplit(video_dir_path=video_dir_path, model_path="run/1_january_weights/C3D-DesktopAssembly_iter-4800.pth.tar")

