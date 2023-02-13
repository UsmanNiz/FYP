import cv2
import os

class FrameGenerator:

    def generateFrames(self, output_path: str, video_path: str, sampling_rate=1):
        if not (os.path.exists(output_path)):
            os.mkdir(output_path)

        VID_NAME = video_path.replace("/", ".").split('.')[-2]
        # genVidPath =  video_path.replace("Videos",Geb)
        camera = cv2.VideoCapture(video_path)
        frame_count = 0
        count = 0

        if not (os.path.isdir(os.path.join(output_path, VID_NAME))):
            os.mkdir(os.path.join(output_path, VID_NAME))

            while True:
                (grabbed, frame) = camera.read()

                if not grabbed:
                    break
                if frame_count % sampling_rate == 0:
                    filename = "%s/%s/%06d.png" % (output_path, VID_NAME, count)

                    image = cv2.resize(frame, (122, 122), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(filename, image)
                    count += 1
                frame_count += 1
        return frame_count
        # print(frame_count)


def main():
    fg = FrameGenerator()
    for video in os.listdir("c3d data"):
        if "mp4" in video:
            fg.generateFrames(video_path="c3d data/"+video, output_path="dataloader/cricket/train/")


if __name__ == "__main__":
    print("generating frames")
    main()
