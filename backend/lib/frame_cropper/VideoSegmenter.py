import cv2
import argparse
import os
import numpy as np
from collections.abc import Generator
import time
import random

BOX_SIZE = 224

class VideoSegmenter:
    def __init__(self, output_size=224, show_debug:bool=False) -> None:
        self.output_size = output_size
        self.show_debug = show_debug
        self.processed_frame_counter = 0
        self.background = None
        self.fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        if self.show_debug:
            video_id = os.path.basename(self.video_path).split(" ")[0]
            video_name = f"video_{video_id}_{int(time.time_ns()/1000)}_debug.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vwriter = cv2.VideoWriter(video_name, fourcc, 20, (2*output_size, output_size*4))
            self.vwriter.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

    def crop_frame(self, frame:cv2.Mat) -> tuple[int,int]:
        # Get Saturation and Value mask
        (frame_h,frame_w) = frame.shape[:2]
        frame_blur = cv2.GaussianBlur(frame, (7,7), 10)
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        saturation_channel = cv2.normalize(frame_hsv[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
        _, mask_saturation = cv2.threshold(saturation_channel, 100, 255, cv2.THRESH_BINARY) # For saturation
        
        value_channel = cv2.normalize(frame_hsv[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
        _, mask_value = cv2.threshold(value_channel, 60, 255, cv2.THRESH_BINARY_INV)

        fgmask = self.fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        mask =  cv2.bitwise_and(fgmask, mask_value)
        mask = cv2.bitwise_and(mask, mask_saturation)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Fill small holes to remove noise
        
        for c in contours:
            if cv2.boundingRect(c)[2]<25:
                mask = cv2.drawContours(mask, [c], -1, 0, cv2.FILLED)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.boundingRect(x)[2] > 35, contours))
        if len(contours) > 0:
            areas = list(map(lambda x:cv2.contourArea(x), contours))
            max_index = np.argmax(areas)
            if areas[max_index] > 0: #Take the largest hole
                c = contours[max_index]

                x, y, w, h = cv2.boundingRect(c)
                center = (x+(w//2),y+(h//2))
                top_left = (center[0]-BOX_SIZE//2,center[1]-BOX_SIZE//2)
                bottom_right = (center[0]+BOX_SIZE//2,center[1]+BOX_SIZE//2)
                if bottom_right[1]>frame_h:
                    out_size = bottom_right[1]-frame_h
                    top_left = (top_left[0], top_left[1]-out_size)
                if top_left[1] < 0:
                    bottom_right = (bottom_right[0], bottom_right[1]-top_left[1])
                    top_left = (top_left[0], 0)                        
                x1, y1 = top_left
                x2, y2 = bottom_right
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame_w), min(y2, frame_h)
                return (x1,y1),(x2,y2)
                # cropped_frame = frame[y1:y2, x1:x2].copy()
                # frame = cv2.circle(frame, center, 5, (0,0,255), 2)
                # cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 20)
                # return frame
                # frame = cv2.putText(frame, f"Area: {areas[max_index]}", bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # return frame, cropped_frame
        return 0,0 
        '''
        if self.show_debug and self.vwriter is not None:
            mask_saturation = cv2.cvtColor(mask_saturation, cv2.COLOR_GRAY2BGR)
            mask_value = cv2.cvtColor(mask_value, cv2.COLOR_GRAY2BGR)
            motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = np.vstack([mask_saturation, mask_value, motion_mask, mask,frame])
            frame = cv2.resize(frame, (2*self.output_size, self.output_size*4))
            #self.vwriter.write(frame)
            cv2.imshow("Preview",frame)
            cv2.waitKey(1000//50)
        '''



if __name__ == "__main__":
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Input video to segmet")
    args = parser.parse_args()
    vs = VideoSegmenter( output_size=224)
    videos = glob.glob(os.path.join(args.video,"*.mkv"))
    random.shuffle(videos)
    for video_path in videos:
        grabber = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=True)
        while True:
            ret, frame = grabber.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
            
            top_left,bottom_right = vs.crop_frame(frame)
            if top_left != 0 and bottom_right != 0:
                frame = cv2.rectangle(frame, top_left, bottom_right, (255,255,255), 20)


            cv2.imshow("HSV Orange masked",frame)
            cv2.waitKey(1000//25)
        grabber.release()
