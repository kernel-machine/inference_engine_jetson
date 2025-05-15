from datetime import datetime
from lib.frame_generator.frame_generator import VideoFileExtractor, CameraStreamExtractor
from lib.frame_cropper.VideoSegmenter import VideoSegmenter
#from lib.inference_module.inference_module import InferenceModule
import cv2
import argparse
from lib.inference_module.inference_module import InferenceModule
from lib.frame_generator.frame_buffer import FrameBuffer
from lib.validation_metric import ValidationMetrics
import time
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str, default="compiler/exported_models/trt_std.ep")
args = parser.parse_args()

inf_module = InferenceModule(args.model)
vs = VideoSegmenter()
validation_metrics = {
    "per_segment": ValidationMetrics(),
    "most_common": ValidationMetrics(),
    "true_inside": ValidationMetrics(),
}   

for video in glob.glob(os.path.join(args.dataset,"*","*.mkv")):
    cap = cv2.VideoCapture(video)
    fb = FrameBuffer(32)
    is_infested = "infested" in video.split(os.path.sep)[-2]
    print(f"Processing {video} -> Class {is_infested}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        top_left, bottom_right = vs.crop_frame(frame)
        if top_left != 0 and bottom_right != 0:
            cropped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
            frame = cv2.resize(frame, (224,224))
            fb.append(frame)
            print(f"Filling buffer {len(fb)} | {len(fb)//32} segments", end="\r")
    print()
    
    predictions = []
    for segment in fb.get_segments():
        prediction = inf_module.inference(segment)
        validation_metrics["per_segment"].add_prediction(prediction, is_infested)
        predictions.append(prediction)

    true_inside = True in predictions
    most_common = max(set(predictions), key=predictions.count)
    validation_metrics["true_inside"].add_prediction(true_inside, is_infested)
    validation_metrics["most_common"].add_prediction(most_common, is_infested)

    print(f"Predictions: {predictions} -> Label: {is_infested} -> True inside: {true_inside} -> Most common: {most_common}")

    
    print(f"Per segment: {validation_metrics['per_segment']}")
    print(f"Most common: {validation_metrics['most_common']}")
    print(f"True inside: {validation_metrics['true_inside']}")

