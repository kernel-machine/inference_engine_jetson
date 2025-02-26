from datetime import datetime
from lib.frame_generator.frame_generator import VideoFileExtractor, CameraStreamExtractor
from lib.frame_cropper.VideoSegmenter import VideoSegmenter
from lib.inference_module.inference_module import InferenceModule
import cv2
from flask import Flask, Response, make_response, send_from_directory
from flask_cors import CORS
import threading
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", action="store_true", default=False)
args = parser.parse_args()
# Use this to get videos from dataset
if args.dataset:
    ve = VideoFileExtractor("/media/luca/Home/datasets/prin/raw/dataset_v2_raw_s32_val")
else:
    # Or uncomment this to get frames from webcam
    ve = CameraStreamExtractor(0)

frames = ve.get_frames()
inference_module = InferenceModule(fake=not args.dataset)
last_class_prediction = "Predicting..."
prediction_lock = threading.Lock()

vs = VideoSegmenter()
app = Flask(__name__, static_folder="dist")
CORS(app)


def generate_frames():
    global last_class_prediction
    inference_buffer = []
    distance_epsilon = 300
    last_position = (0,0)
    color = (255,255,255)
    try:
        while True:
            frame = next(frames)
            
            top_left, bottom_right = vs.crop_frame(frame) #Returns rectangle vertices
            if top_left != 0 and bottom_right != 0:
                #print(f"Dist: {math.dist(last_position, top_left)}")
                if math.dist(last_position, top_left) > distance_epsilon:
                    last_position = top_left
                    color = (255,255,255)
                    inference_buffer.clear()
                cropped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    cropped = cv2.resize(cropped, (224,224))
                    inference_buffer.append(cropped)
                    if len(inference_buffer) == 32:
                        label = inference_module.inference(inference_buffer)
                        print(f"-----------------------------------> Label {label}")
                        with prediction_lock:
                            last_class_prediction = ("Varroa" if label else "No Varroa")+" ("+datetime.now().strftime("%H:%M:%S")+")"
                            color = (0,0,255) if label else (0,255,0)
                        inference_buffer.clear()
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, 20)
            
            _, img_buffer = cv2.imencode('.jpg', frame)
            if not _:
                continue
            frame = img_buffer.tobytes()
            if frame:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #time.sleep(frame_rate)
    except KeyboardInterrupt:
        print("Stream stopped by user")

@app.route('/get_label')
def get_label():
    def critical_section():
        with prediction_lock:
            print(f"Accessing label: {last_class_prediction}")
            return last_class_prediction
    resp = make_response(critical_section())
    resp.mimetype = "text/plain"
    return resp


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory("dist", filename)

@app.route('/')
def index():
    return send_from_directory("dist", "index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)