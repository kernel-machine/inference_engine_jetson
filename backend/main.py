from datetime import datetime
from lib.frame_generator.frame_generator import VideoFileExtractor, CameraStreamExtractor
from lib.frame_cropper.VideoSegmenter import VideoSegmenter
#from lib.inference_module.inference_module import InferenceModule
import cv2
from flask import Flask, Response, make_response, send_from_directory
from flask_cors import CORS
import threading
import argparse
from lib.inference_module.inference_module import InferenceModule
import time
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", action="store_true", default=False)
parser.add_argument("--model", type=str, default="compiler/exported_models/trt_std.ep")
args = parser.parse_args()
# Use this to get videos from dataset
if args.dataset:
    ve = VideoFileExtractor("/dataset")
else:
    # Or uncomment this to get frames from webcam
    ve = CameraStreamExtractor(0)

frames = ve.get_frames()
#inference_module = InferenceModule(fake=not args.dataset, compile=False)
inf_module = InferenceModule(args.model)
last_class_prediction = "Predicting..."
prediction_lock = threading.Lock()

vs = VideoSegmenter()
app = Flask(__name__, static_folder="dist")
CORS(app)


def generate_frames():
    global last_class_prediction
    inference_buffer = []

    useSpatialTracking = True #False -> Temporal, True -> Spatial
    # Spatial tracking
    distance_epsilon = 200
    last_position = 0
    # Temporal tracking
    last_time_tracked = time.time()
    time_epsilon = 2
    color = (255,255,255)

    elaborated_frames = 0
    start_time = time.time()

    cropping_times = []
    inference_times = []
    try:
        while True:
            frame = next(frames)
            start_time = time.time()
            top_left, bottom_right = vs.crop_frame(frame) #Returns rectangle vertices
            cropping_time = time.time() - start_time
            cropping_times.append(cropping_time)

            if not useSpatialTracking:
                delta_time = time.time() - last_time_tracked
                if delta_time > time_epsilon:
                    color = (255,255,255)
                    inference_buffer.clear
            
            if top_left != 0 and bottom_right != 0:
                cropped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:

                    if useSpatialTracking:
                        delta_distance = abs(last_position - top_left[1])

                        if delta_distance > distance_epsilon:
                            color = (255,255,255)
                            inference_buffer.clear()

                    # Update temporal information
                    last_time_tracked = time.time()
                    # Update spatial information
                    last_position = top_left[1]

                    print("CROPPED SHAPE", cropped.shape)
                    cropped = cv2.resize(cropped, (224,224))
                    inference_buffer.append(cropped)

                    if len(inference_buffer) == 32:
                        start_time = time.time()
                        label = inf_module.inference(inference_buffer)
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                        print(f"-----> Label {label}")
                        print(f"Frame rate: {elaborated_frames/(time.time()-start_time)} | Cropping times: {mean(cropping_times)} | Inference times: {mean(inference_times)}")
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
                elaborated_frames += 1
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except KeyboardInterrupt:
        print("Stream stopped by user")

@app.route('/get_label')
def get_label():
    def critical_section():
        with prediction_lock:
            #print(f"Accessing label: {last_class_prediction}")
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