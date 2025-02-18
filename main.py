import time
from lib.frame_generator.frame_generator import VideoFileExtractor, CameraStreamExtractor
from lib.frame_cropper.VideoSegmenter import VideoSegmenter
from lib.inference_module.inference_module import InferenceModule
import cv2
from flask import Flask, Response
import queue

# Uncomment for video file extraction
#ve = VideoFileExtractor("/dataset")
ve = CameraStreamExtractor(0)
frames = ve.get_frames()
inference_module = InferenceModule()

vs = VideoSegmenter()
app = Flask(__name__)


def generate_frames():
    buffer = []
    buffer_view = []
    frame_rate = 1/15
    last_run = 0
    while True:
        print("Requesting frames")
        frame = next(frames)
        frame, cropped = vs.crop_frame(frame)
        
        buffer.append((frame, cropped))
        if len(buffer) == 32:
            valid_cropped_frames = list(filter(lambda x: x[1] is not None, buffer))
            if len(valid_cropped_frames) == 32:
                valid_cropped_frames = list(map(lambda x: x[1], valid_cropped_frames))
                print(f"Processing frame {len(valid_cropped_frames)}")
                label = inference_module.inference(valid_cropped_frames)
                print(f"Label: {label}")
            else:
                buffer_view=buffer_view+list(map(lambda x: x[0], buffer))
            buffer.clear()


        if len(buffer_view)>32:
            frame = buffer_view.pop(0)
            _, img_buffer = cv2.imencode('.jpg', frame)
            frame = img_buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            now = time.time()
            elapsed = now - last_run
            if elapsed < frame_rate:
                time.sleep(frame_rate - elapsed)
                last_run = time.time()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Webcam Streaming</h1><img src='/video_feed' width='840' height='480'/>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)