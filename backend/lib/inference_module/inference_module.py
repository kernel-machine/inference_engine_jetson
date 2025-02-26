import zmq
import numpy as np
import cv2
import struct
import torch
import torchvision
from lib.inference_module.model_vivit import ModelVivit
import time

class InferenceModule:
    def __init__(self, compile:bool=False, fake:bool=True) -> None:
        self.model = ModelVivit(hidden_layers=5)
        self.model = torch.nn.DataParallel(self.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 32 , 3, 224, 224).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=self.device))
        self.model = self.model.module.eval()
        if compile:
            print("Compiling...")
            self.model = torch.compile(self.model)
        self.model(dummy_input)
        self.fake = fake

    def pre_process_images(self, batch_data):
        processed_tensors = []
        for image in batch_data:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            processed_tensors.append(tensor)
        processed_tensors = torch.stack(processed_tensors)
        processed_tensors = processed_tensors.unsqueeze(0)
        processed_tensors = torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(processed_tensors)
        return processed_tensors

    def inference(self, frames:list[cv2.Mat]) -> bool:
        frames = self.pre_process_images(frames).to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            prediction_logits = self.model(frames)
            torch.cuda.synchronize()
            end_time = time.time()
        inference_time = end_time - start_time
        if self.fake:
            print(f"Logits {prediction_logits.flatten().cpu()}")
            return prediction_logits.flatten().cpu()[0]>0.5
        else:
            predicted_classes = torch.sigmoid(prediction_logits).round().flatten().cpu()
            predicted_classes = list(map(lambda x:bool(x),predicted_classes))
            print(f"prediction: {prediction_logits} | {predicted_classes} | Time: {(inference_time*1000)}ms")
            return predicted_classes[0]
            
