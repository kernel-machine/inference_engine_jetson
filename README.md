## EV2: Edge Vision against Varroa - Jetson Inference
Software to run inference on a Nvidia Jetson Orin nano with Balena

The Varroa destructor mite threatens honey bee populations. We introduce a deep learning framework that analyzes video clips of bees for accurate, 98% detection of infestations. Our method outperforms existing techniques, offering a scalable, non-invasive solution for early detection, reducing chemical use, and supporting sustainable beekeeping.

## How to run
### Download the code
Clone the code of the repository
```bash
git clone https://github.com/kernel-machine/inference_engine_jetson.git
```
### Download the pretrained model
Download the pretrained model from [here](https://drive.google.com/file/d/1OCxnKLH7f3-Sg_QMEOJDJu7KXMlvFL2W/view?usp=sharing), the file is named `trt_std.ep` and must be placed in the `backend` directory
### Build and push the container
The software can be deployed with Balena on a Nvidia Jetson Orin Nano running this command:
```bash
balena push fleet_name
```
### Access to the service
The device exposes an HTTP server on the port 80 which can be accessible with a web browser at the url
`http://<DEVICE_IP>`.

The local IP is visualized also in the Balena dashboard, otherwise enable the public device URL, the user can enter in the service also with the generated URL.

## Model
The preatrained model can be downloaded [here](https://drive.google.com/file/d/1OCxnKLH7f3-Sg_QMEOJDJu7KXMlvFL2W/view?usp=sharing)

## Project
Additional information about the research project [here](https://alcorlab.diag.uniroma1.it/projects/ev2)

## Acknowledgments
This work was (partially) supported by project no. 202277 WMAE CUP B53D23012820006, “EdgeVision against Varroa (EV2): Edge computing in defence of bees” funded by the Italian’s MUR PRIN2022 (ERC PE6) research program.

## Licence
The code, models and dataset used by this repository subject to the licence [Attribution-NonCommercial-NoDerivatives 4.0 International](LICENCE.md)

