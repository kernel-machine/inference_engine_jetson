FROM docker.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y libopenmpi-dev git htop nvtop btop python3.11 python3.11-venv libgl1 libglib2.0-0 sudo v4l-utils

WORKDIR /workspace
COPY backend /workspace
COPY frontend/dist /workspace/dist

RUN python3.11 -m venv env
RUN env/bin/pip install -r requirements.txt

EXPOSE 5000

#CMD [ "env/bin/python","main.py" ]
CMD ["bash"]

