docker build -t compile_model -f Dockerfile.compile ..
mkdir -p exported_models
docker run -d --name model_compiler -it --rm --ipc=host -v $(pwd)/exported_models:/exported_models compile_model 
docker exec model_compiler bash -c "python model_download.py --model model.pth --hidden_layers 5"
docker exec model_compiler bash -c "python model_compile.py --model model.pth --hidden_layers 5 && cp trt.ep /exported_models/"
docker stop model_compiler
