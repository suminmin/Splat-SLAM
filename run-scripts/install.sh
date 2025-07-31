#!/bin/bash

python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install -e thirdparty/simple-knn/
python -m pip install -e thirdparty/evaluate_3d_reconstruction_lib/

python -c "import torch; import lietorch; import simple_knn; import
diff_gaussian_rasterization; print(torch.cuda.is_available())"

python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install pytorch-lightning==1.9 --no-deps


apt update
apt install libgl1-mesa-glx -y

pip uninstall opencv-python -y
# rm -dr /usr/local/lib/python3.8/dist-packages/cv2
rm -dr /opt/conda/lib/python3.*/site-packages/cv2 
pip install opencv-python-headless

pip install Werkzeug==2.2.2

pip install einops
