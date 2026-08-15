call .\venv\Scripts\activate

pip uninstall -y onnxruntime onnxruntime-gpu

Start https://developer.nvidia.com/cuda-12-9-0-download-archive

pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install nvidia-cudnn-cu12