conda activate base
conda remove -n av --all --yes
conda create -n av numpy --yes
conda activate av
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt