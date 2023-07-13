conda activate base
conda remove -n av --all --yes
conda create -n av numpy --yes
conda activate av
pip3 install --upgrade pip
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir 
pip3 install -r requirements.txt --no-cache-dir 