conda remove -n av --all --yes
source ~/.bashrc
conda create -n av pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
source ~/.bashrc
conda activate av
python -m pip3 install --upgrade pip3
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -r requirements.txt --no-cache-dir 
