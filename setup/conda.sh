conda activate base
conda remove -n av --all --yes
conda create -n av numpy pip --yes
conda activate av
pip3 install --upgrade pip
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip3 install -r requirements.txt --no-cache-dir 