conda activate base
conda remove -n av --all --yes
conda create -n av numpy pip pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda activate av
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
pip install -r setup/requirements.txt --no-cache-dir