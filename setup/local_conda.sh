conda activate base
conda remove -n av --all --yes
conda env create -f setup/local_requirements.yml
conda activate av