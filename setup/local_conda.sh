conda activate base
conda remove -n av --all --yes
conda env create -f setup/local_requirements.yaml
conda activate av