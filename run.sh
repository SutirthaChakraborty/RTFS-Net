name="lrs2_conf_small_tdanet"
config=$name".yml"
exp="ctcnet_small_tdanet_vidbatchnorm"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt