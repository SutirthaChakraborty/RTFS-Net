name="lrs2_conf_small_tdanet"
config=$name".yml"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
name="lrs2_conf_small_frcnn"
config=$name".yml"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
exp="ctcnet_small_tdanet_21_02_23"
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt
exp="ctcnet_small_frcnn_21_02_23"
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt