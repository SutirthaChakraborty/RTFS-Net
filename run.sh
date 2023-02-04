name="lrs2_conf_large_frcnn"
config=$name".yml"
exp="ctcnet_large_frcnn"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt