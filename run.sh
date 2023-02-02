name="lrs2_conf_small_frcnn"
config=$name".yml"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$name"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$name/conf.yml --test-dir data-preprocess/LRS2/tt