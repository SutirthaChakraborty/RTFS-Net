name="lrs2_conf_small_frcnn_context_com"
config=$name".yml"
exp="ctcnet_small_frcnn_context_18_02_23"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt
name="lrs2_conf_small_tdanet_context_com"
config=$name".yml"
exp="ctcnet_small_tdanet_context_18_02_23"
echo "Stage 1: Training: python train.py --conf-dir config/"$config
python train.py --conf-dir config/$config
echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$exp"/conf.yml --test-dir data-preprocess/LRS2/tt"
python eval.py --conf-dir ../experiments/audio-visual/$exp/conf.yml --test-dir data-preprocess/LRS2/tt