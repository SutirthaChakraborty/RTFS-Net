name=(
        lrs2_tdavnet_mini
        lrs2_tdavnet_mini_gridnetdpt
        lrs2_tdavnet_mini_gridnetdpt_time
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        avnet_mini/tdavnet/06_08_23
        avnet_mini/tdavnet/06_08_23_gridnetdpt
        avnet_mini/tdavnet/06_08_23_gridnetdpt_time
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
