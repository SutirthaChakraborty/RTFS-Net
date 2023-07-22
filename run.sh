name=(
        lrs2_tdavnet_mini_512
        lrs2_tdavnet_mini_256
        lrs2_tdavnet_mini_128
        lrs2_tdavnet_mini_64
        lrs2_tdavnet_mini
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        avnet_mini/tdavnet/22_07_23
        avnet_mini/tdavnet/22_07_23_64
        avnet_mini/tdavnet/22_07_23_128
        avnet_mini/tdavnet/22_07_23_256
        avnet_mini/tdavnet/22_07_23_512
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
