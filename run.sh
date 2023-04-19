name=(
        "lrs2_tdanet2d_small"
        "lrs2_tdanet2d_mini"
        "lrs2_tdanet2d_mini_galr"
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        "ctcnet_small/tdanet2d/19_04_23"
        "ctcnet_mini/tdanet2d/19_04_23"
        "ctcnet_mini/tdanet2d/19_04_23_GALR"
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
