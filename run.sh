name=(
        "lrs2_tdanet2d_mini_galr12_1"
        "lrs2_tdanet2d_mini_galr12_2"
        "lrs2_tdanet2d_mini_galr12_3"
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        "ctcnet_mini/tdanet2d/04_05_23_GALR_12_1_layers"
        "ctcnet_mini/tdanet2d/04_05_23_GALR_12_2_layers"
        "ctcnet_mini/tdanet2d/04_05_23_GALR_12_3_layers"
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
