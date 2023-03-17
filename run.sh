name=(
        "lrs2_conf_small_tdanet2d_ae_2d"
        "lrs2_conf_small_tdanet2d_ae_2d_convmod"
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        "ctcnet_small_tdanet2d_16_03_2023_DFT"
        "ctcnet_small_tdanet2d_16_03_2023_ConvMod"
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
