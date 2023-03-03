name=(
        "lrs2_conf_small_tdanet"
        "lrs2_conf_small_tdanet_ffn_only"
        "lrs2_conf_small_tdanet_cnn"
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        "ctcnet_small_tdanet_3_3_2023_Attention_Fixed"
        "ctcnet_small_tdanet_3_3_2023_FFN_Only"
        "ctcnet_small_tdanet_3_3_2023_CNN"
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
