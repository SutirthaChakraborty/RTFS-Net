name=(
        "lrs2_conf_small_tdanet_context_com"
        "lrs2_conf_small_tdanet_context_com_convrnn"
        "lrs2_conf_small_tdanet_context_com_attention"
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        "ctcnet_small_tdanet_context_06_03_23"
        "ctcnet_small_tdanet_context_convrnn_06_03_23"
        "ctcnet_small_tdanet_context_attention_06_03_23"
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
