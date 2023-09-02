name=(
        lrs2_tdavnet_mini_attn
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        avnet_mini/tdavnet/01_09_23_Attnrnn
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
