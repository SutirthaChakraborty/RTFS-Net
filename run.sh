name=(
        lrs2_gridnet_mini_LSTM
        lrs2_gridnet_mini_LSTM_f_first
        lrs2_gridnet_mini_GRU
        lrs2_gridnet_mini_GRU_f_first
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        avnet_mini/gridnet/01_07_23_LSTM
        avnet_mini/gridnet/01_07_23_LSTM_f_first
        avnet_mini/gridnet/01_07_23_GRU
        avnet_mini/gridnet/01_07_23_GRU_f_first
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
