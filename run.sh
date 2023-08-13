name=(
        # tdavnet_q2_likai_cmsm_2chan
        tdavnet_q2_likai_cmsm_2chan_RI
        # tdavnet_q2_likai_cmsm_2chan_direct
)
for n in "${name[@]}"
do
    config=$n".yml"
    echo "Stage 1: Training: python train.py --conf-dir config/"$config
    python train.py --conf-dir config/$config
done
exp=(
        # avnet_mini/tdavnet/08_08_23_cmsm_2chan
        avnet_mini/tdavnet/08_08_23_cmsm_2chan_RI
        # avnet_mini/tdavnet/08_08_23_cmsm_2chan_direct
)
for e in "${exp[@]}"
do
    echo "Stage 2: Evaluation: python eval.py --conf-dir ../experiments/audio-visual/"$e"/conf.yml --test-dir data-preprocess/LRS2/tt"
    python eval.py --conf-dir ../experiments/audio-visual/$e/conf.yml --test-dir data-preprocess/LRS2/tt
done
