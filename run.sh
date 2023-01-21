###
 # @Author: Kai Li
 # @Date: 2021-06-19 23:53:17
 # @LastEditors: Kai Li
 # @LastEditTime: 2021-08-01 11:46:34
### 
#!/bin/bash
mode="audio-visual" # model is audio or audio-visual
stage=1  # Controls from which stage to start
if [ $mode == "audio" ]; then
    if [[ $stage -le  0 ]]; then
        echo "Stage 0: Generating json files including wav path and duration"    
        python local/preprocess_wham.py --in_dir /home/likai/data1/Dataset/wham/wham/wav8k/min --out_dir local/wham
    fi

    if [[ $stage -le  1 ]]; then
        echo "Stage 1: Training"
        python train.py --exp_dir exp/tmp --conf_dir local/lrs2_conf.yml
    fi

    # if [[ $stage -le  2 ]]; then
    #     echo "Stage 2: Evaluation"
    #     python eval.py --test_dir /home/likai/data1/src/egs/convtasnet/local/wham/tt/
    # fi
fi

if [ $mode == "audio-visual" ]; then
    if [[ $stage -le  0 ]]; then
        echo "Stage 0: Generating json files including wav path and duration"    
        python data-preprocess/preprocess_lrs2.py --in_dir /home/likai/data1/Dataset/wham/wham/wav8k/min --out_dir local/wham
    fi

    if [[ $stage -le  1 ]]; then
        echo "Stage 1: Training"
        python train.py --conf-dir config/lrs2_conf_64_64_3_adamw_1e-1_blocks8_pretrain.yml
    fi

    if [[ $stage -le  2 ]]; then
        echo "Stage 2: Evaluation"
        python eval.py --conf-dir /home/anxihao/data2/av-experiments/ctcnet_pretrain_baseline_1_2/conf.yml --test-dir data-preprocess/LRS2/tt
    fi
fi