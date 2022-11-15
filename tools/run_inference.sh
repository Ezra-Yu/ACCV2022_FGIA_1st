#!/bin/sh
set -e
declare -A dict
dict=(['l-448-arc-rounda3-0.4.pth']="7473" \
      ['l-448-arc-rounda3-0.5.pth']="7418" \
      ['l-448-arc-roundb1.pth']='7520' \
      ['l-448-arc-roundb2.pth']='7570' \
      ['l-448-arc-roundb3.pth']='7620' \
      ['l-448-ce-rounda3-0.4.pth']='7471' \
      ['l-448-ce-rounda3-0.5.pth']='7405' \
      ['l-448-ce-roundb1.pth']='7450' \
      ['l-448-ce-roundb2.pth']='7500' \
      ['l-448-ce-roundb3.pth']='7550')

echo ${dict['l-448-ce-roundb3.pth']}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}


for file in `ls checkpoints`
do  
    ### ViT ###
    result=$(echo $file | grep "448")
    echo $result
    if [[ "$result" != "" ]]
    then 
        echo "=> find ViT-448 checkpoint"
        checkpoint_path="checkpoints/$file"
        name=${checkpoint_path: 12: -4}
        acc=${dict[$file]}
        pkl_name=pkls/$name-$acc.pkl
        echo "=> pkl name is $pkl_name"
        config_path=configs/vit/$name.py
        echo "=> running inference on $checkpoint_path"
        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            tools/infer_folder.py \
            $config_path \
            $checkpoint_path \
            ./data/ACCV_workshop/testb \
            --dump $pkl_name \
            --tta \
            --cfg-option test_dataloader.batch_size=$BATCH_SIZE \
            --launcher pytorch 
    else
        echo "=> find Swin-384 checkpoint"
        # TODO: implement swin here
    fi
done