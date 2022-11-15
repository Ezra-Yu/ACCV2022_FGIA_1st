set -e

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=$1


################## 1. Run inference with ViT ##################
for file in `ls checkpoints`
do
    checkpoint_path="checkpoints/$file"
    name=${checkpoint_path: 12: -4}
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
           --dump pkls/$name.pkl \
           --tta \
           --cfg-option test_dataloader.batch_size=32 \
           --launcher pytorch
done