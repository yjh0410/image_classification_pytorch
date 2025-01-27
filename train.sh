# ------------------- Args setting -------------------
MODEL=$1
DATASET_ROOT=$2
WORLD_SIZE=$3
MASTER_PORT=$4
RESUME=$5

# ------------------- Training setting -------------------
BATCH_SIZE=128
GRAD_ACCUM=32
MAX_EPOCH=100
WP_EPOCH=10
EVAL_EPOCH=5
BASE_LR=1e-3
MIN_LR=1e-6

# ------------------- Training pipeline -------------------
if [ $WORLD_SIZE == 1 ]; then
    python train.py --data_path ${DATASET_ROOT} \
                    --model ${MODEL} \
                    --wp_epoch ${WP_EPOCH} \
                    --max_epoch ${MAX_EPOCH} \
                    --eval_epoch ${EVAL_EPOCH} \
                    --batch_size ${BATCH_SIZE} \
                    --grad_accumulate ${GRAD_ACCUM} \
                    --base_lr ${BASE_LR} \
                    --min_lr ${MIN_LR} \
                    --resume ${RESUME}
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
                    --distributed \
                    --data_path ${DATASET_ROOT} \
                    --model ${MODEL} \
                    --wp_epoch ${WP_EPOCH} \
                    --max_epoch ${MAX_EPOCH} \
                    --eval_epoch ${EVAL_EPOCH} \
                    --batch_size ${BATCH_SIZE} \
                    --base_lr ${BASE_LR} \
                    --min_lr ${MIN_LR} \
                    --world_size ${WORLD_SIZE} \
                    --resume ${RESUME} \
                    --sybn
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi