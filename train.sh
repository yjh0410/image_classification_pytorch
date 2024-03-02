# ------------------- Args setting -------------------
MODEL=$1
BATCH_SIZE=$2
DATASET_ROOT=$3
WORLD_SIZE=$4
MASTER_PORT=$5
RESUME=$6

# ------------------- Training setting -------------------
MAX_EPOCH=90
WP_EPOCH=-1
EVAL_EPOCH=5
BASE_LR=0.1
MIN_LR=0.0
OPTIMIZER="sgd"

# ------------------- Training pipeline -------------------
if [ $WORLD_SIZE == 1 ]; then
    python train.py --data_path ${DATASET_ROOT} \
                    --model ${MODEL} \
                    --optimizer ${OPTIMIZER} \
                    --wp_epoch ${WP_EPOCH} \
                    --max_epoch ${MAX_EPOCH} \
                    --eval_epoch ${EVAL_EPOCH} \
                    --batch_size ${BATCH_SIZE} \
                    --base_lr ${BASE_LR} \
                    --min_lr ${MIN_LR} \
                    --use_pixel_statistic \
                    --resume ${RESUME}
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
                    --distributed \
                    --data_path ${DATASET_ROOT} \
                    --model ${MODEL} \
                    --optimizer ${OPTIMIZER} \
                    --wp_epoch ${WP_EPOCH} \
                    --max_epoch ${MAX_EPOCH} \
                    --eval_epoch ${EVAL_EPOCH} \
                    --batch_size ${BATCH_SIZE} \
                    --base_lr ${BASE_LR} \
                    --min_lr ${MIN_LR} \
                    --use_pixel_statistic \
                    --world_size ${WORLD_SIZE} \
                    --resume ${RESUME} \
                    --sybn
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi