python train.py --cuda \
                --data_path /data/datasets/imagenet-1k/ \
                --num_classes 1000 \
                -m elannet_pico \
                --wp_epoch 10 \
                --max_epoch 100 \
                --eval_epoch 5 \
                --batch_size 128 \
                --optimizer adamw \
                --grad_accumulate 32 \
                --ema \
