python train.py --cuda \
                --data_path /mnt/share/ssd2/dataset/imagenet/ \
                --num_classes 1000 \
                -m cspdarknet53_silu \
                --wp_epoch 10 \
                --max_epoch 90 \
                --eval_epoch 5 \
                --batch_size 128 \
                --optimizer adamw \
                --base_lr 4e-3 \
                --min_lr 1e-6 \
                -accu 32 \
                --ema \
                # --resume weights/elannet_huge/elannet_huge_epoch_70_73.49.pth \
                # --start_epoch 70 \
                # --fp16 \

