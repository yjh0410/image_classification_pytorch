python train.py --cuda \
                --data_path /mnt/share/ssd2/dataset/imagenet/ \
                --num_classes 1000 \
                -m elannet \
                --wp_epoch 20 \
                --max_epoch 300 \
                --batch_size 1 \
                --optimizer adamw \
                --base_lr 4e-3 \
                --min_lr 1e-6 \
                -accu 32 \
                --ema \
                # --fp16 \
                # --resume weights/cspd-s/cspd-s_epoch_41_67.70.pth

