python train.py --cuda \
                --data_path /mnt/share/ssd2/dataset/imagenet/ \
                --num_classes 1000 \
                -m elannet_huge \
                --wp_epoch 20 \
                --max_epoch 300 \
                --eval_epoch 10 \
                --batch_size 128 \
                --optimizer adamw \
                --base_lr 4e-3 \
                --min_lr 1e-6 \
                -accu 32 \
                --ema \
                --resume weights/elannet_huge/elannet_huge_epoch_70_73.49.pth
                # --fp16 \

