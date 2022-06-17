python train.py --cuda \
                --data_path /mnt/share/ssd2/dataset/imagenet/ \
                --num_classes 1000 \
                -m cspd-s \
                --max_epoch 300 \
                --batch_size 256 \
                --img_size 256 \
                --optimizer sgd \
                --base_lr 0.1 \
                --min_lr_ratio 0.05
