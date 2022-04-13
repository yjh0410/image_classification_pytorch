python train.py --cuda \
                --data_path \
                --num_classes 10 \
                -m resnet18 \
                -p \
                --norm_type BN \
                --max_epoch 30 \
                --batch_size 64 \
                --img_size 224 \
