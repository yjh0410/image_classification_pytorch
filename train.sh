python train.py --cuda \
                --data_path data/zitai/ \
                --num_classes 2 \
                -m resnet18 \
                -p \
                --norm_type BN \
                --max_epoch 30 \
                --batch_size 16 \
                --img_size 224 \
                --optimizer sgd \
                --lr 1e-3 \
