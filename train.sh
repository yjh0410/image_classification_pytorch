# python train.py --cuda \
#                 --data_path /mnt/share/ssd2/dataset/imagenet/ \
#                 --num_classes 1000 \
#                 -m cspdarknet_tiny \
#                 --wp_epoch 10 \
#                 --max_epoch 90 \
#                 --eval_epoch 5 \
#                 --batch_size 128 \
#                 --optimizer adamw \
#                 --base_lr 4e-3 \
#                 --min_lr 1e-6 \
#                 -accu 32 \
#                 --ema \

# Train ConvMixer
python train.py --cuda \
                --data_path /mnt/share/ssd2/dataset/imagenet/ \
                --num_classes 1000 \
                -m convmixer_nano \
                --wp_epoch 10 \
                --max_epoch 90 \
                --eval_epoch 5 \
                --batch_size 128 \
                --optimizer adamw \
                --base_lr 0.01 \
                --min_lr 5e-4 \
                --ema \
