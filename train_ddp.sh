# 32 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python  -m torch.distributed.run --nproc_per_node=2 train.py --cuda \
                                                            -dist \
                                                            --data_path /mnt/share/ssd2/dataset/imagenet/ \
                                                            --num_classes 1000 \
                                                            -m elannet_nano \
                                                            --wp_epoch 20 \
                                                            --max_epoch 300 \
                                                            --eval_epoch 10 \
                                                            --batch_size 1 \
                                                            --optimizer adamw \
                                                            --base_lr 4e-3 \
                                                            --min_lr 1e-6 \
                                                            -accu 4 \
                                                            --ema \
                                                            # --fp16 \

