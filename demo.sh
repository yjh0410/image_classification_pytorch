python train.py --cuda \
                --data_path data/zitai/ \
                --num_classes 2 \
                -m resnet18 \
                --weight weight/best_model.pth \
                --img_size 224
