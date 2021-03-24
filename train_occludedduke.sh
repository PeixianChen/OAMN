CUDA_VISIBLE_DEVICES=0,1,2 \
python -u train_oamn.py \
--data-dir  /home/chenpeixian/reid/dataset/ \
-s occludedduke \
-t occludedduke \
--margin 0.5 \
--num-instance 2 \
-a resnet50 \
-b 64 \
--height 256 \
--width 128 \
--logs-dir ./logs/ \
--epoch 30 \
--workers=8 \
--features 2048 \
--logs-dir ./logs/onemask/occludedduke/ \
# --resume ./size24_5mask13_1scoretype1_maskdetach_resnet/checkpoint.pth.tar \
# --evaluate \
