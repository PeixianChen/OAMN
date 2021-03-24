CUDA_VISIBLE_DEVICES=0,1,2 \
python -u train_oamn_market.py \
--data-dir  /home/chenpeixian/reid/dataset/ \
-s market1501_ \
-t Partial_iLIDS \
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
--logs-dir ./logs/onemask/market/ \
# --resume ./size24_5mask13_1scoretype1_maskdetach_resnet/checkpoint.pth.tar \
# --evaluate \
