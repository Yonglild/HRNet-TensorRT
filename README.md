# HRNet-Semantic-Segmentation

This repo implemtents [HRNet-Semantic-Segmentation-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1) with TensorRT7 network definition API 

The code refers to https://github.com/wang-xinyu/tensorrtx/tree/master/hrnet

If you want to know more about HRNet, please click [HRNet-segmentation的网络结构分析](https://blog.csdn.net/sinat_38685124/article/details/119787868)

## How to Run
### For HRNet-Semantic-Segmentation-v1.1
1. generate .wts, use config `experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` and pretrained weight `hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth` as example. change `PRETRAINED` in `experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` to `""`.
```
cp gen_wts.py $HRNET--Semantic-Segmentation-PROJECT-ROOT/tools
cd $HRNET--Semantic-Segmentation-PROJECT-ROOT
python tools/gen_wts.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml --ckpt_path hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth --save_path hrnet_w48.wts
cp hrnet_w48.wts $HRNET-TENSORRT-ROOT
cd $HRNET-TENSORRT-ROOT
```
2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
  first serialize model to plan file
  ```
  ./hrnet -s [.wts] [.engine] [small or 18 or 32 or 48] # small for W18-Small-v2, 18 for W18, etc.
  ```
  such as
  ```
  ./hrnet -s ../hrnet_w48.wts ./hrnet_w48.engine 48
  ```
  then deserialize plan file and run inference
  ```
  ./hrnet -d  [.engine] [image dir]
  ```
  such as 
  ```
  ./hrnet -d  ./hrnet_w48.engine ../samples
  ```
## Result

TRT Result:

![0_false_color_map](0_false_color_map.png)

pytorch result:

![frankfurt_000001_058914_leftImg8bit_segtorch1](frankfurt_000001_058914_leftImg8bit_segtorch1.png)
