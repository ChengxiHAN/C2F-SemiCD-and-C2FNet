# C2F-SemiCD-and-C2FNet:https://chengxihan.github.io/
# C2F-SemiCD：A Semi-Supervised CD method
# C2FNet：A Supervised CD method


The Pytorch implementation for:
“[C2F-SemiCD and C2FNet: A coarse-to-fine semi-supervised change detection method based on consistency regularization in High-Resolution Remote-Sensing Images](https://ieeexplore.ieee.org/document/10445496),” IEEE Transactions on Geoscience and Remote Sensing（TGRS）, 2024, DOI: 10.1109/TGRS.2024.3370568.Chengxi Han,Chen Wu,Meiqi Hu,Jiepan Li,Hongruixuan Chen

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-levir-cd)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-levir-cd-1)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-1?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-levir-cd-2)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-2?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-levir-cd-3)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-3?p=c2f-semicd-a-coarse-to-fine-semi-supervised)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-whu-5)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-5?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-whu-10)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-10?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-whu-20)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-20?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/semi-supervised-change-detection-on-whu-40)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-40?p=c2f-semicd-a-coarse-to-fine-semi-supervised)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-whu-cd)](https://paperswithcode.com/sota/change-detection-on-whu-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-levir-cd)](https://paperswithcode.com/sota/change-detection-on-levir-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-sysu-cd)](https://paperswithcode.com/sota/change-detection-on-sysu-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-s2looking)](https://paperswithcode.com/sota/change-detection-on-s2looking?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-cdd-dataset-season-1)](https://paperswithcode.com/sota/change-detection-on-cdd-dataset-season-1?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-dsifn-cd)](https://paperswithcode.com/sota/change-detection-on-dsifn-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-googlegz-cd)](https://paperswithcode.com/sota/change-detection-on-googlegz-cd?p=c2f-semicd-a-coarse-to-fine-semi-supervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2f-semicd-a-coarse-to-fine-semi-supervised/change-detection-on-levir)](https://paperswithcode.com/sota/change-detection-on-levir?p=c2f-semicd-a-coarse-to-fine-semi-supervised)




![image-20230415](/picture/C2F-SemiCD-C2FNet.png)
![image-20230415](/picture/Visualization.png)

### Requirement  
```bash
-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  
```

## Training, Test and Visualization Process   

```bash
1.Semi-supervised training:
python train_C2F-SemiCD.py --epoch 2 --batchsize 16 --gpu_id '1' --data_name 'WHU' --train_ratio 0.05 --model_name 'SemiModel_noema04'

2.Fully supervised training:
python train_C2FNet.py --epoch 2 --batchsize 16 --gpu_id '1' --data_name 'WHU' --train_ratio 0.05 --model_name 'SemiModel_noema04'

3.Ablation experiment training:
python train_C2F-SemiCD_Ablation.py --epoch 2 --batchsize 16 --gpu_id '1' --data_name 'WHU' --train_ratio 0.05 --model_name 'SemiModel_noema04'

1.Semi-supervised testing:
python test_C2F-SemiCD.py --gpu_id '2' --data_name 'WHU' --model_name 'SemiModel_noema04'

2.Fully supervised testing:
python test_C2FNet.py --gpu_id '2' --data_name 'WHU' --model_name 'SemiModel_noema04'

3.Ablation experiments test:
python test_Ablation.py --gpu_id '2' --data_name 'WHU' --model_name 'SemiModel_noema04'

4.Feature visualization:
python test_visualisation.py --gpu_id '2' --data_name 'WHU' --model_name 'SemiModel_noema04'

```

## Test our trained model result  
You can directly test our model by our provided training weights in  `out/`. And make sure the weight name is right. Of course, for different methods and datasets, the `Dataset mean and std setting` is different.
```bash
path = opt.weight_dir+'final_epoch99.pt'
parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/WHU-5/')  # Semi-supervised models save paths！！
parser.add_argument('--save_path', type=str, default='./output/C2FNet/WHU-5/')  # Fully supervised models save paths！！
```

## Dataset Download   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 
 WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html ,our paper split in [Baidu Disk](https://pan.baidu.com/s/16g3H1UsDMgqmXaVjiE319Q?pwd=6969),pwd:6969
 
SYSU-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1p0QfogZm4BM0dd1a0LTBBw?pwd=2023),pwd:2023

S2Looking-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1wAXPHhCLJTqPX0pC2RBMsg?pwd=2023),pwd:2023

CDD-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1cwJ0mEhcrbCWOJn5n-N5Jw?pwd=2023),pwd:2023

DSIFN-CD: Our split in [Baidu Disk]( https://pan.baidu.com/s/1-GD3z_eMoQglSJoi9P-6gw?pwd=2023),pwd:2023

 Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it.
 ![image-20230415](/picture/GoogleGZ-CD.gif)
 ![image-20230415](/picture/WHU-CD.gif)
 ![image-20230415](/picture/LEVIR-CD.gif)
 
 And also we provide all test results of our C2F-SemiCD and C2FNet in the output!!!! Download in output or [Baidu Disk](),pwd:2023 😋😋😋

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD  or GoogleGZ-CD
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  ```        
 Where A contains images of the first temporal image, B contains images of the second temporal images, and the label contains ground truth maps.  
 
## Quantization accuracy
![image-20230415](/picture/C2F-SemiCD-GoogleGZ.png)
![image-20230415](/picture/C2F-SemiCD-WHU.png)
![image-20230415](/picture/C2F-SemiCD-LEVIR.png)
![image-20230415](/picture/C2F-SemiCD-WHUsup-LEVIRunsup.png)
![image-20230415](/picture/C2F-SemiCD-LEVIRsup-WHUunsup.png)

## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{10445496,
  author={Han, Chengxi and Wu, Chen and Hu, Meiqi and Li, Jiepan and Chen, Hongruixuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={C2F-SemiCD: A Coarse-to-Fine Semi-Supervised Change Detection Method Based on Consistency Regularization in High-Resolution Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-21},
  keywords={Feature extraction;Training;Remote sensing;Decoding;Semisupervised learning;Data models;Predictive models;Attention mechanism;deep learning;high-resolution remote sensing images;semi-supervised change detection (CD)},
  doi={10.1109/TGRS.2024.3370568}}

```
## Acknowledgments
 
 Thanks for my co-authors [Jiepan Li](https://henryjiepanli.github.io/Jiepanli_Henry.github.io/),[Haonan Guo](https://www.poleguo98.top/), Thanks  for their great work!!  



## Reference  
[1] Han, C., Wu, C., Hu, M., Li, J. and Chen, H., 2024. 
“[C2F-SemiCD and C2FNet: A Coarse-to-Fine Semi-Supervised Change Detection Method Based on Consistency Regularization in High-Resolution Remote-Sensing Images](https://ieeexplore.ieee.org/document/10445496),” . IEEE Transactions on Geoscience and Remote Sensing.


[2] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.

[3] Han, C., Wu, C., Guo, H., Hu, M., Li, J. and Chen, H., 2023. 
“[CGNet: Change guiding network: Incorporating change prior to guide change detection in remote sensing imagery](https://ieeexplore.ieee.org/abstract/document/10234560/),” . IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

[4]Han, C., Wu, C. and Du, B., 2023, [HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection](https://ieeexplore.ieee.org/abstract/document/10283341), July. In IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium (pp. 5511-5514). IEEE.



(Don't hesitate to tell me about the latest progress and useful methods in the CD field, I will spare no effort to thank you. Good luck to you guys, I wish we can be the best friend in the CD field.)
