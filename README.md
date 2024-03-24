# C2F-SemiCD-and-C2FNet::https://chengxihan.github.io/
The Pytorch implementation for:
“[C2F-SemiCD and C2FNet: A coarse-to-fine semi-supervised change detection method based on consistency regularization in High-Resolution Remote-Sensing Images](https://ieeexplore.ieee.org/document/10445496),” IEEE Transactions on Geoscience and Remote Sensing（TGRS）, 2024, DOI: 10.1109/TGRS.2024.3370568.Chengxi Han,Chen Wu,Meiqi Hu,Jiepan Li,Hongruixuan Chen

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

## Revised parameters 
 You can revise related parameters in the `metadata.json` file.  
 
## Training, Test and Visualization Process   

```bash
python trainHCX.py 
python test.py 
python Output_Results.py
```

## Test our trained model result  
You can directly test our model by our provided training weights in  `tmp/WHU, LEVIR, SYSU, and S2Looking `. And make sure the weight name is right. Of course, for different datasets, the `Dataset mean and std setting` is different.
```bash
path = opt.weight_dir+'final_epoch99.pt'
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
 
 And also we provide all test results of our HANet in the HANetTestResult!!!! Download in HANetTestResult or [Baidu Disk](https://pan.baidu.com/s/1nwPYkqtUIKe90KZoT5VO-A?pwd=2023 ),pwd:2023 😋😋😋

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD  
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
## Dataset mean and std setting 
We calculated mean and std for seven data sets in line 27-38 of `utils/datasetHCX` , you can use one directly and then annotate the others.
```bash
# It is for LEVIR!
# self.mean1, self.std1, self.mean2, self.std2 =[0.45025915, 0.44666713, 0.38134697],[0.21711577, 0.20401315, 0.18665968],[0.3455239, 0.33819652, 0.2888149],[0.157594, 0.15198614, 0.14440961]
# It is for WHU!
self.mean1, self.std1, self.mean2, self.std2 = [0.49069053, 0.44911194, 0.39301977], [0.17230505, 0.16819492,0.17020544],[0.49139765,0.49035382,0.46980983], [0.2150498, 0.20449342, 0.21956162]
```
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
 
 Our code is inspired and revised by [pytorch-MSPSNet](https://github.com/QingleGuo/MSPSNet-Change-Detection-TGRS),[pytorch-SNUNet](https://github.com/likyoo/Siam-NestedUNet), Thanks  for their great work!!  



## Reference  
[1] Han, C., Wu, C., Hu, M., Li, J. and Chen, H., 2024. 
“[C2F-SemiCD and C2FNet: A Coarse-to-Fine Semi-Supervised Change Detection Method Based on Consistency Regularization in High-Resolution Remote-Sensing Images](https://ieeexplore.ieee.org/document/10445496),” . IEEE Transactions on Geoscience and Remote Sensing.


[2] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.

[3] Han, C., Wu, C., Guo, H., Hu, M., Li, J. and Chen, H., 2023. 
“[CGNet: Change guiding network: Incorporating change prior to guide change detection in remote sensing imagery](https://ieeexplore.ieee.org/abstract/document/10234560/),” . IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

[4]Han, C., Wu, C. and Du, B., 2023, [HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection](https://ieeexplore.ieee.org/abstract/document/10283341), July. In IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium (pp. 5511-5514). IEEE.



(Don't hesitate to tell me about the latest progress and useful methods in the CD field, I will spare no effort to thank you. Good luck to you guys, I wish we can be the best friend in the CD field.)
