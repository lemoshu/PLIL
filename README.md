# PLIL
MICCAI'23: Towards Expert-Amateur Collaboration: Prototypical Label Isolation Learning for Left Atrium Segmentation with Mixed-Quality Labels

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

## Abstract
Deep learning-based medical image segmentation usually requires abundant high-quality labeled data from experts, yet, it is often infeasible in clinical practice. Without sufficient expert-examined labels, the supervised approaches often struggle with inferior performance. Unfortunately, directly introducing additional data with low-quality cheap annotations (e.g., crowdsourcing from non-experts) may confuse the training. To address this, we propose a Prototypical Label Isolation Learning (PLIL) framework to robustly learn left atrium segmentation from scarce high-quality labeled data and massive low-quality labeled data, which enables effective expert-amateur collaboration. Particularly, PLIL is built upon the popular teacher-student framework. Considering the structural characteristics that the semantic regions of the same class are often highly correlated and the higher noise tolerance in the high-level feature space, the self-ensembling teacher model isolates clean and noisy labeled voxels by exploiting their relative feature distances to the class prototypes via multi-scale voting. Then, the student follows the teacher's instruction for adaptive learning, wherein the clean voxels are introduced as supervised signals and the noisy ones are regularized via perturbed stability learning, considering their large intra-class variation. 

## Requirements
Some important required packages include:
* Pytorch version >=0.4.1.
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy, etc. Please check the requirement.txt.

## Usage

1. HQ-LQ Data Preparation:
Refer to ./data for various HQ-LQ label simulation.


2. Training script
```
cd ./code
python train_PLIL_voting_3D.py --labeled_num {HQ num} --gpu 0 --root_path {path/to/simulated folder}
```

3. Test script 
```
cd ./code
python test_3D.py --exp {path/to/exp/folder} --model vnet_MTPD_voting
```

## Citation
If our work brings some insights to you, please cite our paper as:
```
@artical{xu2023PLIL,
  title={Towards Expert-Amateur Collaboration: Prototypical Label Isolation Learning for Left Atrium Segmentation with Mixed-Quality Labels},
  author={Zhe Xu, Jiangpeng Yan, Donghuan Lu, Yixin Wang, Jie Luo, Yefeng Zheng and Xiu Li},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2023}
}
```   
