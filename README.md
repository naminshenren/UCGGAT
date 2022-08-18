# UCGGAT
UCGGAT: User Purchase Intention Prediction Based on User Fine-grained Module Click Stream of Product Detail Page

# Framework
![image](https://github.com/naminshenren/UCGGAT/blob/master/pre_trained/cora/ucggat.PNG)

## Overview
Here we provide the implementation of a User Click Graph-Graph Attention Network (UCGGAT) layer in TensorFlow. The repository is organised as follows:
- `data/` put you data here (The data form is a diagram composed of click module flow);
- `models/` contains the implementation of the UCGGAT network (`ucggat.py`);
- `pre_trained/` contains a pre-trained UCGGAT model;


Finally, `execute_ucg.py` puts all of the above together and may be used to execute a full training run on you data by executing `python execute_ucg.py`.


## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Acknowledge
This work was supported by the National Key R&D Program of China under Grant No. 2020AAA0103804 (Sponsor: <a  href ="https://bs.ustc.edu.cn/chinese/profile-74.html">Hefu Liu</a>) and partially supported by grants from the National Natural Science Foundation of China (No.72004021). This work belongs to the University of science and technology of China.

