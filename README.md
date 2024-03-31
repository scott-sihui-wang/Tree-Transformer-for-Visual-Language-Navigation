# Tree Transformer for Visual Language Navigation

## 1. Introduction

### 1.1 Background and Motivation

The aim of this project is to incorporate syntactic information into VLN (`Visual Language Navigation`) networks to improve cross modality alignment and decision making quality.

`Tree transformer` proposed to impose `hierarchical constraints` and `constituent attention` in transformers, so that low layers only have short-ranges attention, which gradually merge into long-ranged attention in high layers. This is to mimic the tree structures that we usually obtain when we parse a sentence. It is believed that tree transformer helps transformer better understand the grammar structures of the text.

In this project, we added `hierarchical constraints` and `constituent attention` to `VLN Bert` to see if `tree transformer` helps improve understanding of syntactic information and leads to better performance.

### 1.2 Project Overview

The project is hosted on `Colab`. Please follow the [link](https://drive.google.com/drive/folders/11PMBFEDVkjrm4O2td1NitIy0Z41TFcSh?usp=sharing) to check out the full details of this project.

To setup the project and replicate the experimental results, please check out the `Colab` pages: 

- [experiments with LSTM Baseline Model](https://colab.research.google.com/drive/1ii_f83InJxKFnvwDk3n0w8eNeEulwv83?usp=sharing);
  
- [experiments with LSTM Tree Model](https://colab.research.google.com/drive/1u_vp1ye6PqmSCn7WsZT5uW3_-2PV7KBW?usp=sharing);
  
- [experiments with VLN Bert Model](https://colab.research.google.com/drive/1zdbEnWL8yf7YFZpsNqNtYsdCXYLTTSWU?usp=sharing);
  
- [experiments with VLN Bert Model with Tree Transformer](https://colab.research.google.com/drive/1i0L6nzryegeVfYneaiRTM_JblJ65DWhk?usp=sharing).

  Please refer to my [presentation](/demo/CMPT_713_Final_Report.pdf) and [technical report](/demo/Syntactic_Aware_Cross_Modality_Alignment_for_Vision_Language.pdf) for more details. There is also a presentation video for this project and you can find it [here](https://youtu.be/hAMiFiiKHzI).

**Topics:** _Visual Language Navigation (VLN)_, _Embodied AI_, _BERT_, _Tree Transformer_, _Natural Language Processing_, _Robotics_

**Skills:** _Pytorch_, _Python_, _Deep Neural Networks_, _Jupyter Lab_, _Colab_

## 2. Results

## 3. Acknowledgement

We acknowledge the use of codes from [SyntaxVLN](https://github.com/jialuli-luka/SyntaxVLN), [Recurrent VLN Bert](https://github.com/YicongHong/Recurrent-VLN-BERT), and [Tree Transformer](https://github.com/yaushian/Tree-Transformer).
