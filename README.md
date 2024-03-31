# Tree Transformer for Visual Language Navigation

## 1. Introduction

### 1.1 Background and Motivation

The aim of this project is to incorporate syntactic information into VLN (`Visual Language Navigation`) networks to improve cross modality alignment and decision making quality.

`Tree transformer` proposed to impose `hierarchical constraints` and `constituent attention` in transformers, so that low layers only have short-ranges attention, which gradually merge into long-ranged attention in high layers. This is to mimic the tree structures that we usually obtain when we parse a sentence. It is believed that tree transformer helps transformer better understand the grammar structures of the text.

In this project, we added `hierarchical constraints` and `constituent attention` to `VLN Bert` to see if `tree transformer` helps improve understanding of syntactic information and leads to better performance.

### 1.2 Project Overview

The project is hosted on `Colab`. Please follow the [link](https://drive.google.com/drive/folders/11PMBFEDVkjrm4O2td1NitIy0Z41TFcSh?usp=sharing) to check out the full details of this project.

To replicate the experimental results, please check out the `Colab` pages: 

**Topics:** _Visual Language Navigation (VLN)_, _Embodied AI_, _BERT_, _Tree Transformer_, _Natural Language Processing_, _Robotics_

**Skills:** _Pytorch_, _Python_, _Deep Neural Networks_, _Jupyter Lab_, _Colab_

## 2. Results

## 3. Acknowledgement

We acknowledge the use of codes from [SyntaxVLN](https://github.com/jialuli-luka/SyntaxVLN), [Recurrent VLN Bert](https://github.com/YicongHong/Recurrent-VLN-BERT), and [Tree Transformer](https://github.com/yaushian/Tree-Transformer).
