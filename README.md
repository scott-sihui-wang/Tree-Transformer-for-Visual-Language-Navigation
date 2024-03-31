# Tree Transformer for Visual Language Navigation

## 1. Introduction

### 1.1 Background and Motivation

The aim of this project is to incorporate syntactic information into VLN (`Visual Language Navigation`) networks to improve cross modality alignment and decision making quality.

Here is the general workflow of `Visual Language Navigation`:

![VLN Workflow](/demo/VLNWorkflow.png)

`Tree transformer` proposed to impose `hierarchical constraints` and `constituent attention` in transformers, so that low layers only have short-ranges attention, which gradually merge into long-ranged attention in high layers. This is to mimic the tree structures that we usually obtain when we parse a sentence. It is believed that tree transformer helps transformer better understand the grammar structures of the text.

In this project, we added `hierarchical constraints` and `constituent attention` to `VLN Bert` to see if `tree transformer` helps improve understanding of syntactic information and leads to better performance.

Here is the architecture of `VLN Bert` network:

![VLNBert](/demo/VLNBert.png)

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

Below shows the `success rates` of navigation for `VLN Bert` (the ones start with `Base`) and our method (the ones start with `Syn`) on `training`, `validation-seen`, and `validation-unseen` datasets.

![Success Rate Comparison](/demo/SuccessRate.png)

Below shows the heatmap of `constituent priors` for an example instruction: _Turn around and enter the house. Head past the blue chairs. When you are behind the red chair on the left, turn and enter the bathroom to the left. Stop inside the bathroom right in front of the sink facing the sink and mirror._

![Heatmap of Self Attention](/demo/SelfAttention.png)

Our conclusions are:

- The `Transformer-Syntactic model` (our method) tends to outperform the `Transformer-Baseline model` (`VLN Bert`) in success rate, however the `Transformer-Baseline model` tends to generate more concise paths. Overall, the difference between the two models is quite small.

- Although the `tree transformer` did show short-ranged attention at lower layers and long-ranged attention at higher layers, it didn’t seem to learn tree structures of syntactic constituents for better cross modality alignment in `VLN` scenarios.

## 3. Acknowledgement

We acknowledge the use of codes from [SyntaxVLN](https://github.com/jialuli-luka/SyntaxVLN), [Recurrent VLN Bert](https://github.com/YicongHong/Recurrent-VLN-BERT), and [Tree Transformer](https://github.com/yaushian/Tree-Transformer).
