# DiffusionABSA
Official implementation of DiffusionABSA: Let’s Rectify Step by Step: Improving Aspect-based Sentiment Analysis with Diffusion Models


## Abstract
Aspect-Based Sentiment Analysis (ABSA) stands as a crucial task in predicting the sentiment polarity associated with identified aspects within text. However, a notable challenge in ABSA lies in precisely determining the aspects’ boundaries (start and end indices), especially for long ones, due to users’ colloquial expressions. We propose DiffusionABSA, a novel diffusion model tailored for ABSA, which extracts the aspects progressively step by step. Particularly, DiffusionABSA gradually adds noise to the aspect terms in the training process, subsequently learning a denoising process that progressively restores these terms in a reverse manner. To estimate the boundaries, we design a denoising neural network enhanced by a syntax-aware temporal attention mechanism to chronologically capture the interplay between aspects and surrounding text. Empirical evaluations conducted on eight benchmark datasets underscore the compelling advantages offered by DiffusionABSA when compared against robust baseline models.


## Preparing the Environment
```
conda create --name absa python=3.8
conda activate absa
pip install -r requirements.txt
```
## Datasets
The preprocessed datasets are available at: [ACE2004](https://drive.google.com/drive/folders/10DYZGYqYSRFQZUbGs8OhFtAvaVD1FC0D?usp=sharing), [GENIA](https://drive.google.com/drive/folders/1krNw98zi5mp0KPZGoCo5D5ne8dWV6pUD?usp=sharing), [CoNLL03](https://drive.google.com/drive/folders/17BXWQ2W0zzrbYR8W1KAWSCNSYJcoUGiw?usp=sharing), [MSRA](https://drive.google.com/drive/folders/1wt0XTEG3FFl8uiUyTUYxVwQ1i3oZtOHn?usp=sharing). Please download them and put them into the data/datasets folder. And we obtained syntax information like part-of-speech tags and dependency trees on these datasets by using [StandfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/). If you require more detailed data, please contact me via email (iblislsy@gmail.com). Prior to doing so, please ensure that you have obtained the necessary license.


## Training
We have prepared the default training parameters in `configs`. Try this demo:
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python diffusionabsa.py train --config configs/penga_14lap.conf
```
In general, training with only one RTX 3090 GPU achieves comparable performance with the results reported in the paper.

## Acknowledgement
Thanks to the work ([DiffusionNER](https://github.com/tricktreat/DiffusionNER)) of relevant researcher for inspiring me.
