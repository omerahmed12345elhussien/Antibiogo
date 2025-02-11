# Antibiogo: ML approach for antibiotic susceptibility testing

This project is an attempt to replace the image processing module ([Improc](https://github.com/mpascucci/AST-image-processing/tree/master?tab=readme-ov-file)) in [[1](https://www.nature.com/articles/s41467-021-21187-3)] from classical image processing approaches to ML one. 

Content overview:
- [Basic concepts](#basic-concepts)
- [Current Improc](#current-improc)
- [Proposed approach](#proposed-approach)
  - [1: Semantic segmentation](#1-semantic-segmentation)
  - [2: Data collection](#2-data-collection-and-preparation)
  - [Installation](#installation)
- [References](#references)



## Basic concepts
There are three main concepts that are needed throughout the work which are: Petri dish (Plate), antibiotic disk, and the inhibition zone[[2](https://asm.org/getattachment/2594ce26-bd44-47f6-8287-0657aa9185ad/Kirby-Bauer-Disk-Diffusion-Susceptibility-Test-Protocol-pdf.pdf)].

<p align="center">
  <img src="assets/Petri_desc.jpg?raw=true" width="550" title="Petri_dish">
</p>

>
> * Petri dish: is a transparent lidded dish with an agar-based growth medium previously inoculated with bacteria. 
> 
> * Antibiotic disk: is a white round cellulose disk with a given concentration (6 mm diameter).
> 
> *  Inhibition zone: the black area where existing bacteria died, and current one cannot grow due to being susceptible.
>

## Current Improc
[Improc](https://github.com/mpascucci/AST-image-processing/tree/master?tab=readme-ov-file) diagram is shown below. It has three steps:

1- Crop the petri dish and locate the antibiotic disks.

2- Read the label at each antibiotic disk.

3- Identify and measure the inhibition zones.

![Improc](assets/Improc_old.png?raw=true)
*Figure taken from [[1](https://www.nature.com/articles/s41467-021-21187-3)].*

Currently, [Improc](https://github.com/mpascucci/AST-image-processing/tree/master?tab=readme-ov-file) uses [GrabCut](https://pub.ista.ac.at/~vnk/papers/grabcut_siggraph04.pdf) for cropping the petri dish, then uses the contrast of the Blue channel to locate the antibiotic disks, and finally uses intensity radial profile extraction and segmentation for measuring the inhibition zones (for further information check [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-21187-3/MediaObjects/41467_2021_21187_MOESM1_ESM.pdf)).

# Proposed approach
In this work, we formulate the problem as a semantic segmentation problem. We built our basline model suing U-Net architecture [[4](https://arxiv.org/pdf/1611.07004)] with MoblieNet V2 as encoder [[5](https://arxiv.org/pdf/1801.04381)]. Then, we experiment with other backbones such as MoblieNet V3 [[6](https://arxiv.org/pdf/1905.02244)] and EfficientNet-B0 [[7](https://arxiv.org/pdf/1905.11946)].

## 1: Semantic Segmentation
As a first stage, we built three models, each one of them is doing one of the steps that current Improc is doing. Then, we wish to have a single compact model.

![St1_new_Improc](assets/st1_ss.jpg?raw=true)

## 2: Data collection and preparation
A dataset of 3,363 AST images has been used to train our models.For full documentation regarding the used dataset, you can check [Data_Collection.md](Data_Collection.md).

For data preparation and analysis codes, you can check [data_engineering](data_engineering/) folder.


## Installation
The code requires `python>=3.10`. Clone the repository, then install the dependencies on a GPU machine using:

```bash
git clone https://github.com/omerahmed12345elhussien/Antibiogo.git && cd Antibiogo

python3.10 venv -m /your_desired_path_to_create_venv/.venv_name

. /your_desired_path_to_create_venv/.venv_name/bin/activate

pip install -r requirements.txt
```


# References
[1] [Pascucci, M., Royer, G., Adamek, J., Asmar, M. A., Aristizabal, D., Blanche, L., ... & Madoui, M. A. (2021). AI-based mobile application to fight antibiotic resistance. Nature communications, 12(1), 1173.](https://www.nature.com/articles/s41467-021-21187-3)

[2][Hudzicki, J. (2009). Kirby-Bauer disk diffusion susceptibility test protocol. American society for microbiology, 15(1), 1-23.](https://asm.org/getattachment/2594ce26-bd44-47f6-8287-0657aa9185ad/Kirby-Bauer-Disk-Diffusion-Susceptibility-Test-Protocol-pdf.pdf)

[3][Rother, C., Kolmogorov, V., & Blake, A. (2004). " GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG), 23(3), 309-314.](https://pub.ista.ac.at/~vnk/papers/grabcut_siggraph04.pdf)

[4] [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).](https://arxiv.org/pdf/1611.07004)

[5] [Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).](https://arxiv.org/pdf/1801.04381)

[6] [Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1314-1324).](https://arxiv.org/pdf/1905.02244)

[7] [Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.](https://arxiv.org/pdf/1905.11946)