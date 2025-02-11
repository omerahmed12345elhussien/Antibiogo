# Training Code
This folder contains the training code. It has two subfolders: [stage1](stage1/) is for training models for cropping the petri dish. So the expected masks of background and foreground values only. For [stage2](stage2/) is for training models that crop the petri dish and identifying the antibiotic disks. The codes in [stage1](stage1/) and [stage2](stage2/) are the same, with the main difference how the data is served to the model. In [stage2](stage2/) we train with images of size $(512,512)$, and so we used [tfrecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord). Also, for the code of [stage2](stage2/) is now provided to be run in clusters, since several hyperparameters should be passed. So, we will explain how to train using the code in [stage1](stage1/).

## Structure
For codes in [stage1](stage1/) subfolder:
* `basemodel.py` contains the encoder-decoder architecture.
* `callback.py` contains callbacks that are needed to save model progress during training.
* `dataloader.py` contains the dataloader pipeline for serving the data during training, validation, and testing stages.
* `dataugmen.py` contains the data augmentation pipeline.
* `modelclass.py` contains a custom training loop.
* `prediction.py` contains a function for model inference.
* `train.py` contains the code to start training job.
* `utils.py` contains hyperparameters and functions that are needed to facilitate the training process.

For codes in [stage2](stage2/) subfolder, most of them are similar to stage1, except for the following:
* `modelclass.py` contains a custom training loop using [focal loss](https://arxiv.org/pdf/1708.02002).
* `tfrecord.py` contains the tfrecord class that coverts the dataset to Tfrecord format.

## Getting started
To get started with the training code of [stage1](stage1/), we provide a simple example using [MobileNet V2](https://arxiv.org/pdf/1801.04381) as encoder, and a decoder architecture $C512-C256-C128-C64$ (for further information regarding the decoder part, check [[1](https://arxiv.org/pdf/1611.07004)]).

#### Requirements:
- We assume training on GPUs.
- Please replicate the dataset in [Data_stage1](Data_stage1/) to at least $10$ images in each subfolder. 
- In case of using your own images, the images have to be of size $(256,256)$. Also, the masks should have two classes.

#### Steps to train the model:
- Install the required packages in `requirements.txt`.
- Change `root_path` in `utils.py`. Then, run `dataugmen.py` to augment the dataset.
- In `train.py` add your weights and biases [api-key](https://docs.wandb.ai/support/find_api_key/), then you can run a training job using `train.py`.


# References
[1] [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).](https://arxiv.org/pdf/1611.07004)