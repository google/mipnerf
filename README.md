# mip-NeRF

This repository contains the code release for
[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/).
This implementation is written in [JAX](https://github.com/google/jax), and
is a fork of Google's [JaxNeRF implementation](https://github.com/google-research/google-research/tree/master/jaxnerf).
Contact [Jon Barron](https://jonbarron.info/) if you encounter any issues.

![rays](https://user-images.githubusercontent.com/3310961/118305131-6ce86700-b49c-11eb-99b8-adcf276e9fe9.jpg)

## Abstract

The rendering procedure used by neural radiance fields (NeRF) samples a scene
with a single ray per pixel and may therefore produce renderings that are
excessively blurred or aliased when training or testing images observe scene
content at different resolutions. The straightforward solution of supersampling
by rendering with multiple rays per pixel is impractical for NeRF, because
rendering each ray requires querying a multilayer perceptron hundreds of times.
Our solution, which we call "mip-NeRF" (à la "mipmap"), extends NeRF to
represent the scene at a continuously-valued scale. By efficiently rendering
anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable
aliasing artifacts and significantly improves NeRF's ability to represent
fine details, while also being 7% faster than NeRF and half the size. Compared
to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented
with NeRF and by 60% on a challenging multiscale variant of that dataset that
we present. mip-NeRF is also able to match the accuracy of a brute-force
supersampled NeRF on our multiscale dataset while being 22x faster.


## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
git clone https://github.com/google/mipnerf.git; cd mipnerf
# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name mipnerf python=3.6.13; conda activate mipnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r requirements.txt
```

[Optional] Install GPU and TPU support for Jax
```
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.65+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.

### Generate multiscale dataset
You can generate the multiscale dataset used in the paper by running the following command,
```
python scripts/convert_blender_data.py --blenderdir /nerf_synthetic --outdir /multiscale
```

## Running

Example scripts for training mip-NeRF on individual scenes from the three
datasets used in the paper can be found in `scripts/`. You'll need to change
the paths to point to wherever the datasets are located.
[Gin](https://github.com/google/gin-config) configuration files for our model
and some ablations can be found in `configs/`.
An example script for evaluating on the test set of each scene can be found
in `scripts/`, after which you can use `scripts/summarize.ipynb` to produce
error metrics across all scenes in the same format as was used in tables in the
paper.

### OOM errors
You may need to reduce the batch size to avoid out of memory errors. For example the model can be run on a NVIDIA 3080 (10Gb) using the following flag. 
```
--gin_param="Config.batch_size = 1024"
```

## Citation
If you use this software package, please cite our paper:

```
@misc{barron2021mipnerf,
      title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
      author={Jonathan T. Barron and Ben Mildenhall and Matthew Tancik and Peter Hedman and Ricardo Martin-Brualla and Pratul P. Srinivasan},
      year={2021},
      eprint={2103.13415},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Thanks to [Boyang Deng](https://boyangdeng.com/) for JaxNeRF.
