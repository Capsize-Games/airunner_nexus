# RunAI

A generic StableDiffusion server

## Installation

Create a conda environment, then install torch

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

Install xformers

`conda install xformers -c xformers/label/dev`

Install other requirements via pip

`pip install -r requirements.txt`

Install diffusers from github repo

```
cd diffusers
pip install -e .
```

