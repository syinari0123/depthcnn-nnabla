# DepthCNN-nnabla
This is a [Neural Network Libraries](https://github.com/sony/nnabla)(nnabla) implementation of DepthCNN (unofficial, for practice).

This DepthCNN architecture is typical U-Net referring to [SharpNet](https://github.com/MichaelRamamonjisoa/SharpNet).
I used [imagenet pretrained ResNet50 model](https://nnabla.readthedocs.io/en/latest/python/api/models/imagenet.html) for the encoder part.

## ToDo
- [ ] Prepare c++ inference code
- [ ] Hyperparameter search

## Environment
- Ubuntu18.04 (GPU: NVIDIA GeForce GTX 1080) 
- CUDA10.0
- cuDNN7.6
- Python3.6 (for training)
- nnabla-ext-cuda100 (for training)

This package requires CUDA10.0 & cuDNN7.6 according to [this information](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).
```
pip install nnabla-ext-cuda100
```

## Dataset
Download the preprocessed [NYU Depth v2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) in HDF5 formats (from [Sparse-to-Dense](https://github.com/fangchangma/sparse-to-dense.pytorch) by Fangchang et al.). The NYU dataset requires 32G of storage space.
```
mkdir data && cd data
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
```

## How to run
### Training
You can train DepthCNN with following command.
```
python train.py \
    --dataset-path "/path/to/nyudepthv2" \
    --batch-size 8 \
    --img-height 228 \
    --img-width 304 \
    --optimizer 'sgd' \
    --learning-rate 1e-3 \
    --epochs 30 \
    --log-dir "log"
```

### Inference (python)
After training, you can inference your model!
```
python inference.py \
    --dataset-path "/path/to/nyudepthv2" \
    --pretrained-path "/path/to/checkpoint.h5" \
    --img-height 228 \
    --img-width 304 \
    --result-dir "result"
```