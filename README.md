
# Monocular Neural Image-based Rendering with Continuous View Control
This is the code base for our paper [**Monocular Neural Image-based Rendering with Continuous View Control**](https://arxiv.org/abs/1901.01880). We propose an approach to generate novel views of objects from only one view, with fine-grained control over the virtual viewpoints.

## Prerequisites
- Ubuntu 16.04
- Python 3
- NVIDIA GPU + CUDA CuDNN
- Pytorch 0.4.0

### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom), [dominate](https://github.com/Knio/dominate) and [imageio](https://pypi.org/project/imageio/)
```bash
pip install visdom
pip install dominate
pip install imageio
pip install pandas
```
- Clone this repo:
```bash
git clone git@github.com:cx921003/ContViewSynthesis.git
cd ContViewSynthesis
```
## Demo

- Download a pre-trained model from our [Google Drive](https://goo.gl/P7jA4a);
- Unzip the model under ``./checkpoints/`` folder;
- Run ``./demo_car.sh`` or ``./demo_kitti.sh`` to run the demo.

### Training
- Download a dataset from our [Google Drive](https://goo.gl/4bj6GD);
- Unzip the dataset under ``./datasets/`` folder;
- Train a model by running ``./training_car.sh``, ``./training_chair.sh`` or ``./training_kitti.sh``
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/$name/`

### Testing:
- Configure the following arguments in ``./test.sh``:
    - ``dataroot``: the path to the test images
    - ``name``: the name of the model, make sure the model exists under ``./checkpoint/``
    - ``test_views``: number of views to generate per input image
- Test the model: ``./testing.sh``

The test results will be saved to `.gif` files and a html file here: `./results/car/latest_test/`.

## Citation
If you find this repository useful for your research, please consider citing our paper.
```
@article{chen2019mono,
  title={Monocular Neural Image Based Rendering with Continuous View Control},
  author={Chen, Xu and Song, Jie and Hilliges, Otmar},
  year= {2019},
  booktitle = {International Conference on Computer Vision (ICCV)},
}
```

## Acknowledgments
Code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git) written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89) and [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) written by [Clément Pinard](https://github.com/ClementPinard).
