# APS: A deep learning framework for MR-to-CT image synthesis

This repository is the official implementation of _An Improved Deep Learning
Framework for MR-to-CT Image Synthesis with a New Hybrid Objective Function_.

## Getting started

- Download or clone this repo to your computer.
- Run `pip install -r requirements.txt` to install the required Python packages.
- The code was developed and tested on _Python 3.6.13_ and _Ubuntu 16.04_.
- Note that this codebase runs on [PyTorch Lightning library](https://www.pytorchlightning.ai).

## Training and testing the APS framework on your own  dataset

### Steps
1. Create a custom Dataset class for your own data. Follow the official
   PyTorch [tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
   if you are unfamiliar with this step.  
2. Integrate your custom Dataset class into `main_train.get_training_dataloaders()`. 
3. Run `python main_train.py --gpus 1` to start training and testing the APS model on 1 GPU. Execute `python main_train.py --help` for the available input arguments.

### Notes

- Ensure that the data are normalized using min-max formula and resized to the
  expected input image size (default is 288x288). The expected input image size 
  is set via `--image_size` input argument of `main_train.py`. 
- The `__getitem__()` of your custom Dataset class has to return a _dict_ with the following keys: 
  - _ct_: the CT image tensor.
  - _in_phase_: the MR in-phase image tensor.
  - _ct_min_: the smallest CT's HU value before the min-max normalization. This is needed for
    metric computations.
  - _ct_max_. the biggest CT's HU value before the min-max normalization. This is needed for
    metric computations.


## Visual results

![visual-results](figures/visual-results.gif)

## Citation

If you use this code for your research, please cite our paper.

```
@inproceedings{ang2022,
  title={An Improved Deep Learning Framework for MR-to-CT Image Synthesis with a New Hybrid Objective Function},
  author={Ang, Sui Paul and Phung, Son Lam and Field, Matthew and Schira, Mark Matthias},
  booktitle={Proceedings of the IEEE International Symposium on Biomedical Imaging},
  year={2022}
}
```

## Acknowledgement

Some parts of our code are inspired by [F-LSeSim](https://github.com/lyndonzheng/F-LSeSim)
and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).