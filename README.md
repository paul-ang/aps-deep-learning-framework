# APS: A deep learning framework for MR-to-CT image synthesis

This repository is the official implementation of _An Improved Deep Learning
Framework for MR-to-CT Image Synthesis with a New Hybrid Objective Function_.

## Getting started

- Download or clone this repo to your computer.
- Run `pip install -r requirements.txt` to install the required Python packages.
- The code was developed and tested on Python 3.6.13 and Ubuntu 16.04.

[//]: # (- Note that this codebase uses the [PyTorch Lightning framework]&#40;https://www.pytorchlightning.ai&#41;.)

## Training and testing the APS framework on your own  dataset

1. Create a custom Dataset class for your own data. Follow the official
   PyTorch [tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
   if you are unfamiliar with this step. Ensure that the data are normalized using min-max formula and resized to the expected image size (default is 288x288).
2. Integrate your custom Dataset class into `main_train.get_training_dataloaders()`.
3. Run `python main_train.py` to start training and testing the APS model. Run `python main_train.py -h` for the available hyperparameters.

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