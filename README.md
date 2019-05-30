# DiracDeltaNet

PyTorch implementation of DiracDeltaNet from paper [Synetgy: Algorithm-hardware Co-design for ConvNet Accelerators on Embedded FPGAs](https://arxiv.org/abs/1811.08634) by `Yifan Yang`. This uses the [ShiftResNet](https://github.com/alvinwan/shiftresnet-cifar) codebase written by [Alvin Wan](http://alvinwan.com) and [label-refinery](https://github.com/hessamb/label-refinery) by [Hessam Bagherinezhad](http://homes.cs.washington.edu/~hessam/).

DiracDeltaNet is an efficient convolution neural network tailored for embedded FPGAs on ImageNet classification task. Its macro-architecture originates from [ShuffleNet V2](https://arxiv.org/abs/1807.11164). DiracDeltaNet is codesigned with its embedded FPGA accelerator. It has the following features:

- The operator set in DiracDeltaNet is shrunk to 1x1 convolution, 2x2 max pooling, shift, channel shuffle and concatenation  for hardware efficiency
- All of the 3x3 convolutions in ShuffleNet V2 are replaced with [shift operations](https://arxiv.org/abs/1612.01051) and 1x1 convolutions
- Several 2x2 max-pooling layers are added and the kernel size of the existing 3x3 max-pooling are reduced to 2x2
- Transpose based channel shuffle is changed into shift-based channel shuffle
- It can be aggressively quantized into 4-bit weights and 4-bit activations with less than 1\% top-5 accuracy loss

In this repository, we offer:

- Our ShuffleNet V2 implementation
- Source code of DiracDeltaNet
- Pre-trained ShuffleNetv2 and DiracDeltaNet
- Training and testing code

## [Synetgy: Algorithm-hardware Co-design for ConvNet Accelerators on Embedded FPGAs](https://arxiv.org/abs/1811.08634)

By Yifan Yang, Qijing Huang, Bichen Wu, Tianjun Zhang, Liang Ma, Giulio Gambardella, Michaela Blott, Luciano Lavagno, Kees Vissers, John Wawrzynek and Kurt Keutzer

The ideas behind the design of DiracDeltaNet, details about its embedded FPGA accelerator and more experimental results can be found in the paper ([link](https://arxiv.org/abs/1811.08634)).

If you find this work useful for your research, please consider citing:

    @article{synetgy,
    author    = {Yifan Yang and
                Qijing Huang and
                Bichen Wu and
                Tianjun Zhang and
                Liang Ma and
                Giulio Gambardella and
                Michaela Blott and
                Luciano Lavagno and
                Kees A. Vissers and
                John Wawrzynek and
                Kurt Keutzer},
    title     = {Synetgy: Algorithm-hardware Co-design for ConvNet Accelerators on
                Embedded FPGAs},
    journal   = {CoRR},
    volume    = {abs/1811.08634},
    year      = {2018},
    url       = {http://arxiv.org/abs/1811.08634},
    archivePrefix = {arXiv},
    eprint    = {1811.08634},
    }

## Download

The training of DiracDeltaNet adopts a pre-trained ResNet50 ([download](https://www.dropbox.com/sh/84yp02gpibk5sgi/AAAPmNXBUbi-Ah1heyTEOyoKa?dl=0)) as [label-refinery](https://arxiv.org/abs/1805.02641).

We offer the following pre-trained model:

- Our implementation of ShuffleNet V2 1x with 90 epoch of training
- Full precision DiracDeltaNet with 90 epoch of training
- Quantized DiracDeltaNet (4-bit weights, 4-bit activations)

The pre-trained models can be found on [Dropbox](https://www.dropbox.com/sh/84yp02gpibk5sgi/AAAPmNXBUbi-Ah1heyTEOyoKa?dl=0).

Please put the ResNet50 model and pre-trained model in the following file structure:

```
DiracDeltaNet/
   |
   |-- test.py
   |-- resnet50.t7
   |-- checkpoint/
       |-- ShuffleNetv2.t7
       |-- DiracDeltaNet_full.t7
       |-- ...
```

## Usage

The source code requires PyTorch 0.4.0 (there is known incompatible issue when using PyTorch 0.4.1, haven't tested on PyTorch 1.0). Python 3.5+ is needed (there is known incompatible issue when using Python 2.7).

The full list of arguments can be accessed using `--help`

### Inference

For example, to run inference of our ShuffleNet V2 1x implementation, simply type:

```bash
python test.py --datadir=PATH-TO-IMAGENET-FOLDER --inputdir=./checkpoint/ShuffleNetv2.t7
```

### Training

For example, to train full precision DiracDeltaNet from scratch, simply type:

```bash
python train.py --datadir=PATH-TO-IMAGENET-FOLDER --outputdir=./checkpoint/DiracDeltaNet_full.t7
```

The default values of arguments are the hyperparameter we used.

### Fine Tuning

For example, to fine tune 8-bit weights and 8-bit activations (except for the first and last conv) DiracDeltaNet from full precision pre-trained DiracDeltaNet, simply type:

```bash
python train.py --datadir=PATH-TO-IMAGENET-FOLDER --inputdir=./checkpoint/DiracDeltaNet_full.t7 --outputdir=./checkpoint/DiracDeltaNet_w8a8.t7 --lr_policy=step --weight_bit=8 --act_bit=8
```

You can set smaller lr as well as # of epochs.

## Experimental Results

| Model             | Weight Bitwidth | Activation Bitwidth | Top-1 Acc | Top-5 Acc | Note               |
|-------------------|-----------------|---------------------|-----------|-----------|--------------------|
| ShuffleNet V2 1x  | 32              | 32                  | 69.4%     | N/A       | [original paper](https://arxiv.org/abs/1807.11164) |
| ShuffleNet V2 1x  | 32              | 32                  | 67.9%     | 88.0%     | our implementation with 90 epoch training  |
| DiracDeltaNet     | 32              | 32                  | 68.9%     | 88.7%     | 90 epoch training  |
| DiracDeltaNet     | 4               | 4                   | 68.3%     | 88.1%     |                    |

More can be found in the paper.