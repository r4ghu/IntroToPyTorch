# Intro To PyTorch

This is the implementation of basic MNIST classification using PyTorch.


## Requirements

- OS/VM of your choice
- Anaconda

## Environment 

Setup your Anaconda Environment using the following commands -
```
$ conda create -n oml python=3.6
$ conda activate oml
(oml) $ conda install pytorch torchvision -c pytorch
(oml) $ conda install tensorflow-gpu
(oml) $ conda install tensorboardX
```

## Usage

After successfully setting up environment, you can run the code by - 
```
tensorboard --logdir=./runs/ --host 0.0.0.0 --port 6007 & python main.py --phase train --continue_train 0
```
Now click here to visualize the results - [http://localhost:6007](http://localhost:6007)

## Tutorial
A detailed tutorial about the contents covered in this repo can found at - [LINK](https://sriraghu.com/2019/03/23/intro-to-pytorch/)

## Results

Here are the final TensorBoard Results - ![TensorBoardX](https://github.com/r4ghu/IntroToPyTorch/blob/master/content/IntroToPyTorch-TensorboardX.png)


## Author

Sri Raghu Malireddi / [@r4ghu](https://sriraghu.com)
