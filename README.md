# deeplearning-model
This project is a collection of deep learning model, and I will update it in the future.

## Quickstart

To clone this project, use the following command line:
```
git clone https://github.com/shy218/deeplearning-model.git
```

## Models

### FCN8 (by pytorch)
FCN8 is a fully convolutional network which can classify each pixel in the image.
It uses pre-trained vgg16 as transfer learning model.

```
fcn8.FCN8(num_classes, weight_path=None)
```
num_classes -- the number of classes of each pixel.
weight_path -- the weight of vgg16 network. You can download from https://download.pytorch.org/models/vgg16-397923af.pth
(You must implement weight_path to run this network)


To run this model, first create a directory for weight file:
```
mkdir weights
cd weights
```
Then download the weight from https://download.pytorch.org/models/vgg16-397923af.pth

Run following code:
```
from fcn8 import FCN8
net = FCN8(10, 'weights/vgg16-397923af.pth')
```

To train the network, you need to extract the trainable parameters. To extract these parameters:
```
parameters = net.extract_parameters()
```

## License
`deeplearning-model` is a public work. Feel free to do whatever you want with it.
