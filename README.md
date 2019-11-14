# CNN
---

## Part 1

The result of **print(score)** of the [improved model](https://github.com/genigarus/ImprovedCNN/blob/master/EIP_4_1st_DNN.ipynb) is [0.03502082550192863, 0.9901].

----
## Part 2

### 1) Convolution

In mathematics, *Convolution* is an operation on two functions f(x) and g(x) to give a third function h(x), where g(x) is flipped and translated over f(x). In each translation, each point of both functions are multiplied and then added to give the resulting function h(x), represented by:
h(x) = (f*g)(x)
A similar concept of cross-correlation, where g(x) is translated without flipping, is utilised in Neural Networks(NN). In NN terminology, this operation is called convolution and g(x) is called as *kernel* or *filter* when it is used for filtering purposes. Convolution is used to extract features or to filter images. The following example illustrates the use of convolution. Image can be considered to be a matrix of pixel values. So, f(x) and g(x) can be represented as matrix of values. Consider a [5,5] input image with 1 channel and a [3,3] matrix with same number of channels as input,i.e 1 channel.

5x5 Input Image      and          3x3 matrix
![Convolution_schematic – the data science blog](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196)

The above diagram shows convolution computation. The filter matrix is placed above the input image. For every overlapped position, each element of input matrix is multiplied with the corresponding overlapping element of filter matrix. Then all the multiplied values are added to give a single value which forms the element of the output matrix. The filter is then slid over the input matrix by one pixel(in this example) along one direction at a time. This is called *Stride*. It is the number of position by which the filter matrix is slid over the input matrix at a time. The 3 x 3 matrix is the *filter or kernel or feature detector* and the resultant matrix is called *convolved feature or activation map or feature map*. Although the convolved feature is of a smaller dimension, the spatial relation between pixels is maintained.

### 2) Filters/Kernels

Filters detect or extract features. This can be seen from the following example: Suppose there is a filter matrix with values -1 and 1 on two adjacent pixels and 0 everywhere else. Then, in the input image, when side by side pixels are similar, convolution results in 0. Whereas on the edges of image, adjacent pixels have very different values resulting in a large difference and a non-zero value. Thus, this matrix helps in detecting edges. In this way, by filling filter matrix with different values, different feature maps can be derived from the same image focusing on different aspects of the image.

In a CNN, the filters(or matrix of values) in the initial layers detect/extract edges in various forms. The subsequent layers build on these to create filters which detect texture, which in turn detect parts of object and in turn the object itself. These filter values are learnt by the network during backpropagation. 


### 3) Epochs

A forward propagation of each input through the network to generate an output followed by a backward propagation of the error between the generated output and the target value based on the cost function through the network, is called as training over an input and its corresponding target(together known as training example). When this training is performed on all the examples within the dataset, it is known as an epoch.


### 4) 1x1 Convolution

1x1 convolution involves the standard convolution procedure except that filter of size 1x1 is used. Since each input value is multiplied with the single 1x1 filter value, the resultant feature map will be of same width and height as input though the depth, ie number of channels, can be varied by using those many number of 1x1 filter.

Suppose an input of dimension [H,W,F] (where H-height, W-width, F-number of channels of the image) is convolved using F_1 1x1, dimension [1,1,F_1], with zero padding and 1 stride, then the resulting feature map will have  dimension [H,W,F_1]. Here F_1 can be greater, smaller or equal to F. Thus, 1x1 convolution is used to change the dimensionality of the image. When F_1 < F, 1x1 convolution reduces the number of features and hence the computational cost. When F_1 = F, it adds non-linearity to the neural network. When F_1 >F, it adds new parameters to the network. It is mainly used in conjunction with 3x3 or 5x5 convolution to decrease the computational cost. The following example shows this.

Consider a 200x200 image with 128 channels. If it undergoes a 3x3 convolution with 32x3x3 filters having 128 channels, the convolved feature will have 198x198 size with 32 channels. Its computational cost will be:
Volume of output x Volume of filter
= (198 x 198 x 32) x (3 x 3 x 128)
= (approx)1445 million
 If the same image undergoes a 1x1 convolution with 32 1x1 filters having 128 channels, the convolved feature will have 200x200 size with 32 channels. Its computational cost will be:
Volume of output x Volume of filter
= (200 x 200 x 32) x (1 x 1 x 128)
= (approx)163 million
3 x 3 convolution is approximately 8 times costlier than 1x1 convolution in computation. Thus, 1x1 convolution facilitates faster computation with less information loss.


### 5) 3x3 Convolution

3x3 convolution is an operation performed on the input matrix using a 3x3 filter/kernel. In this operation, a 3x3 filter matrix is placed above the input matrix. For every overlapped position, each element of input matrix is multiplied with the corresponding overlapping element of filter matrix. Then all the multiplied values are added to give a single value which forms the element of the output matrix. The filter is then slid over the input matrix by one pixel along one direction at a time. This operation performed with a 3x3 filter is known as **3x3 convolution**.


### 6) Feature Maps

During the convolution operation, the filter matrix is placed above the input matrix. For every overlapped position, each element of input matrix is multiplied with the corresponding overlapping element of filter matrix. Then all the multiplied values are added to give a single value which forms the element of the output matrix. The filter is then slid over the input matrix by one pixel along one direction at a time. This is called *Stride*. It is the number of position by which the filter matrix is slid over the input matrix at a time. The resultant matrix is called *convolved feature or activation map or feature map*. Each feature map highlights the feature which its corresponding kernel is trained/created to detect, while diminishing the other features. Thus, each feature map tends to be a picturesque view of the feature in various forms.


### 7) Activation Function

Activation function is a non-linear mathematical operation performed on the input to generate an output. This non-linearity of the activation function helps in mimicing the real world complex mapping between the input and output. A nueral network is composed of neurons. Each neuron takes an input and passes it through this function to give an output. The neuron gets activated or generates output at certain input values, thus the name activation function. Examples of activation functions include - sdigmoid, tanh, ReLU, LeakyReLU.

ReLU or Rectified Linear Unit is a function which return 0 when input is negative and the number itself when it is positive. Its a popular choice of activation function as it helps overcome vanishing gradient problem. 


### 8) Receptive Field

Receptive Field is the area which the filter is looking at in the input matrix. Local receptive field is the area in the immediate input matrix,i.e. the input to the current layer(to which the filter belongs) of the network, which the filter is looking at while global receptive field is the area in the initial input matrix to the network which the filter is looking at.
