**Convolution**
In mathematics, *Convolution* is an operation on two functions $f(x)$ and $g(x)$ to give a third function $h(x)$, where $g(x)$ is flipped and translated over $f(x)$. In each translation, each point of both functions are multiplied and then added to give the resulting function $h(x)$, represented by:
$$h(x) = (f*g)(x) = \int_{-\infty}^{\infty} {f(\tau)g(x-\tau)}d\tau$$
A similar concept of cross-correlation, where $g(x)$ is translated without flipping, is utilised in Neural Networks(NN). In NN terminology, this operation is called convolution and $g(x)$ is called as *kernel* or *filter* when it is used for filtering purposes. Convolution is used to extract features or to filter images. The following example illustrates the use of convolution. Image can be considered to be a matrix of pixel values. So, $f(x)$ and $g(x)$ can be represented as matrix of values. Consider a [5,5] input image with 1 channel and a [3,3] matrix with same number of channels as input,i.e 1 channel.
$\left[ {\begin{matrix} 1&1&1&0&0\\0&1&1&1&0\\0&0&1&1&1\\0&0&1&1&0\\0&1&1&0&0\end{matrix}} \right]$ $\left[ \begin{matrix} 1&0&1\\0&1&0\\1&0&1 \end{matrix} \right]$
$5\times 5$ Input Image      and          $3\times 3$ matrix
![Convolution_schematic – the data science blog](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196)
The above diagram shows convolution computation. The filter matrix is placed above the input image. For every overlapped position, each element of input matrix is multiplied with the corresponding overlapping element of filter matrix. Then all the multiplied values are added to give a single value which forms the element of the output matrix. The filter is then slid over the input matrix by one pixel(in this example) along one direction at a time. This is called *Stride*. It is the number of position by which the filter matrix is slid over the input matrix at a time. The $3 \times 3$ matrix is the *filter or kernel or feature detector* and the result matrix is called *convolved feature or activation map or feature map*. Although the convolved feature is of a smaller dimension, the spatial relation between pixels is maintained.

Filters detect or extract features. This can be seen from the following example: Suppose there is a filter matrix with values -1 and 1 on two adjacent pixels and 0 everywhere else. Then, in the input image, when side by side pixels are similar, convolution results in 0. Whereas on the edges of image, adjacent pixels have very different values resulting in a large difference and a non-zero value. Thus, this matrix helps in detecting edges. In this way, by filling filter matrix with different values, different feature maps can be derived from the same image focusing on different aspects of the image. Hence, more features can be derived if more filters are used.

Convolution reduces the size of input image. In some cases, where we want to apply filter to bordering parts of image and to avoid circular convolution, the input matrix is padded with zeros. Adding zero padding is called *Wide Convolution* while no padding addition is called *Narrow Convolution*.

For 2D, the dimension of the convolved feature matrix can be calculated as shown. Consider a 2D input image of spatial size $h \times w$ padded by P, with a filter size $F \times F$ and using stride S, the output matrix of convolved image is $H \times W$ for each channel, where
$$H = \frac{h-F+2P}{S} + 1$$
$$W = \frac{w-F+2P}{S} + 1$$


**$1 \times 1$ Convolution**
$1 \times 1$ convolution involves the standard convolution procedure except that filter of size $1 \times 1$ is used. This implies that the feature map will be of same width and height as input image though the depth, ie number of channels, can be varied by using those many number of $1 \times 1$ filter.

Suppose an image of dimension [H,W,F] (where H-height, W-width, F-number of channels of the image) is convolved using $F_1 \space 1 \times 1$, dimension [1,1,$F_1$], with zero padding and 1 stride, then the resulting feature map will have  dimension [H,W,$F_1$]. Here $F_1$ can be greater, smaller or equal to F. Thus, $1 \times 1$ convolution is used to change the dimensionality of the image. When $F_1$ < $F$, $1 \times 1$ convolution reduces the number of features and hence the computational cost. When $F_1$ = $F$, it adds non-linearity to the neural network. When $F_1$ >$F$, it adds new parameters to the network. It is mainly used in conjunction with $3 \times 3$ or $5 \times 5$ convolution to decrease the computational cost. The following example shows this.

Consider a $200 \times 200$ image with 128 channels. If it undergoes a $3 \times 3$ convolution with $32 \space 3 \times 3$ filters having 128 channels, the convolved feature will have $198 \times 198$ size with 32 channels. Its computational cost will be:
$$Volume \space of \space output \times Volume \space of \space filter$$
$$= (198 \times 198 \times 32) \times (3 \times 3 \times 128)$$
$$\approx 1445 \space million$$
 If the same image undergoes a $1 \times 1$ convolution with $32 \space1 \times 1$ filters having 128 channels, the convolved feature will have $200 \times 200$ size with 32 channels. Its computational cost will be:
$$Volume \space of \space output \times Volume \space of \space filter$$
$$= (200 \times 200 \times 32) \times (1 \times 1 \times 128)$$
$$\approx 163 \space million$$
$3 \times 3$ convolution is approximately 8 times costlier than $1 \times 1$ convolution in computation. Thus, $1 \times 1$ convolution facilitates faster computation with less information loss.


**10 Examples of use of MathJax in Markdown:**

In addition to some examples found above, below are some other uses:
1. **_Theory of Relativity:_** $E = mc^2$
2. $$\boxed {S_{BH} = {\frac{kc^3}{4{\hbar}G}} A}$$
3. **_For a quadratic equation:_** $f(x) = ax^2+bx+c$
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
4. $$\int {\cosec {x}} = \log {\vert \cosec x - \cot x \vert} + c$$
5. $$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$
6. *Given* $B = [2,3] \times [1,2] \times [0,1]$
$$ \iiint_B {8xyz}\, {dV} = \int_1^2\int_2^3\int_0^1 {8xyz} \,dz\,dx\,dy$$
$$= \int_1^2\int_2^3 {4xyz^2} \,dz\,dx\,dy$$
$$= \int_1^2\int_2^3 {4xyz^2} \vert_0^1 \,dx\,dy$$
$$= \int_1^2\int_2^3 {4xy} \,dx\,dy$$
$$= \int_1^2 {2x^2y} \vert_2^3 \,dy$$
$$= \int_1^2 {10y} \,dy = 15$$
7. A matrix can be shown as : $\left[ {\begin{matrix} 1&1&1&0&0\\0&1&1&1&0\\0&0&1&1&1\\0&0&1&1&0\\0&1&1&0&0\end{matrix}} \right]$
8. $$ \frac{d(x^2)}{dx} = 2x $$
9. $P(S \bigcup \mathcal E) = 1$, where $S$ is the sample set and $\mathcal E$ is the universal set
10. **K-map representation of a circuit:** $f(A,B,C) = \prod (0,1,3,4)$