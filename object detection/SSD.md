# SSD：Single Shot MultiBox Detector

## 1. 模型

## 1.1 采用多尺度 feature maps 进行检测

所谓多尺度，就是采用大小不同的 feature maps 来进行检测，这样做的好处是：可以采用较大的 feature maps 来检测小的 object， 而小的 feature maps（即 high-level feature maps）具有较强的语义特征，而位置信息损失比较严重，可以用于检测较大的 object。



在 SSD 中， 采用 VGG16 作为 backbone（当然你也可以采用其他经典的网络），使用前面的 `5` 层，然后利用 atrous convolution（即空洞卷积），将 `fc6` 和 `fc7` 这两个 fc layers 转换为 conv layers，然后再额外地增加 `3` 个 conv layers 和 `1` 个 GAP layer（全局平均池化层）。



这样，不同出层次的 feature maps 分别用于预测 default box 所属不同 class 的 score 以及该box的  `offset` ，最后通过 NMS 得到最终的检测结果。

![](../images/SSD%20Model.png)



从上图可以看到，SSD 的 `conv4_3` 作为用于检测的第一个 feature map，接着后面增加的 `conv7` 、`conv8_2`、`conv9_2`、`conv10_2`、`conv11_2` 将用于共同作为用于检测的 feature maps，它们的大小分别为：`(38, 38)`、`(19, 19)`、`(10, 10)`、`(5, 5)`、`(3, 3)`、`(1, 1)`。



## 1.2 采用全卷积的形式进行预测

其实 SSD 的网络结构类似于 FCN，即全部采用 conv layers 的形式。其实，在现在 CNN 的设计潮流中，fc layer 被广泛诟病，主要输造成参数量巨大，导致计算量增加。现在主流的方法是在最后一层加上一个 GAP layer，当然这个点子来自 NIN。原来利用 fc layer 来进行分类，但这却缺乏数学思想在里面，感觉就是纯靠试出来的感觉一样。而利用 GAP 进行 pooling 之后，feature map 的每一维都有特定的含义，利用它们再来进行分类，便可以在一定程度上打开神经网络这个黑箱。



## 1.3 设置先验框

说白了，就是借鉴了 Faster R-CNN 的 anchor 机制。不同 feature map 设置的先验框的数目不同，即：

-  `conv4_3` ：`4`
- ` conv7`、 `conv8_2`、 `conv9_2` ：`6`
-  `conv10_2`、 `conv11_2` ：`4`



默认情况下，每个 feature map 会有一个 $a_{r}=1$ ，且尺度为 $s_{k}$ 的先验框。除此之外，还会这是一个尺度为 $s_{k}^{\prime}=\sqrt{S_{k} S_{k+1}}$ 且 $a_{r}=1$ 的先验框。这样，每个 feature map 都设置了两个长宽比（aspect radio）为 `1` 但大小不一样的正方形先验框。先验框的长宽比为：
$$
a_{r}=\left\{1,2,3, \frac{1}{2}, \frac{1}{3}\right\}
$$


在设置先验框的尺度时，也是有技巧的。在此论文中，作者采用了一个线性规则，即随着 feature map 的尺寸减小，先验框的尺度线性增加。
$$
s_{k}=s_{\min }+\frac{s_{\max }-s_{\min }}{m-1}(k-1), \quad k \in[1, m]
$$
 其中，$S_{k}$ 表示先验框相对于 input image 的比例；$s_{max}$、$s_{min}$ 分别代表比例的最大值和最小值。



对于第一个 feature map，

