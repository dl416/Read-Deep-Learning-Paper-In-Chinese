# SSD：Single Shot MultiBox Detector

## 1. 模型

### 1.1 采用多尺度 feature maps 进行检测

所谓多尺度，就是采用大小不同的 feature maps 来进行检测，这样做的好处是：可以采用较大的 feature maps 来检测小的 object， 而小的 feature maps（即 high-level feature maps）具有较强的语义特征，而位置信息损失比较严重，可以用于检测较大的 object。



在 SSD 中， 采用 VGG16 作为 backbone（当然你也可以采用其他经典的网络），使用前面的 `5` 层，然后利用 atrous convolution（即空洞卷积），将 `fc6` 和 `fc7` 这两个 fc layers 转换为 conv layers，然后再额外地增加 `3` 个 conv layers 和 `1` 个 GAP layer（全局平均池化层）。



这样，不同出层次的 feature maps 分别用于预测 default box 所属不同 class 的 score 以及该box的  `offset` ，最后通过 NMS 得到最终的检测结果。

![](../images/SSD%20Model.png)



从上图可以看到，SSD 的 `conv4_3` 作为用于检测的第一个 feature map，接着后面增加的 `conv7` 、`conv8_2`、`conv9_2`、`conv10_2`、`conv11_2` 将用于共同作为用于检测的 feature maps，它们的大小分别为：`(38, 38)`、`(19, 19)`、`(10, 10)`、`(5, 5)`、`(3, 3)`、`(1, 1)`。



### 1.2 采用全卷积的形式进行预测

其实 SSD 的网络结构类似于 FCN，即全部采用 conv layers 的形式。其实，在现在 CNN 的设计潮流中，fc layer 被广泛诟病，主要输造成参数量巨大，导致计算量增加。现在主流的方法是在最后一层加上一个 GAP layer，当然这个点子来自 NIN。原来利用 fc layer 来进行分类，但这却缺乏数学思想在里面，感觉就是纯靠试出来的感觉一样。而利用 GAP 进行 pooling 之后，feature map 的每一维都有特定的含义，利用它们再来进行分类，便可以在一定程度上打开神经网络这个黑箱。



### 1.3 设置默认框

说白了，就是借鉴了 Faster R-CNN 的 anchor 机制。不同 feature map 设置的默认框的数目不同，即：

-  `conv4_3` ：`4`（不使用长宽比为 `3`、`1/3` 的默认框）
- ` conv7`、 `conv8_2`、 `conv9_2` ：`6`
-  `conv10_2`、 `conv11_2` ：`4`（不使用长宽比为 `3`、`1/3` 的默认框）



默认情况下，每个 feature map 会有一个 $a_{r}=1$ ，且尺度为 $s_{k}$ 的默认框。除此之外，还会这是一个尺度为 $s_{k}^{\prime}=\sqrt{S_{k} S_{k+1}}$ 且 $a_{r}=1$ 的默认框。这样，每个 feature map 都设置了两个长宽比（aspect radio）为 `1` 但大小不一样的正方形默认框。默认框的长宽比为：
$$
a_{r} \in \left\{ 1,2,3, \frac{1}{2}, \frac{1}{3}
\right\} 
$$


在设置默认框的尺度时，也是有技巧的。在此论文中，作者采用了一个线性规则，即随着 feature map 的尺寸减小，默认框的尺度线性增加。
$$
s_{k}=s_{\min }+\frac{s_{\max }-s_{\min }}{m-1}(k-1), \quad k \in[1, m]
$$
 其中，$S_{k}$ 表示默认框相对于 input image 的比例；$s_{max}$、$s_{min}$ 分别代表比例的最大值和最小值（paper 里面取 `0.2` 和 `0.9`）。



对于第一个 feature map（即 `conv4_3`），其默认框的尺度比例一般设置为 $s_{min}/2=0.1$。那么尺度为 $300\times0.1=30$。对于后面的 feature map，默认框尺度按照公式线性增加，但是先将尺度比例先扩大 `100` 倍，此时增长步长为：
$$
\left\lfloor\frac{\left\lfloor s_{\max } \times 100\right\rfloor-\left\lfloor s_{\min } \times 100\right\rfloor}{m-1}\right\rfloor= 17
$$


这样，各个 feature map的 $s_{k}$ 为 `20`、`37`、`54`、`71`、`88`。将这些比例除以 `100`，然后再乘以图片大小，可以得到各个 feature map 的尺度为：`30`、`60`、`111`、`162`、`213`、`264`（对于 TF 在实现时，最小比例  $s_{min}$ 采用 `0.15` 而非 `0.2`）。【注意】：最后一个 feature map 需要参考一个虚拟的 $s_{m+1}=300 \times105/100=315$ 来计算 $s'_{m}$。



每个单元的默认框的中心点分布在各个单元的中心，即 $\frac{i+0.5}{|f_k|},\frac{j+0.5}{|f_k|}),i,j\in[0, |f_k|)$，其中 $|f_k|$ 为特征图的大小。



得到了特征图之后，需要对特征图进行卷积得到检测结果。下图为 `3×3` 大小的 feature map 的检测过程：

![](../images/Feature%20Map.jpg)



其中，`Priorbox` 是得到默认框。检测值包含两个部分：类别置信度和边界框位置，各采用一次 $3\times3$ 卷积来进行完成。令 $n_{k}$ 为该 feature map 所采用的默认框数目，那么类别置信度需要的卷积核数量为 $n_{k}\times c$ ，而边界框位置需要的卷积核数量为 $n_{k}\times 4$。由于每个默认框都会预测一个边界框，所以 SSD300 一共可以预测 $38\times38\times4+19\times19\times6+10\times10\times6+5\times5\times6+3\times3\times4+1\times1\times4=8732$ 个边界框，这是一个相当庞大的数字，所以说 SSD 本质上是密集采样。



## 2. 训练过程

### 2.1 默认框匹配 
在训练过程中，首先要确定训练图片中的 ground truth 与哪个默认框来进行匹配，与之匹配的默认框所对应的边界框将负责预测它。



在 YOLO 中，ground truth 的中心落在哪个单元格，该单元格中与其 IoU 最大的边界框负责预测它；而 SSD 的默认框与 ground truth 的匹配原则主要有两点:

* 第一个原则：首先，对于图片中每个 ground truth，找到与其 IoU 最大的默认框，该默认框与其匹配，这样，可以保证每个 ground truth 一定与某个默认框匹配。通常称与 ground truth 匹配的默认框为正样本；反之，若一个默认框没有与任何 ground truth 进行匹配，那么该默认框只能与背景匹配，就是负样本。一个图片中 ground truth 是非常少的， 而默认框却很多，如果仅按第一个原则匹配，很多默认框会是负样本，正负样本极其不平衡，所以需要第二个原则。
* 第二个原则：对于剩余的未匹配默认框，若某个 ground truth 的 IoU 大于某个阈值（一般是 `0.5`），那么该默认框也与这个 ground truth 进行匹配。这意味着某个 ground truth 可能与多个默认框匹配，这是可以的。但是反过来却不可以，因为一个默认框只能匹配一个 ground truth，如果多个 ground truth 与某个默认框 IoU 大于阈值，那么默认框只与 IoU 最大的那个默认框进行匹配。



尽管一个 ground truth 可以与多个默认框匹配，但是 ground truth 相对默认框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD 采用了 hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的 top-k 作为训练的负样本，以保证正负样本比例接近 `1:3`。



### 2.2 损失函数

训练样本确定了，然后就是损失函数了。损失函数定义为置信度误差（confidence loss，`conf`）与位置误差（localization loss，`loc`）的加权和： 
$$
L(x, c, l, g)=\frac{1}{N}\left(L_{c o n f}(x, c)+\alpha L_{l o c}(x, l, g)\right)
$$
其中：

* $N$ 是默认框的正样本数量。



#### 2.2.1 locatization loss

对于 locatization loss，其采用 Smooth L1 loss，定义如下：
$$
L_{l o c}(x, l, g)=\sum_{i \in P o s}^{N} \sum_{m \in\{c x, c y, w, h\}} x_{i j}^{k} {smooth}_{\mathrm{L1}}\left(l_{i}^{m}-\hat{g}_{j}^{m}\right)\\
{\hat{g}_{j}^{c x}=\left(g_{j}^{c x}-d_{i}^{c x}\right) / d_{i}^{w}} \\
{\hat{g}_{j}^{c y}=\left(g_{j}^{c y}-d_{i}^{c y}\right) / d_{i}^{h}} \\ 
{\hat{g}_{j}^{w}=\log \left(\frac{g_{j}^{w}}{d_{i}^{w}}\right) \quad \\
\hat{g}_{j}^{h}=\log \left(\frac{g_{j}^{h}}{d_{i}^{h}}\right)}
$$


其中， Smooth L1 loss 的定义为：
$$
{smooth}_{L_{1}}(x)=\left\{\begin{array}{ll}{0.5 x^{2}} & {\text { if }|x|<1} \\ {|x|-0.5} & {\text { otherwise }}\end{array}\right.
$$


其中：

* $x^p_{ij}\in \{ 1,0 \}$ 为一个指示参数，当  $x^p_{ij}= 1$ 时表示第 $i$ 个默认框与第 $j$ 个 ground truth 匹配，并且 ground truth 的类别为 $p$。
* $c$为类别置信度预测值。
* $l$ 为默认框的所对应边界框的位置预测值
* $g$ 是 ground truth 的位置参数。



由于 $x^p_{ij}$ 的存在，所以位置误差仅针对正样本进行计算。值得注意的是，要先对 ground truth 的 $g$ 进行编码得到 $\hat{g}$，因为预测值 $l$ 也是编码值，若设置 `variance_encoded_in_target=True`，编码时要加上 `variance`：


$$
\begin{array}{l}{\hat{g}_{j}^{c x}=\left(g_{j}^{c x}-d_{i}^{c x}\right) / d_{i}^{w} / \text {variance}[0]\\
\hat{g}_{j}^{c y}=\left(g_{j}^{c y}-d_{i}^{c y}\right) / d_{i}^{h} / \text {variance}[1]} \\
{\hat{g}_{j}^{w}=\log \left(g_{j}^{w} / d_{i}^{w}\right) / \text {variance}[2]\\
\hat{g}_{j}^{h}=\log \left(g_{j}^{h} / d_{i}^{h}\right) / \text {variance}[3]}\end{array}
$$


#### 2.2.2 confidence loss

对于 confidence loss，其采用 softmax loss:


$$
L_{conf}(x, c)=-\sum_{i \in P o s}^{N} x_{i j}^{p} \log \left(\hat{c}_{i}^{p}\right)-\sum_{i \in N e g} \log \left(\hat{c}_{i}^{0}\right) \\
\quad \text { where } \quad \hat{c}_{i}^{p}=\frac{\exp \left(c_{i}^{p}\right)}{\sum_{p} \exp \left(c_{i}^{p}\right)}
$$


### 2.3 **Data augmentation：**

即对每一张 image 进行如下之一变换获取一个 patch 进行训练：

- 直接使用原始的图像（即不进行变换）
- 采样一个 patch，保证与 ground truth 之间最小的 IoU 为：`0.1`，`0.3`，`0.5`，`0.7` 或 `0.9`
- 完全随机的采样一个 patch



同时在 paper 中还提到：

- 采样的 patch 占原始图像大小比例在 $[0.1,\,1]$之间
- 采样的 patch 的长宽比在 $[0.5,\,2]$ 之间
- 当 ground truth box 中心恰好在采样的 patch 中时，保留整个 ground truth box
- 最后每个 patch 被 resize 到固定大小，并且以 `0.5` 的概率随机的水平翻转



