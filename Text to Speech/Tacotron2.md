## 1. 概述
Tacotron2是由Google Brain 2017年提出来的一个语音合成框架。
Tacotron2:一个完整神经网络语音合成方法。模型主要由三部分组成：
1. 声谱预测网络：一个引入注意力机制（attention）的基于循环的Seq2seq的特征预测网络，用于从输入的字符序列预测梅尔频谱的帧序列。
2. 声码器（vocoder）：一个WaveNet的修订版，用预测的梅尔频谱帧序列来生成时域波形样本。
3. 中间连接层：使用低层次的声学表征-梅尔频率声谱图来衔接系统的两个部分。
![tacotron2 模型架构](https://img-blog.csdnimg.cn/2019041521574663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQyNTMxNzM=,size_16,color_FFFFFF,t_70)
## 2.声谱预测网络：
声谱预测网络主要包含一个编码器和一个包含注意力机制的解码器。编码器把字符序列转换成一个隐层表征，解码器接受这个隐层表征用以预测声谱图。![在这里插入图片描述](https://img-blog.csdnimg.cn/20190422213020731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQyNTMxNzM=,size_16,color_FFFFFF,t_70)
### 编码器：
编码器模块包含一个字符嵌入层（Character Embedding），一个3层卷积，一个双向LSTM层。
1.  输入字符被编码成512维的字符向量；
2.  然后穿过一个三层卷积，每层卷积包含512个5x1的卷积核，即每个卷积核横跨5个字符，卷积层会对输入的字符序列进行大跨度上下文建模（类似于N-grams），这里使用卷积层获取上下文主要是由于实践中RNN很难捕获长时依赖；卷积层后接批归一化（batch normalization），使用ReLu进行激活；
3. 最后一个卷积层的输出被传送到一个双向的LSTM层用以生成编码特征，这个LSTM包含512个单元（每个方向256个单元）。

$$ f_e = ReLU(F_3*ReLU(F_2*ReLU(F_1*E(X))))$$
$$H = EncoderRecurrency(f_e) $$
其中，F1、F2、F3为3个卷积核，ReLU为每一个卷积层上的非线性激活，E表示对字符序列X做embedding，EncoderRecurrency表示双向LSTM。
```python
class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks  5x1 
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2), 
                         # 进行填充，保持输入，输出的维度一致。
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim)) 
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
    
```

## 注意力网络：
Tacotron2中使用了基于位置敏感的注意力机制（Attention-Based Models for Speech Recognition），是对之前注意力机制的扩展（Neural machine translation by jointly learning to align and translate）；这样处理可以使用之前解码处理的累积注意力权重作为一个额外的特征，因此使得模型在沿着输入序列向前移动的时候保持前后一致，减少了解码过程中潜在的子序列重复或遗漏。位置特征用32个长度为31的1维卷积核卷积得出，然后把输入序列和为位置特征投影到128维隐层表征，计算出注意力权重。关于具体的注意力机制计算可以参考这篇[博客](https://www.cnblogs.com/mengnan/p/9527797.html)。

Tacotron2中使用的是混合注意力机制，在对齐中加入了位置特征。
$$ e_i,_j=score(s_i,cα_{i−1},h_j) =v_a ^Ttanh(Ws_i+Vh_j+Uf_i,_j+b)$$

其中，$v_a$、W、V、U和b为待训练参数，$s_i$为当前解码器隐状态，$h_j$是当前编码器隐状态，$f_{i,j}$是之前的注意力权重$α_{i−1}$经卷积而得的位置特征(location feature)，$fi=F∗cα_{i−1}$。混合注意力机制能够同时考虑内容和输入元素的位置。

下面是代码实现：
```python
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')      # 解码器隐状态
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')                 # 编码器隐状态
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)    # 计算位置特征
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)            当前解码隐状态
        processed_memory: processed encoder outputs (B, T_in, attention_dim)         当前编码器隐状态
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)     之前累积的注意力权重

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))             # 当前解码隐状态
        processed_attention_weights = self.location_layer(attention_weights_cat)   # 位置特征，之前累积的注意力权重
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))   # 对齐能量值

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs        编码器隐状态
        processed_memory: processed encoder outputs    解码器隐状态
        attention_weights_cat: previous and cummulative attention weights    累积的注意力权重，位置特征
        mask: binary mask for padded data

        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)  # 对齐

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)         # 归一化的注意力权重
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)        # 上下文权重向量

        return attention_context, attention_weights
```
计算位置特征：位置特征由之前的累加的注意力权重经过卷积得来。
```python
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        """
              PARAMS
              ------
              attention_n_filters： 32维
              attention_kernel_size： 卷积核大小 31
              RETURNS
              -------
        """
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    # 位置特征使用累加的注意力权重卷积而来
    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
```

## 解码器：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429212357537.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQyNTMxNzM=,size_16,color_FFFFFF,t_70)

解码器是一个自回归循环神经网络，它从编码的输入序列预测输出声谱图，一次预测一帧。
1. 上一步预测出的频谱首先被传入一个“pre-net”，每层由256个隐藏ReLU单元组成的双层全连接层，pre-net作为一个信息瓶颈层（boottleneck）,对于学习注意力是必要的。
2.  pre-net的输出和注意力上下文向量拼接在一起，传给一个两层堆叠的由1024个单元组成的单向LSTM。LSTM的输出再次和注意力上下文向量拼接在一起，然后经过一个线性投影来预测目标频谱帧。
3.  最后，目标频谱帧经过一个5层卷积的“post-net”来预测一个残差叠加到卷积前的频谱帧上，用以改善频谱重构的整个过程。post-net每层由512个5X1卷积核组成，后接批归一化层，除了最后一层卷积，每层批归一化都用tanh激活。
4.  并行于频谱帧的预测，解码器LSTM的输出与注意力上下文向量拼接在一起，投影成一个标量后传递给sigmoid激活函数，来预测输出序列是否已经完成的概率。

下面看代码：
pre-net层：双层全连接层，使用了0.5的dropout。
```python
class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

```

解码器主体部分：可以看出解码器由哪些层组成，pre-net(预处理层)，attention_rnn,attention_layer,(注意力网络)，decoder_rnn(解码器LSTM)，linear_project(线性投影层)，gate_layer(判断预测是否结束)。
```python
class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')
```

解码器的步骤：
```python
    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
            
        # 注意力权重累加
        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        
        # 解码器的输入
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        
        # 初始化解码器状态
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
            
        # 处理解码器的输出
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
```

后处理网络（post-net）: 5层1维卷积层，最后一层没有使用tanh。使用残差进行计算。
$$y_{final} = y + y_r$$
其中y为原始输入；
$$y_r = PostNet(y)= W_{ps}f_{ps}+b_{ps}$$
其中$f_{ps} = F_{ps,i}∗x$,$x$为上个卷积层的输出或解码器输出。

```python
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        # 最后一层没有使用tanh
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x
```
### 与Tacotron对比：
1. Tacotron 2使用了更简洁的构造模块，在编码器和解码器中使用是普通的LSTM和卷积层;Tacotron中使用的是“CBHG”堆叠结构和GRU循环层；
2. Tacotron2在解码器的输出中没有使用“缩小因子（reduction factor）”，即每个解码步骤只输出一个单独的频谱帧。

关于[Tacotron](https://arxiv.org/pdf/1703.10135.pdf)部分可以参考这篇[博客](https://blog.csdn.net/yunnangf/article/details/79585089)。

## 3.声码器：
[Tacontron2](https://arxiv.org/abs/1712.05884)原论文使用的是一个修正版的[WaveNet](https://arxiv.org/abs/1609.03499)，把梅尔频谱特征表达逆变换为时域波形样本。现在也有用[Waveglow](https://arxiv.org/abs/1811.00002)声码器的。声码器可以分为一个完整的部分，有兴趣可以看看相关论文。

## 参考链接：
本文只解读了部分代码，完整代码可以参考https://github.com/NVIDIA/tacotron2，其中还有很多语音处理的部分。
https://www.cnblogs.com/mengnan/p/9527797.html
https://blog.csdn.net/yunnangf/article/details/79585089
http://blog.sina.com.cn/s/blog_8af106960102xj6j.html
