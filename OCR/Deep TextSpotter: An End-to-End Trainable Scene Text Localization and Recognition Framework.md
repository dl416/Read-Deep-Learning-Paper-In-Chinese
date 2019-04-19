# Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework

这篇文章想把 Text Detect 和 Text Recognition 整合到一起来做。但是实际上还是分得比较开的两个部分。

首先 Text Detect 是一个类似 YOLO v2 的网络， Text Recognition 是一个类似 CRNN 的网络结构，值得提的是在 Text Detect 中间有一个 STN 来对检测到的字符进行矫正



将 OCR 的两个关键步骤进行和并是一个自然的想法，因为不论是文本检测还是字符识别都是参用CNN++的框架来进行，特征提取的步骤是没有必要重复去作的，这样自然能节约 OCR 过程中所需要的大量时间。

所以如何将两个部分进行粘连就是一个比较有技巧的手段了，这个粘连模块我还没有看到一个非常完美的模块。

文字由人类创作，出去一些花里胡哨的文字，应该是自然界中最有规律的物体了。所以识别准确率也一直比 Object Recognition 要高。