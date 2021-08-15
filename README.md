[toc]

# 图神经网络在推荐系统的应用

# Fi-GNN

## 背景

> **「点击率预测」** 是在线广告和推荐系统等网络应用中的一项重要任务，其特点是多领域的。该任务的关键是 **「对不同特征域之间的特征交互」**进行建模。
>
> 基于深度学习的模型遵循了一种通用的范式：首先将原始的 **「稀疏」** 输入多场特征映射到 **「密集」** 的场嵌入向量中，然后将这些特征向量拼接在一起，输入到**「深度神经网络(DNN)」**或其他专门设计的网络中，以学习高阶特征交互。然而，特征域的简单非结构化组合将不可避免地限制以足够灵活和显式的方式建模不同字段之间复杂交互的能力。
>
> 论文提出 **「图结构中直观地表示多字段的特征」**，其中每个节点对应一个特征字段，不同的字段可以通过边进行交互。因此，建模特征交互的任务可以转换为对相应图上的节点交互进行建模。
>
> 论文设计了一个新的模型- 特征交互图神经网络(Fi-GNN)。利用图的强代表性，不仅可以灵活、明确地对复杂的特征交互进行建模，而且可以为CTR预测提供良好的模型解释。

## 模型

### 整体框架

> 模型整体结构如下：
>
> 第一部分：对离散型的变量做 embedding 操作；
>
> 第二部分：利用图网络来学习结构信息；
>
> 第三部分：整合图信息预测；

img

### 第一部分

> 与传统的ctr模型结构相同，稀疏特征经过embedding layer转成embedding vector，之后利用Multi-head Self-attention Layer建模特征pairs的关系，得到特征表示，具体如下：

img2

Img3

### 第二部分

#### 图构建

> 这是整篇论文的精华部分，主要思路是将多个域特征(假设总共<a href="https://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="https://latex.codecogs.com/svg.latex?m" title="m" /></a>个域特征)表示成一张图<a href="https://www.codecogs.com/eqnedit.php?latex=G(N,&space;\varepsilon)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?G(N,&space;\varepsilon)" title="G(N, \varepsilon)" /></a>，图中的每个节点<a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}\in&space;N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n_{i}\in&space;N" title="n_{i}\in N" /></a>对应特征域<a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?i" title="i" /></a>，不同域通过边交互。由于所有的特征都需要交互，所以该图中的任意两个节点之间都有一个双向的箭头表示，用邻接矩阵表示的话，会得到一个全是 1 的邻接矩阵。

> 基于GGNN，论文设计了基于特征图的节点交互模型 Fi-GNN，它能够以灵活和显式的方式对交互进行建模。

> 在Fi-GNN中，每个节点<a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n_{i}" title="n_{i}" /></a>对应着隐藏状态向量<a href="https://www.codecogs.com/eqnedit.php?latex=h_{i}^{t}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?h_{i}^{t}" title="h_{i}^{t}" /></a>，图的状态由这些节点状态组成：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=H^{t}=[h_{1}^{t},&space;h_{2}^{t},&space;h_{3}^{t},...,h_{m}^{t}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H^{t}=[h_{1}^{t},&space;h_{2}^{t},&space;h_{3}^{t},...,h_{m}^{t}]" title="H^{t}=[h_{1}^{t}, h_{2}^{t}, h_{3}^{t},...,h_{m}^{t}]" /></a>
>
> 式中，<a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t" title="t" /></a>代表相互作用的步骤，第一部分multi-head self-attention layer输出的向量是节点的初始状态<a href="https://www.codecogs.com/eqnedit.php?latex=H^{1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H^{1}" title="H^{1}" /></a>。
>
> 节点以循环的方式交互并更新它们的状态，在每个交互步骤中，节点与邻居聚合转换后的状态信息，然后通过GRU和剩余连接根据聚合的信息和历史更新节点状态，具体过程如下介绍。

#### 节点状态更新

##### State Aggregation

> 在相互作用步骤t，每个节点将聚合来自邻居的状态信息，节点<a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n_{i}" title="n_{i}" /></a>的汇总信息为其邻居节点变换后的状态信息之和，具体公式如下：

img

> 式中，<a href="https://www.codecogs.com/eqnedit.php?latex=W_{p}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W_{p}" title="W_{p}" /></a>是转换函数，<a href="https://www.codecogs.com/eqnedit.php?latex=A\in&space;\mathbb{R}^{m*m}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?A\in&space;\mathbb{R}^{m*m}" title="A\in \mathbb{R}^{m*m}" /></a>是包含边权重的邻接矩阵。例如，<a href="https://www.codecogs.com/eqnedit.php?latex=A[n_{j},n_{i}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?A[n_{j},n_{i}]" title="A[n_{j},n_{i}]" /></a>是节点<a href="https://www.codecogs.com/eqnedit.php?latex=n_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n_{j}" title="n_{j}" /></a>到<a href="https://www.codecogs.com/eqnedit.php?latex=n_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n_{i}" title="n_{i}" /></a>的边的权值，可以反映它们之间相互作用的重要性。显然，转换函数和邻接矩阵决定了节点间的相互作用。由于每条边上的相互作用应该是不同的，我们的目标是实现边的相互作用，这需要每条边有一个唯一的权值和变换函数。转换函数和邻接矩阵的生成方式如下：

###### (1) Attentional Edge Weights

> 传统GNN模型中的邻接矩阵通常为二进制形式，只包含0和1。它只能反映节点之间的连接关系，而不能反映节点之间关系的重要性。
>
> 为了推断出不同节点之间交互作用的重要性，论文提出通过一个注意机制来学习边权值：根据节点的初始状态计算边权值，具体如下：

img

###### (2) Edge-wise Transformation

> 所有边如果采用固定变换函数无法对灵活的相互作用进行建模，必须对每条边进行唯一的变换。但是论文构建的图是具有大量边的完全图，简单地为每条边分配一个唯一的转换权值会消耗太多的参数空间和运行时间。
>
> 为了减少时间和空间的复杂性，同时实现边向变换，论文为每个节点分配了两个矩阵<a href="https://www.codecogs.com/eqnedit.php?latex=W_{out}^{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W_{out}^{i}" title="W_{out}^{i}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=W_{in}^{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W_{in}^{i}" title="W_{in}^{i}" /></a>，转换矩阵的构造过程如下：

img

##### State Update

> 汇聚状态信息后，节点通过GRU残差连接更新状态向量，具体如下：

###### (1) State update via GRU

img

###### (2) State update via Residual Connections

> 论文引入了残差连接来随着GRU更新状态，这可以促进低阶特征重用和梯度反向传播，如下：

img

### 第三部分

> 每个字段节点的最终状态捕获了全局信息。换句话说，这些域节点是邻居感知的。论文分别对每个领域的最终状态进行评分，并使用一个注意力机制来衡量它们对整体预测的影响，具体如下：

img

## 实验结果

### 模型比较

> 论文对比了Fi-GNN与经典的CTR模型的效果，具体如下：
>
> - LR效果最差，表明单一的特征在CTR预测中是不够的
> - FM和AFM效果优于LR，AFM效果优于FM，证明二阶特征的交互对CTR建模的有效性，attention在不同特征的交互起着重要作用
> - 高阶模型的效果均优于一阶和二阶，这表明CTR建模中二阶特征交互是不够的
> - Fi-GNN模型效果最好，这主要归因于在节点的交互中，图结构化表达的优越能力和GNN的有效性。

img

### Alation study

> 论文在Fi-GNN的做了不同结构的abltaion study，具体如下：

img

## Model Explanation

> 论文给出了Fi-GNN在边和节点的attention weights，具体如下：

### Attentional Edge weights

img

### Attentional Node weights







# GMCF

## 背景



## 模型



## 结果

