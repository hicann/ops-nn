# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/celu_v2/README.md">celu_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
<tr>
    <td>activation</td>
    <td><a href="../../activation/elu/README.md">elu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量self中的每个元素x调用指数线性单元激活函数ELU，并将得到的结果存入输出张量out中。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/elu_grad/README.md">elu_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对应Elu操作的反向传播梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/elu_grad_v2/README.md">elu_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>激活函数的反向计算，输出ELU激活函数正向输入的梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/erfinv/README.md">erfinv</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/fast_gelu/README.md">fast_gelu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>快速高斯误差线性单元激活函数。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/fast_gelu_grad/README.md">fast_gelu_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>FastGelu反向传播。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/fatrelu_mul/README.md">fatrelu_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Threshold计算，将计算结果与x2相乘。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/ge_glu_grad_v2/README.md">ge_glu_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成aclnnGeGlu和aclnnGeGluV3的反向。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/ge_glu_v2/README.md">ge_glu_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>高斯误差线性单元激活门函数，针对aclnnGeGlu，扩充了设置激活函数操作数据块方向的功能。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu/README.md">gelu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>高斯误差线性单元激活函数。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu_grad/README.md">gelu_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>激活函数的反向传播算子，用于计算Gelu激活函数的梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu_grad_v2/README.md">gelu_grad_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu_mul/README.md">gelu_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行GELU计算，将计算结果与x2相乘。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu_quant/README.md">gelu_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将GeluV2与DynamicQuant/AscendQuantV2进行融合，对输入的数据self进行GELU激活后，对激活的结果进行量化，输出量化后的结果。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/gelu_v2/README.md">gelu_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_shrink/README.md">hard_shrink</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_shrink_grad/README.md">hard_shrink_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_sigmoid/README.md">hard_sigmoid</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_sigmoid_grad/README.md">hard_sigmoid_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_swish/README.md">hard_swish</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hard_swish_grad/README.md">hard_swish_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/hardtanh_grad/README.md">hardtanh_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>激活函数aclnnHardtanh的反向。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/heaviside/README.md">heaviside</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入input中每个元素的Heaviside阶跃函数，作为模型的激活函数。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/leaky_relu/README.md">leaky_relu</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/leaky_relu_grad/README.md">leaky_relu_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/logsigmoid/README.md">logsigmoid</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/logsigmoid_grad/README.md">logsigmoid_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/logsoftmax_grad/README.md">logsoftmax_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/logsoftmax_v2/README.md">logsoftmax_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/mish/README.md">mish</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/mish_grad/README.md">mish_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/prelu/README.md">prelu</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/prelu_grad_reduce/README.md">prelu_grad_reduce</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/prelu_grad_update/README.md">prelu_grad_update</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/relu/README.md">relu</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/relu_grad/README.md">relu_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对应Relu操作的反向传播梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/relu_grad_v2/README.md">relu_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对应ReluV2操作的反向传播梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/relu_v2/README.md">relu_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>激活函数，返回与输入tensor shape相同的tensor，tensor中value大于等于0时，取该value，小于0，取0。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/selu/README.md">selu</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/selu_grad/README.md">selu_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/shrink/README.md">shrink</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/sigmoid/README.md">sigmoid</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/sigmoid_grad/README.md">sigmoid_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/silu_grad/README.md">silu_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softmax_cross_entropy_with_logits/README.md">softmax_cross_entropy_with_logits</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算softmax和cross entropy的交叉熵损失，并给出对输入logits的反向梯度。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softmax_grad/README.md">softmax_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成softmax的反向传播。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softmax_v2/README.md">softmax_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量计算Softmax值。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softplus/README.md">softplus</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>激活函数，用于将任意实数映射到正数区间。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softplus_v2/README.md">softplus_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softplus_v2_grad/README.md">softplus_v2_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softshrink/README.md">softshrink</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/softshrink_grad/README.md">softshrink_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/squared_relu/README.md">squared_relu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>SquaredRelu函数是一个基于标准Relu函数的变体，其主要特点是对Relu函数的输出进行平方，常作为模型的激活函数。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/swi_glu/README.md">swi_glu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Swish门控线性单元激活函数，实现x的SwiGlu计算。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/swi_glu_grad/README.md">swi_glu_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成aclnnSwiGlu的反向计算，完成x的SwiGlu反向梯度计算。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/swish/README.md">swish</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/swish_grad/README.md">swish_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Swish激活函数的反向传播，用于计算Swish激活函数的梯度。</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/threshold/README.md">threshold</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>activation</td>
    <td><a href="../../activation/threshold_grad_v2_d/README.md">threshold_grad_v2_d</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/assert/README.md">assert</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI CPU</td>
    <td>断言给定条件为 true，如果输入张量 input_condition 判定为 false，则打印 input_data 中的张量列表。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/identity/README.md">identity</a></td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/identity_n/README.md">identity_n</a></td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/rank/README.md">rank</a></td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/shape/README.md">shape</a></td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>control</td>
    <td><a href="../../control/shape_n/README.md">shape_n</a></td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/conv3d_backprop_filter_v2/README.md">conv3d_backprop_filter_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算三维卷积核权重张量w的梯度 ∂L/∂w。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/conv3d_backprop_input_v2/README.md">conv3d_backprop_input_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算三维卷积正向的输入张量x对损失函数L的梯度 ∂L/∂x。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/conv3d_transpose_v2/README.md">conv3d_transpose_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算三维卷积的转置（反卷积）相对于输入的梯度。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/conv3d_v2/README.md">conv3d_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现 3D 卷积功能。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/conv2d_v2/README.md">conv2d_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现 2D 卷积功能。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/convolution_backward/README.md">convolution_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>卷积的反向传播。根据输出掩码设置计算输入、权重和偏差的梯度。此函数支持1D、2D和3D卷积。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/convolution_forward/README.md">convolution_forward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现卷积功能，支持 1D 卷积、2D 卷积、3D 卷积，同时支持转置卷积、空洞卷积、分组卷积。</td>
  </tr>
  <tr>
    <td>conv</td>
    <td><a href="../../conv/deformable_conv2d/README.md">deformable_conv2d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现卷积功能，支持2D卷积，同时支持可变形卷积、分组卷积。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_abs/README.md">foreach_abs</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表中的每个张量执行逐元素绝对值运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_acos/README.md">foreach_acos</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表中的每个张量执行逐元素反余弦运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_add_list/README.md">foreach_add_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>两个Tensor列表中的元素逐个相加，并可以通过alpha参数调整相加系数。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_add_scalar/README.md">foreach_add_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将指定的张量值加到张量列表中的每个张量中。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_add_scalar_list/README.md">foreach_add_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>输入张量列表和输入标量列表执行逐元素相加运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcdiv_list/README.md">foreach_addcdiv_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对多个张量进行逐元素加、乘、除操作，x2_i和x3_i进行逐元素相除，并将结果乘以scalars，再与x1_i相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcdiv_scalar/README.md">foreach_addcdiv_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对多个张量进行逐元素加、乘、除操作，x2_i和x3_i进行逐元素相除，并将结果乘以scalar，再与x1_i相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcdiv_scalar_list/README.md">foreach_addcdiv_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对多个张量进行逐元素加、乘、除操作，x2_i和x3_i进行逐元素相除，并将结果乘以scalars_i，再与x1_i相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcmul_list/README.md">foreach_addcmul_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>先对张量列表x2和张量列表x3执行逐元素乘法，并将结果乘以张量scalars，最后将之前计算的结果与张量列表x1执行逐元素相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcmul_scalar/README.md">foreach_addcmul_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>先对张量列表x2和张量列表x3执行逐元素乘法，再乘以张量scalar，最后将之前计算的结果与张量列表x1执行逐元素相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_addcmul_scalar_list/README.md">foreach_addcmul_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>先对张量列表x2和张量列表x3执行逐元素乘法，再与张量scalars进行逐元素乘法，最后将之前计算的结果与张量列表x1执行逐元素相加。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_asin/README.md">foreach_asin</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素进行反正弦函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_atan/README.md">foreach_atan</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素进行反正切函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_copy/README.md">foreach_copy</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>用于实现两个张量列表内容的复制。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_cos/README.md">foreach_cos</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素进行余弦函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_cosh/README.md">foreach_cosh</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素进行双曲余弦函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_div_list/README.md">foreach_div_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x1和张量列表x2执行逐元素除法。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_div_scalar/README.md">foreach_div_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算张量列表x除以张量scalar。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_div_scalar_list/README.md">foreach_div_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x和标量列表scalars执行逐元素除法。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_erf/README.md">foreach_erf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素进行误差函数运算（也称之为高斯误差函数，error function or Gaussian error function）。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_erfc/README.md">foreach_erfc</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>按元素执行从x到无穷大积分的互补误差函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_exp/README.md">foreach_exp</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行指数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_expm1/README.md">foreach_expm1</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量执行指数运算，然后将得到的结果减1。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_lerp_list/README.md">foreach_lerp_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对两个张量列表对应位置元素执行插值计算，其中张量列表weight是插值系数。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_lerp_scalar/README.md">foreach_lerp_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对两个张量列表对应位置元素执行插值计算，其中标量weight是插值系数。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_log/README.md">foreach_log</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表执行逐元素自然对数运算（ln(x)）。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_log10/README.md">foreach_log10</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表中的每一个元素执行以10为底的对数函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_log1p/README.md">foreach_log1p</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表中的每一个元素执行先加一再以e为底的对数函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_log2/README.md">foreach_log2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表中的每一个元素执行以2为底的对数函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_maximum_list/README.md">foreach_maximum_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x1和张量列表x2执行逐元素比较，计算每个元素对应的最大值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_maximum_scalar/README.md">foreach_maximum_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表和张量scalar执行逐元素比较，计算每个元素对应的最大值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_maximum_scalar_list/README.md">foreach_maximum_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最大值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_minimum_list/README.md">foreach_minimum_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x1和张量列表x2执行逐元素比较，计算每个元素对应的最小值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_minimum_scalar/README.md">foreach_minimum_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x和张量scalar执行逐元素比较，计算每个元素对应的最小值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_minimum_scalar_list/README.md">foreach_minimum_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最小值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_mul_list/README.md">foreach_mul_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对两个输入张量列表执行逐元素相乘。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_mul_scalar/README.md">foreach_mul_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量与张量scalar执行相乘运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_mul_scalar_list/README.md">foreach_mul_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表与标量列表执行逐元素相乘运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_neg/README.md">foreach_neg</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入张量列表中每个张量的相反数。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_non_finite_check_and_unscale/README.md">foreach_non_finite_check_and_unscale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>遍历scaledGrads中的所有Tensor，检查是否存在Inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_norm/README.md">foreach_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行范数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_pow_list/README.md">foreach_pow_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行x2次方运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_pow_scalar/README.md">foreach_pow_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行n次方运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_pow_scalar_and_tensor/README.md">foreach_pow_scalar_and_tensor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行x次方运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_pow_scalar_list/README.md">foreach_pow_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行n次方运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_reciprocal/README.md">foreach_reciprocal</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行倒数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_round_off_number/README.md">foreach_round_off_number</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量执行指定精度的四舍五入运算，可通过roundMode指定舍入方式。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sigmoid/README.md">foreach_sigmoid</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行Sigmoid函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sign/README.md">foreach_sign</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入张量列表中每个张量的符号值。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sin/README.md">foreach_sin</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行正弦函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sinh/README.md">foreach_sinh</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行双曲正弦函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sqrt/README.md">foreach_sqrt</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行平方根运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sub_list/README.md">foreach_sub_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入的两个张量列表执行逐元素相减运算，并可以通过alpha参数调整相减系数。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sub_scalar/README.md">foreach_sub_scalar</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量与张量scalar执行相减运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_sub_scalar_list/README.md">foreach_sub_scalar_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量与标量列表scalars的每个标量逐元素执行相减运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_tan/README.md">foreach_tan</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行正切函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_tanh/README.md">foreach_tanh</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量列表的每个张量进行双曲正切函数运算。</td>
  </tr>
  <tr>
    <td>foreach</td>
    <td><a href="../../foreach/foreach_zero_inplace/README.md">foreach_zero_inplace</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>原地更新输入张量列表，输入张量列表的每个张量置为0。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/apply_top_k_top_p_with_sorted/README.md">apply_top_k_top_p_with_sorted</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对原始输入logits进行top-k和top-p采样过滤。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/embedding_bag/README.md">embedding_bag</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据indices从weight中获得一组被聚合的数，然后根据offsets的偏移和mode指定的聚合模式对获取的数进行max、sum、mean聚合。其余参数则更细化了计算过程的控制。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/embedding_dense_grad_v2/README.md">embedding_dense_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现Embedding的反向计算，将相同索引indices对应grad的一行累加到out上。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/expand_into_jagged_permute/README.md">expand_into_jagged_permute</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将稀疏数据置换索引从表维度扩展到批次维度，适用于稀疏特征在不同rank中具有不同批次大小的情况。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/gather_elements/README.md">gather_elements</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/gather_elements_v2/README.md">gather_elements_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入tensor中指定的维度dim进行数据聚集。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/gather_nd/README.md">gather_nd</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/gather_v2/README.md">gather_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/index/README.md">index</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据索引indices将输入x对应坐标的数据取出。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/index_fill_d/README.md">index_fill_d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>沿输入self的给定轴dim，将index指定位置的值使用value进行替换。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/index_put_v2/README.md">index_put_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据索引 indices 将输入 self 对应坐标的数据与输入 values 进行替换或累加。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/index_put_with_sort_v2/README.md">index_put_with_sort_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行scatter操作，根据pos_idx的值循环对应的values，累加/替换到linear_index指向的self的位置。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/inplace_index_add_with_sorted/README.md">inplace_index_add_with_sorted</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将alpha和value相乘，并按照sorted_indices中的顺序，累加到var张量的指定axis维度中。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/inplace_scatter_add/README.md">inplace_scatter_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据给定的indices，将updates中的值加到输入张量var的第一维度上。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/linear_index/README.md">linear_index</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将多维数组 / 矩阵的元素位置映射为单一整数的索引方式。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/linear_index_v2/README.md">linear_index_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将多维数组 / 矩阵的元素位置映射为单一整数的索引方式。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/masked_scatter/README.md">masked_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据掩码(mask)张量中元素为True的位置，复制(source)中的元素到(selfRef)对应的位置上。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/matrix_inverse/README.md">matrix_inverse</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/non_zero/README.md">non_zero</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>找出`self`中非零元素的位置，设self的维度为D，self中非零元素的个数为N，则返回`out`的shape为N * D，每一列表示一个非零元素的位置坐标。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/quant_update_scatter/README.md">quant_update_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>先将updates在quantAxis轴上进行量化：quantScales对updates做缩放操作，quantZeroPoints做偏移。然后将量化后的updates中的值按指定的轴axis，根据索引张量indices逐个更新selfRef中对应位置的值。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/repeat_interleave/README.md">repeat_interleave</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将tensor self进行flatten后，重复Tensor repeats中的相应次数。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/repeat_interleave_grad/README.md">repeat_interleave_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>算子RepeatInterleave的反向, 将y_grad tensor的axis维度按repeats进行ReduceSum。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/reverse_v2/README.md">reverse_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_add/README.md">scatter_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将tensor src中的值按指定的轴和方向以及对应的位置关系逐个替换/累加/累乘至tensor self中。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_add_with_sorted/README.md">scatter_add_with_sorted</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_elements/README.md">scatter_elements</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_elements_v2/README.md">scatter_elements_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将tensor src中的值按指定的轴和方向以及对应的位置关系逐个替换/累加/累乘至tensor self中。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_list/README.md">scatter_list</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将稀疏矩阵更新应用到变量引用中。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_nd/README.md">scatter_nd</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>算子功能：拷贝data的数据至out，同时在指定indices处根据updates更新out中的数据。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_nd_update/README.md">scatter_nd_update</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/scatter_update/README.md">scatter_update</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/top_k_top_p_sample/README.md">top_k_top_p_sample</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据输入词频logits、topK/topP采样参数、随机采样权重分布q，进行topK-topP-sample采样计算，输出每个batch的最大词频logitsSelectIdx，以及topK-topP采样后的词频分布logitsTopKPSelect。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/top_k_top_p_sample_v2/README.md">top_k_top_p_sample</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据输入词频logits、topK/topP/minP采样参数、随机采样权重分布q，进行topK-topP-minP-sample采样计算。当输入isNeedSampleResult为false时，输出每个batch的最大词频logitsSelectIdx，以及topK-topP-minP采样后的词频分布logitsTopKPSelect；当输入isNeedSampleResult为true时，输出topK-topP-minP采样后的中间计算结果logitsIdx和logitsSortMasked，其中logitsSortMasked为词频logits经过topK-topP-minP采样计算后的中间结果，logitsIdx为logitsSortMasked在logits中对应的索引。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/unique_consecutive/README.md">unique_consecutive</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>去除每一个元素后的重复元素。</td>
  </tr>
  <tr>
    <td>index</td>
    <td><a href="../../index/unique_with_counts_ext2/README.md">unique_with_counts_ext2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/binary_cross_entropy/README.md">binary_cross_entropy</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/binary_cross_entropy_grad/README.md">binary_cross_entropy_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/chamfer_distance_grad/README.md">chamfer_distance_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>ChamferDistance（倒角距离）的反向算子，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/cross_entropy_loss/README.md">cross_entropy_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算输入的交叉熵损失。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/cross_entropy_loss_grad/README.md">cross_entropy_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnCrossEntropyLoss的反向传播。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/cross_v2/README.md">cross_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>用于计算x1和x2张量在维度dim上的向量叉积。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/ctc_loss_v2/README.md">ctc_loss_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算连接时序分类损失值。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/ctc_loss_v2_grad/README.md">ctc_loss_v2_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>ctcLoss的反向传播，计算CTC的损失梯度。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/ctc_loss_v3/README.md">ctc_loss_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算连续（未分段）时间序列与目标序列之间的损失。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/ctc_loss_v3_grad/README.md">ctc_loss_v3_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算连续（未分段）时间序列与目标序列之间的损失的反向。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/kl_div_loss_grad/README.md">kl_div_loss_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/l1_loss_grad/README.md">l1_loss_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/logit/README.md">logit</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。它可以将一个介于0和1之间的概率值转换为一个实数值。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/logit_grad/README.md">logit_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>aclnnLogit的反向传播。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/lp_loss/README.md">lp_loss</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/mse_loss/README.md">mse_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入和目标中每个元素之间的均方误差。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/mse_loss_grad/README.md">mse_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>均方误差函数aclnnMseLoss的反向传播。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/mse_loss_grad_v2/README.md">mse_loss_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>均方误差函数aclnnMseLoss的反向传播。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/mse_loss_v2/README.md">mse_loss_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入x和目标y中每个元素之间的均方误差。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/multilabel_margin_loss/README.md">multilabel_margin_loss</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/nll_loss/README.md">nll_loss</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/nll_loss_grad/README.md">nll_loss_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/sigmoid_cross_entropy_with_logits_grad_v2/README.md">sigmoid_cross_entropy_with_logits_grad_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/sigmoid_cross_entropy_with_logits_v2/README.md">sigmoid_cross_entropy_with_logits_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算输入logits与标签target之间的BCELoss损失</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/smooth_l1_loss_grad_v2/README.md">smooth_l1_loss_grad_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/smooth_l1_loss_v2/README.md">smooth_l1_loss_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算SmoothL1损失函数。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/soft_margin_loss/README.md">soft_margin_loss</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>loss</td>
    <td><a href="../../loss/soft_margin_loss_grad/README.md">soft_margin_loss_grad</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/addmv/README.md">addmv</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/batch_mat_mul_v3/README.md">batch_mat_mul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成带batch的矩阵乘计算。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/convert_weight_to_int4_pack/README.md">convert_weight_to_int4_pack</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/dual_level_quant_batch_matmul/README.md">dual_level_quant_batch_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>完成二级量化mxfp4的矩阵乘计算。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/fused_mat_mul/README.md">fused_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>矩阵乘与通用向量计算融合。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/fused_quant_mat_mul/README.md">fused_quant_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>量化矩阵乘与通用向量计算融合。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/gemm/README.md">gemm</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/gemm_v2/README.md">gemm_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算α乘以A与B的乘积，再与β和input C的乘积求和。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/gemm_v3/README.md">gemm_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算α乘以A与B的乘积，再与β和input C的乘积求和。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/mat_mul_v3/README.md">mat_mul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成通用矩阵乘计算。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/mv/README.md">mv</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/quant_batch_matmul_v3/README.md">quant_batch_matmul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/quant_batch_matmul_v4/README.md">quant_batch_matmul_v4</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化的矩阵乘计算。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/quant_matmul_reduce_sum/README.md">quant_matmul_reduce_sum</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/transpose_batch_mat_mul/README.md">transpose_batch_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成张量x1与张量x2的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的序列进行变更。permX1代表张量x1的转置序列，permX2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/transpose_quant_batch_mat_mul/README.md">transpose_quant_batch_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>完成张量x1与张量x2量化的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的序列进行变更。permX1代表张量x1的转置序列，permX2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。</td>
  </tr>
  <tr>
    <td>matmul</td>
    <td><a href="../../matmul/weight_quant_batch_matmul_v2/README.md">weight_quant_batch_matmul_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/ada_layer_norm/README.md">ada_layer_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>AdaLayerNorm算子将LayerNorm和下游的Add、Mul融合起来，通过自适应参数scale和shift来调整归一化过程。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/ada_layer_norm_quant/README.md">ada_layer_norm_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>AdaLayerNormQuant算子将AdaLayerNorm和下游的量化（目前仅支持DynamicQuant）融合起来。该算子主要是用于执行自适应层归一化的量化操作，即将输入数据进行归一化处理，并将其量化为低精度整数，以提高计算效率和减少内存占用。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/ada_layer_norm_v2/README.md">ada_layer_norm_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>AdaLayerNormV2算子将LayerNorm和下游的Add、Mul融合起来，通过自适应参数scale和shift来调整归一化过程。相比AdaLayerNorm算子，输出新增2个参数（输入的均值和输入的标准差的倒数）；weight和bias支持的数据类型增加对应约束。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_layer_norm/README.md">add_layer_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现AddLayerNorm功能。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_layer_norm_grad/README.md">add_layer_norm_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>LayerNorm是一种归一化方法，可以将网络层输入数据归一化到[0, 1]之间。LayerNormGrad算子是深度学习中用于反向传播阶段的一个关键算子，主要用于计算LayerNorm操作的梯度。AddLayerNormGrad算子是将Add和LayerNormGrad融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_layer_norm_quant/README.md">add_layer_norm_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>LayerNorm算子是大模型常用的归一化操作。AddLayerNormQuant算子将LayerNorm前的Add算子和LayerNorm归一化输出给1个或2个下游的量化算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm/README.md">add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm_cast/README.md">add_rms_norm_cast</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm_dynamic_quant/README.md">add_rms_norm_dynamic_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuantV2算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。AddRmsNormDynamicQuant算子相较于AddRmsNormDynamicQuantV2在RmsNorm计算过程中增加了偏置项betaOptional参数，即计算对应公式中的beta，以及新增输出配置项output_mask参数，用于配置是否输出对应位置的量化结果 。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm_dynamic_quant_v2/README.md">add_rms_norm_dynamic_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuantV2算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm_quant/README.md">add_rms_norm_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/add_rms_norm_quant_v2/README.md">add_rms_norm_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。AddRmsNormQuantV2算子相较于AddRmsNormQuant在RmsNorm计算过程中增加了偏置项bias参数，即计算公式中的`bias`。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/batch_norm_elemt/README.md">batch_norm_elemt</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/batch_norm_grad_v3/README.md">batch_norm_grad_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/batch_norm_v3/README.md">batch_norm_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对一个批次的数据做批量归一化处理，正则化之后生成的数据的统计结果为0均值、1标准差。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/bn_training_reduce/README.md">bn_training_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/deep_norm/README.md">deep_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/deep_norm_grad/README.md">deep_norm_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>DeepNorm的反向传播，完成张量x、张量gx、张量gamma的梯度计算，以及张量dy的求和计算。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/dua_quantize_add_layer_norm/README.md">dua_quantize_add_layer_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>DuaQuantizeAddLayerNorm是一个复杂的计算，结合了量化、加法和层归一化（Layer Normalization）的功能。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/gemma_rms_norm/README.md">gemma_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>GemmaRmsNorm算子是大模型常用的归一化操作，相比RmsNorm算子，在计算时对gamma执行了+1操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm/README.md">group_norm</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm_grad/README.md">group_norm_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>GroupNorm用于计算输入张量的组归一化结果，均值，标准差的倒数，该算子是对GroupNorm的反向计算。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm_silu/README.md">group_norm_silu</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm_swish/README.md">group_norm_swish</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm_swish_grad/README.md">group_norm_swish_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>aclnnGroupNormSwish的反向操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/group_norm_v2/README.md">group_norm_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算输入x的组归一化结果out，均值meanOut，以及标准差的倒数rstdOut。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/inplace_add_layer_norm/README.md">inplace_add_layer_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>是一种结合了原位加法和层归一化的算法。输出参数`x1`（对应公式中的`y`）、`x2`（对应公式中的`x`）复用了输入参数`x1`、`x2`输入地址，实现InplaceAddLayerNorm功能。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/inplace_add_rms_norm/README.md">inplace_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。InplaceAddRmsNorm是一种结合了原位加法和RMS归一化的操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/instance_norm_v3/README.md">instance_norm_v3</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/kv_rms_norm_rope_cache/README.md">kv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/layer_norm_grad/README.md">layer_norm_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>LayerNorm的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/layer_norm_grad_v3/README.md">layer_norm_grad_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>LayerNormV4的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/layer_norm_v3/README.md">layer_norm_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对指定层进行均值为0、标准差为1的归一化计算。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/layer_norm_v4/README.md">layer_norm_v4</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对指定层进行均值为0、标准差为1的归一化计算。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/lp_norm_v2/README.md">lp_norm_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>返回给定张量的矩阵范数或者向量范数。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/masked_softmax_with_rel_pos_bias/README.md">masked_softmax_with_rel_pos_bias</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>替换在swinTransformer中使用window attention计算softmax的部分。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/quantize_add_layer_norm/README.md">quantize_add_layer_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>是一种结合了量化、加法和层归一化的操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/quantized_batch_norm/README.md">quantized_batch_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将输入Tensor执行一个反量化的计算，再根据输入的weight、bias、epsilon执行归一化，最后根据输出的outputScale以及outputZeroPoint执行量化。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/renorm/README.md">renorm</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/rms_norm/README.md">rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/rms_norm_grad/README.md">rms_norm_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm的反向计算。用于计算RmsNorm的梯度，即在反向传播过程中计算输入张量的梯度。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/rms_norm_quant/README.md">rms_norm_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuant算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。</td>
  </tr>
    <tr>
    <td>norm</td>
    <td><a href="../../norm/rms_norm_quant_v2/README.md">rms_norm_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuantV2算子将RmsNorm算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/sync_batch_norm_backward_elemt/README.md">sync_batch_norm_backward_elemt</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/sync_batch_norm_backward_reduce/README.md">sync_batch_norm_backward_reduce</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/sync_batch_norm_gather_stats/README.md">sync_batch_norm_gather_stats</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>norm</td>
    <td><a href="../../norm/sync_batch_norm_gather_stats_with_counts/README.md">sync_batch_norm_gather_stats_with_counts</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/advance_step/README.md">advance_step</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_adagrad_d/README.md">advance_step</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>ApplyAdagrad是自适应梯度算法的核心更新操作，主要用于在优化器（如随机梯度下降）中更新模型参数。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_adam_w_v2/README.md">apply_adam_w_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现adamW优化器功能。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/adam_apply_one/README.md">adam_apply_one</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对模型中的一个参数（如权重），完成Adam优化算法的单步计算和更新。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/adam_apply_one_with_decay/README.md">adam_apply_one_with_decay</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对模型中的一个参数（如权重），完成Adam优化算法的单步计算和更新，在adam_apply_one基础上增加了权重衰减。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/adam_apply_one_with_decay_assign/README.md">adam_apply_one_with_decay_assign</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对模型中的一个参数（如权重），完成Adam优化算法的单步计算和更新，在adam_apply_one_with_decay基础上增加了赋值操作，确保更新后的值被正确的地保存。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_gradient_descent/README.md">apply_gradient_descent</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>用于在梯度下降训练过程中更新模型参数。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_gradient_descent/README.md">apply_gradient_descent</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>用于在梯度下降训练过程中更新模型参数。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_fused_ema_adam/README.md">apply_fused_ema_adam</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现FusedEmaAdam融合优化器功能，结合了Adam优化器和指数移动平均（EMA）技术。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/adaptive_avg_pool3d/README.md">adaptive_avg_pool3d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在指定三维输出shape信息的情况下，完成输入张量的3D自适应平均池化计算。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/adaptive_avg_pool3d_grad/README.md">adaptive_avg_pool3d_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>AdaptiveAvgPool3d的反向计算。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/adaptive_max_pool3d/README.md">adaptive_max_pool3d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作，输出池化后的值outputOut和索引indicesOut。aclnnAdaptiveMaxPool3d与aclnnMaxPool3d的区别在于，只需指定outputSize大小，并按outputSize的大小来划分pooling区域。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/adaptive_max_pool3d_grad/README.md">adaptive_max_pool3d_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>正向自适应最大池化的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/avg_pool3_d/README.md">avg_pool3_d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入Tensor进行窗口为kD * kH * kW、步长为sD * sH * sW的三维平均池化操作，其中k为kernelSize，表示池化窗口的大小，s为stride，表示池化操作的步长。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/avg_pool3_d_grad/README.md">avg_pool3_d_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool3d/README.md">max_pool3d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对于输入信号的输入通道，提供3维最大池化（Max pooling）操作。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool_v3/README.md">max_pool_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对于3维或4维的输入张量，进行最大池化（max pooling）操作。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool_with_argmax_v3/README.md">max_pool_with_argmax_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对于输入数据计算2维最大池化操作。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool_grad_with_argmax_v3/README.md">max_pool_grad_with_argmax_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>正向最大池化MaxPoolGradWithArgmaxV3的反向梯度。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool3d_grad_with_argmax/README.md">max_pool3d_grad_with_argmax</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>正向最大池化的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。</td>
  </tr>
  <tr>
    <td>pooling</td>
    <td><a href="../../pooling/max_pool3d_with_argmax_v2/README.md">max_pool3d_with_argmax_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/ascend_anti_quant_v2/README.md">ascend_anti_quant_v2</a></td>
    <td>✗</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
    <tr>
    <td>quant</td>
    <td><a href="../../quant/ascend_quant/README.md">ascend_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据输入的sacle和offset对输入x进行量化，且scale和offset的size需要是x的最后一维或1。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/ascend_quant_v2/README.md">ascend_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入x进行量化操作，支持设置axis以指定scale和offset对应的轴，scale和offset的shape需要满足和axis指定x的轴相等或1。axis当前支持设置最后两个维度。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dequant_bias/README.md">dequant_bias</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入x反量化操作，将输入的int32的数据转化为FLOAT16/BFLOAT16输出。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dequant_swiglu_quant/README.md">dequant_swiglu_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_block_quant/README.md">dynamic_block_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量，通过给定的row_block_size和col_block_size将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数scale，并根据scale对输入进行量化。输出最终的量化结果，以及每个块的量化参数scale。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_quant/README.md">dynamic_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>为输入张量进行per-token对称动态量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_dual_level_mx_quant/README.md">dynamic_dual_level_mx_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>通过给定的level0BlockSize和level1BlockSize将输入进行两次划分，以数据块为粒度，进行目的数据类型为FLOAT4类的MX二级量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/grouped_dynamic_block_quant/README.md">grouped_dynamic_block_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据传入的分组索引的起始值对各个group以基本块的粒度进行量化，并输出量化参数scale。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/grouped_dynamic_mx_quant/README.md">grouped_dynamic_mx_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据传入的分组索引的起始值，对传入的数据进行分组的float8的动态量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_mx_quant/README.md">dynamic_mx_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对传入的数据进行分组的float8的动态量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_quant_update_scatter/README.md">dynamic_quant_update_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_quant_update_scatter_v2/README.md">dynamic_quant_update_scatter_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/dynamic_quant_v2/README.md">dynamic_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>为输入张量进行per-token对称/非对称动态量化。在MOE场景下，每个专家的smooth_scales是不同的，根据输入的group_index进行区分。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/fake_quant_affine_cachemask/README.md">fake_quant_affine_cachemask</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对于输入数据self，使用scale和zeroPoint对输入self在指定轴axis上进行伪量化处理，并根据quantMin和quantMax对伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/flat_quant/README.md">flat_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该融合算子为输入矩阵x一次进行两次小矩阵乘法，即右乘输入矩阵kroneckerP2，左乘输入矩阵kroneckerP1，然后针对矩阵乘的结果进行per-token量化处理。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/group_quant/README.md">group_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入x进行分组量化操作。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/quantize/README.md">quantize</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量x进行量化处理。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/swi_glu_quant/README.md">swi_glu_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算，支持int8或int4量化输出。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/trans_quant_param/README.md">trans_quant_param</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>quant</td>
    <td><a href="../../quant/trans_quant_param_v2/README.md">trans_quant_param_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化计算参数scale数据类型的转换，将FLOAT32的数据类型转换为硬件需要的UINT64类型。</td>
  </tr>
   <tr>
    <td>rnn</td>
    <td><a href="../../../../rnn/bidirection_lstm/README.md">bidirection_lstm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。</td>
  </tr>
    <tr>
    <td>rnn</td>
    <td><a href="../../../../rnn/bidirection_lstm_v2/README.md">bidirection_lstm_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。</td>
  </tr>
  <tr>
    <td>rnn</td>
    <td><a href="../../rnn/dynamic_rnn/README.md">dynamic_rnn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基础循环神经网络 (Recurrent Neural Network) 算子，用于处理序列数据。它通过隐藏状态传递时序信息，适合处理具有时间/顺序依赖性的数据。</td>
  </tr>
  <tr>
    <td>rnn</td>
    <td><a href="../../rnn/dynamic_rnnv2/README.md">dynamic_rnnv2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基础循环神经网络 (Recurrent Neural Network) 算子，用于处理序列数据。它通过隐藏状态传递时序信息，适合处理具有时间/顺序依赖性的数据， 仅支持单层RNN。</td>
  </tr>
  <tr>
    <td>vfusion</td>
    <td><a href="../../vfusion/multi_scale_deformable_attention_grad/README.md">multi_scale_deformable_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MultiScaleDeformableAttention正向算子功能主要通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。</td>
  </tr>
  <tr>
    <td>vfusion</td>
    <td><a href="../../vfusion/multi_scale_deformable_attn_function/README.md">multi_scale_deformable_attn_function</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。</td>
  </tr>
  <tr>
    <td>vfusion</td>
    <td><a href="../../vfusion/scaled_masked_softmax_grad_v2/README.md">scaled_masked_softmax_grad_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>softmax的反向传播，并对结果进行缩放以及掩码。</td>
  </tr>
  <tr>
    <td>vfusion</td>
    <td><a href="../../vfusion/scaled_masked_softmax_v2/README.md">scaled_masked_softmax_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将输入的数据x先进行scale缩放和mask，然后执行softmax的输出。</td>
  </tr>
  <tr>
    <td>optim</td>
    <td><a href="../../optim/apply_adam_w/README.md">apply_adam_w</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现adamW优化器功能。</td>
  </tr>
  <tr>
    <td>hash</td>
    <td><a href="../../hash/embedding_hash_table_apply_adam_w/README.md">embedding_hash_table_apply_adam_w</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>查询hash表是否存在key，如果存在，则更新其value，如果不存在，则插入key。</td>
  </tr>
  <tr>
    <td>hash</td>
    <td><a href="../../hash/embedding_hash_table_export/README.md">embedding_hash_table_export</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>导出整个table表。</td>
  </tr>
  <tr>
    <td>hash</td>
    <td><a href="../../hash/embedding_hash_table_import/README.md">embedding_hash_table_import</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将输入的key和value插入hash表。</td>
  </tr>
  <tr>
    <td>hash</td>
    <td><a href="../../hash/embedding_hash_table_lookup_or_insert/README.md">embedding_hash_table_lookup_or_insert</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据key值查看table中是否存在key；如果存在则不插入value值，并且导出key当前位置上的值；如果不存在则对对key进行hash，找到位置后插入value。</td>
  </tr>
  <tr>
    <td>hash</td>
    <td><a href="../../hash/init_embedding_hash_table/README.md">init_embedding_hash_table</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>初始化hash表。</td>
  </tr>
</tbody>
</table>