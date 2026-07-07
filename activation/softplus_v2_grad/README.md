# SoftplusV2Grad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     | √ |
| <term>Atlas 200I/500 A2 推理产品</term>                      | × |
| <term>Atlas 推理系列产品</term>                              | × |
| <term>Atlas 训练系列产品</term>                              | √ |

## 功能说明

- 算子功能：[aclnnSoftplus](../softplus_v2/docs/aclnnSoftplus.md)的反向传播，根据前向输入self和上游梯度gradOutput，计算对前向输入的梯度gradInput。

- 计算公式：

  $$
  gradInput = gradOutput \cdot \begin{cases}
  \dfrac{1}{1+e^{(-\beta \cdot self)}}, & \beta \cdot self \le threshold \\
  1, & \beta \cdot self > threshold
  \end{cases}
  $$

  当 $\beta \cdot self > threshold$ 时梯度退化为1.0（即 $gradInput = gradOutput$），避免 $exp$ 溢出，保证数值稳定性。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input_gradients</td>
      <td>输入</td>
      <td>上游梯度张量，公式中的gradOutput，支持0D~8D，与input_features同dtype、满足broadcast关系。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_features</td>
      <td>输入</td>
      <td>前向传播的原始输入张量，公式中的self，支持0D~8D，与input_gradients同dtype、满足broadcast关系。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选属性</td>
      <td><ul><li>公式中的beta，控制softplus陡峭程度，beta越大越接近ReLU。</li><li>默认值为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>可选属性</td>
      <td><ul><li>公式中的threshold，数值稳定性阈值，超过后梯度退化为1.0。</li><li>默认值为20.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_backprops</td>
      <td>输出</td>
      <td>对前向输入的梯度，公式中的gradInput，shape与dtype与input_features相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input_gradients 和 input_features 的 dtype 必须一致，不支持混合精度。
- input_gradients 和 input_features 满足 broadcast 关系，broadcast 后 shape 与 input_features 相同；output_backprops 的 shape 与 input_features 相同。
- beta 和 threshold 为 Host 侧标量属性，非 Tensor。
- 默认确定性实现。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="examples/test_aclnn_softplus_v2_grad.cpp">test_aclnn_softplus_v2_grad</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="examples/test_geir_softplus_v2_grad.cpp">test_geir_softplus_v2_grad</a></td>
    <td>通过<a href="op_graph/softplus_v2_grad_proto.h">算子IR</a>构图方式调用SoftplusV2Grad算子。</td>
  </tr>
</tbody></table>
