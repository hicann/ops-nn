# MseLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算均方误差（MSE）损失对预测值predict的梯度。

- 计算公式：

  当`reduction`为`mean`时：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  其中`x.numel()`表示`x`中的元素个数。如果`reduction`不是`mean`,那么：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
  $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>predict</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的输入y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>输入</td>
      <td>公式中的输入grad，梯度值。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出MselossBackward，shape为predict、label、dout三者broadcast之后的结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td>公式中的reduction，取自前向损失，指定损失的归约方式，默认值为'mean'。对反向梯度而言：'mean'将每个梯度元素按predict的元素数缩放（系数为2/numel(predict)）；'none'和'sum'不做额外缩放（系数为2）。三种模式下输出y均保留predict、label、dout广播后的形状，不进行求和。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入predict、label、dout与输出y的数据类型必须一致。
- 输入predict、label、dout的shape必须满足broadcast关系，输出y的shape为三者broadcast之后的结果。
- 属性reduction仅支持'none'、'mean'、'sum'，传入其他取值时算子执行报错。
- 空tensor：任一输入包含0长度维（空tensor）时，算子在tiling阶段报错拒绝执行，其中reduction为'mean'时输入predict不能为空tensor。
- 动态Shape/动态Rank：本算子支持动态Shape与动态Rank编译。动态shape场景下，shape中的未知维度用-1表示；动态rank场景下，rank未知时用-2表示。shape与rank具体化后需满足上述数据类型一致与broadcast约束。
- 违反上述约束的输入在图编译或tiling阶段被拒绝执行；当前版本未定义算子专属错误码，报错为通用错误码。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE图模式 | - | 通过[算子IR](./op_graph/mse_loss_grad_proto.h)（REG_OP(MseLossGrad)）构图方式调用mse_loss_grad算子。 |
