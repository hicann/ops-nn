# BatchNormV3

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |


## 功能说明

- 算子功能：对一个批次的数据做正则化处理，正则化之后生成的数据的统计结果为0均值、1标准差。

- 计算公式：

  $$
  y = \frac{(x - E(x))}{\sqrt{Var(x) + ε}} * γ + β
  $$
  E(x)表示均值，Var(x)表示方差，均需要在算子内部计算得到；ε表示一个极小的浮点数，防止分母为0的情况。

  
## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>输入</td>
      <td>
      <ul><li>进行批量归一化的输入张量，对应公式中的`x`。</li><li>shape支持4D、5D。</li></ul>
      </td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>
      <ul><li>进行批量归一化的权重，对应公式中的`γ`。</li><li>一个1D张量，shape与输入x的维度C相同，数据类型与x的数据类型支持以下组合：[x: float16, weight: float16/float32], [x: bfloat16, weight: bfloat16/float32], [x: float32, weight: float32]。</li></ul>
      </td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td><ul><li>进行批量归一化的偏置值，对应公式中的`β`。</li><li>一个1D张量，数据类型和shape与入参weight保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>输入</td>
      <td><ul><li>训练场景：训练期间动量更新前的均值；推理场景：推理期间使用的均值，对应公式中的`E(x)`。</li><li>一个1D张量，shape与输入x的维度C相同，数据类型与x的数据类型支持以下组合：[x: float16, running_mean: float16/float32], [x: bfloat16, running_mean: bfloat16/float32], [x: float32, running_mean: float32]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_var</td>
      <td>输入</td>
      <td>训练场景：训练期间动量更新前的方差；推理场景：推理期间使用的方差，对应公式中的`Var(x)`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到方差中的小值以避免除以零，对应公式中的`ε`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>可选属性</td>
      <td><ul><li>动量参数，用于更新训练期间的均值和方差。</li><li>默认值为0.1f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>可选属性</td>
      <td><ul><li>标记是否训练场景，true表示训练场景，false表示推理场景。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示批量归一化后的输出结果，对应公式中的`y`。</li><li>数据类型、数据格式、shape与输入x保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>输出</td>
      <td><ul><li>只训练场景输出，训练期间动量更新后的平均值。</li><li>数据类型、shape与输入running_mean保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_var</td>
      <td>输出</td>
      <td><ul><li>只训练场景输出，训练期间动量更新后的方差。</li><li>数据类型、shape与输入running_var保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_mean</td>
      <td>输出</td>
      <td><ul><li>只训练场景输出，保存的x均值，对应公式中的`E(x)`。</li><li>1D张量。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_rstd</td>
      <td>输出</td>
      <td><ul><li>只训练场景输出，保存的x方差或者x标准差倒数，分别对应公式中的`Var(x)`、(Var(x) + ε)开平方的倒数。</li><li>1D张量。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>


  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 输入x和输出y的数据格式不支持NHWC、NDHWC。
    - 输出参数`save_rstd`保存的是x方差。
  
  - <term>Atlas 训练系列产品</term>：
    - 数据类型：所有的输入和输出不支持BFLOAT16。
    - 数据格式：输入x和输出y不支持NHWC、NDHWC。
    - 输出参数`save_rstd`保存的是x方差。
  
  - <term>Atlas 推理系列产品</term>：
    - 数据类型：所有的输入和输出不支持BFLOAT16。
    - 数据格式：输入x和输出y不支持NHWC、NDHWC。
    - 输出参数`save_rstd`保存的是x方差。
  
  - <term>Ascend 950PR/Ascend 950DT</term>：
  
    输出参数`save_rstd`保存的是x标准差的倒数。
  
## 约束说明

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持训练场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_batch_norm_v3](examples/test_aclnn_batch_norm_v3.cpp) | 通过[aclnnBatchNorm](docs/aclnnBatchNorm.md)接口方式调用BatchNormV3算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/batch_norm_v3_proto.h)构图方式调用BatchNormV3算子。         |