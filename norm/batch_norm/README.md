# BatchNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
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
      <ul><li>进行批量归一化的输入张量，对应公式中的`x`。</li><li>shape维度至少是2D。</li></ul>
      </td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>
      <ul><li>进行批量归一化的权重，对应公式中的`γ`。</li><li>一个1D张量，shape与输入x的维度C相同，如果输入x是ND格式，shape与输入x的第2维度相同。</li></ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td><ul><li>进行批量归一化的偏置值，对应公式中的`β`。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>可选输入</td>
      <td><ul><li>训练场景：训练期间动量更新前的均值，为可选输入；推理场景：推理期间使用的均值，为必选输入，对应公式中的`E(x)`。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>可选输入</td>
      <td><ul><li>训练场景：训练期间动量更新前的方差，为可选输入；推理场景：推理期间使用的方差，为必选输入，对应公式中的`Var(x)`。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到方差中的小值以避免除以零，对应公式中的`ε`。</li><li>默认值为1e-4f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td><ul><li>指定输入x的数据格式，支持"NHWC"、"NCHW"、"NDHWC"、"NCDHW"。</li><li>默认值为"NHWC"。</li></ul></td>
      <td>STRING</td>
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
      <td>exponential_avg_factor</td>
      <td>可选属性</td>
      <td><ul><li>训练期间用于更新动量平均值的因子。</li><li>默认值为1.0f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示批量归一化后的输出结果，对应公式中的`y`。</li><li>数据类型、数据格式、shape与输入x保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输出</td>
      <td><ul><li>训练模式：当前批次的均值，推理模式：等于输入mean。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输出</td>
      <td><ul><li>训练模式：当前批次的方差，推理模式：等于输入variance。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_1</td>
      <td>输出</td>
      <td><ul><li>为梯度计算预留。训练模式：等于输出batch_mean，推理模式：等于输入mean。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_2</td>
      <td>输出</td>
      <td><ul><li>为梯度计算预留。训练模式：保存 inv_var：(1/sqrt(epsilon + variance))，用于反向梯度计算中重用，推理模式：等于输入variance。</li><li>一个1D张量，shape与入参scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_3</td>
      <td>输出</td>
      <td><ul><li>一个只包含一个元素的一维张量，仅出于与TensorFlow兼容的目的而存在，数据无意义。</li><li>shape为[1]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_batch_norm](./examples/test_geir_batch_norm.cpp)   | 通过[算子IR](./op_graph/batch_norm_proto.h)构图方式调用BatchNorm算子。 |
