# LayerNormV3

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：

  对指定层进行均值为0、标准差为1的归一化计算。
  - 归一化：对输入张量的每个样本进行归一化处理，使得每个样本的均值为0，方差为1。
  - 缩放和偏移：在归一化之后，可以通过缩放因子和偏移量进一步调整归一化后的输出，以适应不同的模型需求。

- 计算公式：

  $$
  mean = {E}[x]
  $$

  $$
  rstd = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}} 
  $$

  $$
  y = w*((x - mean) * rstd) + b
  $$

  其中，E[x]表示输入的均值，Var[x]表示输入的方差。

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
      <td>表示进行归一化计算的输入，公式中的输入x。shape为[A1,...,Ai,R1,...,Rj]，其中A1至Ai表示无需norm的维度，R1至Rj表示需norm的维度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示进行归一化计算的权重，公式中的输入w。gamma非空时，数据类型与输入x一致或为FLOAT类型，且当beta存在时gamma与beta的数据类型相同。shape与normalized_shape相等，为[R1,...,Rj]。gamma为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为1的tensor，当beta存在时gamma与beta的数据类型相同，beta不存在时gamma与输入x的数据类型相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示进行归一化计算的偏移量，公式中的输入b。beta非空时，数据类型与输入x一致或为FLOAT类型，且当gamma存在时beta与gamma的数据类型相同。shape与normalized_shape相等，为[R1,...,Rj]。beta为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为0的tensor，当gamma存在时beta与gamma的数据类型相同，gamma不存在时beta与输入x的数据类型相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin_norm_axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行norm计算的维度。值为[R1,...,Rj]，长度小于等于输入x的shape长度，不支持为空。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>begin_params_axis</td>
      <td>可选属性</td>
      <td><ul><li>表示参数的起始维度。在Ascend 950PR/Ascend 950DT AI处理器，begin_params_axis和begin_norm_axis的值是相同的。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定。</li><li>默认值为0.00001f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行归一化计算的结果，公式中的y。shape需要与x的shape相等，为[A1,...,Ai,R1,...,Rj]。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>表示进行归一化后的均值，公式中的mean。与rstd的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>表示进行归一化后的标准差倒数，公式中的rstd。与mean的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_layer_norm_v3](./examples/test_geir_layer_norm_v3.cpp)   | 通过[算子IR](./op_graph/layer_norm_v3_proto.h)构图方式调用LayerNormV3算子。 |
