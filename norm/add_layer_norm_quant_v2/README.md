# AddLayerNormQuantV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：LayerNorm算子是大模型常用的归一化操作。AddLayerNormQuantV2算子将LayerNorm前的Add算子和LayerNorm归一化输出给1个或2个下游的量化算子融合起来，减少搬入搬出操作。LayerNorm下游的量化算子可以是Quantize、AscendQuantV2或DynamicQuant算子，具体的量化算子类型由attr入参div_mode和quant_mode决定。当下游有2个量化算子时，2个量化算子的算子类型、输入输出dtype组合和可选输入的组合需要完全一致。
- 计算公式：

  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$

  - 当quant_mode输入为"static"时，输出out_scales1和out_scales2无实际意义。取决于div_mode的输入，融合的量化算子可能是Quantize或AscendQuantV2：

    - 当div_mode输入为true时，融合的量化算子为Quantize，计算公式如下所示：

      $$
      y1Out = round(y / scales1 + zero_points1)
      $$

      $$
      y2Out = round(y / scales2 + zero_points2), \quad \text{当且仅当scales2存在}
      $$

    - 当div_mode输入为false时，融合的量化算子为AscendQuantV2，计算公式如下所示：

      $$
      y1Out = round(y * scales1 + zero_points1)
      $$

      $$
      y2Out = round(y * scales2 + zero_points2), \quad \text{当且仅当scales2存在}
      $$

  - 当quant_mode输入为"dynamic"时，输入zero_points1和zero_points2无实际意义。融合的量化算子是DynamicQuant，此时div_mode无效：

    - 若scales1和scales2均无输入，则y2Out和scale2Out输出无实际意义，可忽略。计算公式如下所示：

      $$
      out_scales1 = row\_max(abs(y))/127
      $$

      $$
      y1Out = round(y / out_scales1)
      $$

    - 若仅输入scales1，则y2Out和scale2Out输出无实际意义，可忽略。计算公式如下所示：

      $$
      tmp1 = y * scales1
      $$

      $$
      out_scales1 = row\_max(abs(tmp1))/127
      $$

      $$
      y1Out = round(tmp1 / out_scales1)
      $$

    - 若scales1和scales2均存在，则y2Out和scale2Out输出有效。计算公式如下所示：

      $$
      tmp1 = y * scales1, \quad tmp2 = y * scales2
      $$

      $$
      out_scales1 = row\_max(abs(tmp1))/127, \quad out_scales2 = row\_max(abs(tmp2))/127
      $$

      $$
      y1Out = round(tmp1 / out_scales1),\quad y2Out = round(tmp2 / out_scales2)
      $$

      其中row\_max代表对每行求最大值。

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
      <td>x1</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的x1。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的x2。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数，对应公式中的gamma。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示层归一化中的beta参数，对应公式中的beta。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示AddLayerNormQuantV2中加法计算的输入，对应公式中的bias。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>可选输入</td>
      <td>表示第一个被融合的量化计算子中的scale/smooth输入，对应公式中的scales1。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示第二个被融合的量化计算子中的scale/smooth输入，对应公式中的scales2。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示第一个被融合的量化计算子中的zeroPoints输入，对应公式中的zero_points1。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示第二个被融合的量化计算子中的zeroPoints输入，对应公式中的zero_points2。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>可选属性</td>
      <td><ul><li>用于确定融合算子融合的是静态还是动态量化算子，对应公式中的quant_mode。取值可以是"static"或"dynamic"。</li><li>默认值为"dynamic"。</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，对应公式中的epsilon。</li><li>默认值为1e-5。</li><li>应当大于0；传入小于等于0时算子照常执行（方差为0时可能产生inf/NaN），仅打印告警。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x=x1+x2+bias的输出。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示静态量化处理scale的方法是乘法或除法，对应公式中的div_mode。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的y1Out。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的y2Out。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1+x2+bias，对应公式中的x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layernormRes</td>
      <td>输出</td>
      <td>表示layernorm的结果输出，对应公式中的y。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales1</td>
      <td>输出</td>
      <td>表示通过scales1计算的动态量化缩放结果，对应公式中的out_scales1。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales2</td>
      <td>输出</td>
      <td>表示通过scales2计算的动态量化缩放结果，对应公式中的out_scales2。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持产品约束：

  - <term>Ascend 950PR&950DT系列产品</term>：
    - 支持`quant_mode`取值"static"或"dynamic"；`quant_mode="dynamic"`时`gamma`支持2-8维。
    - 支持perChannel与perTensor两种量化模式：perChannel时`scales1`、`scales2`、`zero_points1`、`zero_points2`的shape与`gamma`一致；perTensor时上述输入的shape为`[1]`，`gamma`为`[1, x1的最后一维]`。
    - `scales1`、`scales2`、`zero_points1`、`zero_points2`的数据类型可为与`x1`一致，或为FLOAT32（`x1`为FLOAT16/BFLOAT16时）。

  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
    - 当前场景仅支持`quant_mode`、`additional_output`、`div_mode`分别取值为`static`、`false`、`false`。
    - 当前场景仅支持`scales2`、`zero_points2`为空。
    - 当前场景仅支持perTensor量化模式，即`scales1`、`zero_points1`的shape为`[1]`，`gamma`、`beta`的shape为`[1, x1的最后一维]`。
    - 当前场景仅支持`scales1`、`zero_points1`数据类型与`x1`保持一致。

  - <term>Atlas推理系列产品</term>：
    - 当前场景仅支持`quant_mode`、`additional_output`、`div_mode`分别取值为`static`、`false`、`false`。
    - 当前场景仅支持`scales2`、`zero_points2`为空。
    - 当前场景仅支持perTensor量化模式，即`scales1`、`zero_points1`的shape为`[1]`，`gamma`、`beta`的shape为`[1, x1的最后一维]`。
    - 当前场景仅支持`scales1`、`zero_points1`数据类型与`x1`保持一致。
    - 当前场景不支持BFLOAT16类型。
    - 入参`x1`、`x2`的norm轴长度必须大于等于32 Byte。

- 形状约束：

  - `beta`的shape需与`gamma`一致。
  - `bias`存在时，其shape需与`x1`或`gamma`一致。
  - `gamma`的维度数不能超过`x1`的维度数。perTensor模式时，`gamma`的shape为`[1, x1的最后一维]`；其他情况时，`gamma`的shape需等于`x1`的后`gammaDimNum`维所构成的shape。
  - `quant_mode="dynamic"`时，`x1`的维数需不小于2，`out_scales1`、`out_scales2`为`x1`去掉`gamma`所占尾维后的前缀。

- 功能维度：

  * 可选输入（scales1、scales2、zero_points1、zero_points2）支持的可选输入组合如下所示：

    | scales1 | scales2 | zero_points1 | zero_points2 | quant_mode | 是否合法 |
    | --------------- | --------------- | -------------------- | -------------------- | ---------- | :------: |
    | T               | T               | T                    | T                    | "static"   |    T     |
    | T               | T               | T                    | F                    | "static"   |    F     |
    | T               | T               | F                    | T                    | "static"   |    F     |
    | T               | T               | F                    | F                    | "static"   |    T     |
    | T               | F               | T                    | T                    | "static"   |    F     |
    | T               | F               | T                    | F                    | "static"   |    T     |
    | T               | F               | F                    | T                    | "static"   |    F     |
    | T               | F               | F                    | F                    | "static"   |    T     |
    | F               | X               | X                    | X                    | "static"   |    F     |
    | T               | T               | F                    | F                    | "dynamic"  |    T     |
    | T               | F               | F                    | F                    | "dynamic"  |    T     |
    | F               | T               | F                    | F                    | "dynamic"  |    F     |
    | F               | F               | F                    | F                    | "dynamic"  |    T     |
    | X               | X               | T                    | X                    | "dynamic"  |    F     |
    | X               | X               | X                    | T                    | "dynamic"  |    F     |

    其中：`T`代表可选输入存在或组合方式合法；`F`代表可选输入不存在或组合方式不合法；`X`代表任意情况均可。

  * 数据类型支持说明：

    - 当`quant_mode`为"static"时：

      | x1 | x2 | gamma | beta | bias | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x | layernorm_res | out_scales1 | out_scales2 |
      | -- | -- | ----- | ---- | ---- | ------- | ------- | ------------ | ------------ | -- | -- | - | ------------- | ----------- | ----------- |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |
      | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | FLOAT32  | FLOAT32  | FLOAT32 | FLOAT32 |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |

    - 当`quant_mode`为"dynamic"时：

      | x1 | x2 | gamma | beta | bias | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x | layernorm_res | out_scales1 | out_scales2 |
      | -- | -- | ----- | ---- | ---- | ------- | ------- | ------------ | ------------ | -- | -- | - | ------------- | ----------- | ----------- |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_layer_norm_quant_v2](examples/test_aclnn_add_layer_norm_quant_v2.cpp) | 通过[aclnnAddLayerNormQuantV2](docs/aclnnAddLayerNormQuantV2.md)接口方式调用AddLayerNormQuantV2算子。 |
