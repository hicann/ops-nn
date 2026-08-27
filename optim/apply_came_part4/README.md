# ApplyCamePart4

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>   |     √    |
|  <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  CAME优化器第4段（参数更新段）。输入待更新参数param_in、一阶动量m（形状(N,M)）与置信因子
  r_in(N)、c_in(M)，按CAME更新规则回写param_out/r_out/c_out。sum_r（全N行r_in的归约和）与
  global_shape（全局N,M）为可选输入：分布式场景由前序算子/全局形状传入；单机场景缺省，kernel
  内部完成归约并取本地n/m。

- 计算公式（n = len(r_in),m = len(c_in);N/M取global_shape（给定）否则n/m）：

  $$
  sum\_r=\sum_{i} r\_in_i \quad (\text{sum\_r缺省时kernel内归约})
  $$

  $$
  r\_out=\beta_3 \cdot r\_in+\frac{(1-\beta_3)}{M} \cdot sum\_u\_r
  $$

  $$
  c\_out=\beta_3 \cdot c\_in+\frac{(1-\beta_3)}{N} \cdot sum\_u\_c
  $$

  $$
  denom=\beta_3 \cdot \frac{sum\_r}{N}+(1-\beta_3)\cdot\frac{sum\_u\_rc}{M \cdot N}
  $$

  $$
  param\_out=(1-lr\cdot weight\_decay)\cdot param\_in-\frac{lr \cdot m}{\sqrt{r\_out \otimes c\_out / denom}}
  $$

  其中 $r\_out \otimes c\_out$ 为(N,1)×(1,M)外积。fp16/bf16路径：输入cast到fp32计算，
  输出按RNE(CAST_RINT)round回低精度；param更新以round后的r_out/c_out为输入。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 500px">
  <col style="width: 212px">
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
      <td>param_in</td>
      <td>输入</td>
      <td>待更新参数,形状(N,M)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>输入</td>
      <td>一阶动量,形状(N,M),与param_in一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>r_in</td>
      <td>输入</td>
      <td>行置信因子,形状(N,)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_in</td>
      <td>输入</td>
      <td>列置信因子,形状(M,)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>输入</td>
      <td>权重衰减系数,标量(1,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率,标量(1,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta3</td>
      <td>输入</td>
      <td>置信因子衰减系数,标量(1,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_r</td>
      <td>输入</td>
      <td>r方向更新量归约,形状(N,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_c</td>
      <td>输入</td>
      <td>c方向更新量归约,形状(M,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_rc</td>
      <td>输入</td>
      <td>全局更新量归约,标量(1,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_r</td>
      <td>输入(可选)</td>
      <td>全N行r_in的归约和,标量(1,);缺省时kernel内归约。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_shape</td>
      <td>输入(可选)</td>
      <td>全局(N,M),形状(2,);缺省取本地n/m。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>param_out</td>
      <td>输出</td>
      <td>更新后参数,形状(N,M)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>r_out</td>
      <td>输出</td>
      <td>更新后行置信因子,形状(N,)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_out</td>
      <td>输出</td>
      <td>更新后列置信因子,形状(M,)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- param_in/m必须为2D,r_in/c_in必须为1D;r_in长度 = param_in第0维,c_in长度 = param_in第1维(tiling校验,不一致返回GRAPH_FAILED)。
- 无属性(attr)。
- sqrt域:s = r_out⊙c_out/denom逐元素需 ≥ 0,denom ≠ 0;否则按IEEE语义产生inf/nan传播(与torch公式语义一致,非算子错误)。
- 空tensor(N=0或M=0)不做守卫:不崩溃,但非空维输出不写入(未定义),行为对齐A2。
- 支持关系:Atlas A2系列(ascend910b/ascend910_93)由canndev仓内置ascendc实现支持(ascendc_config.json compute_units=[ascend910b,ascend910_93],kernel binary构建配置binary_json_cfg.ini已收录;TBE旧式aic-ops-info.ini未收录,以动态编译形式提供);Ascend 950PR/950DT由本仓vendor包支持(arch35)。两实现互不影响,vendor包只许安装到Ascend 950环境——其proto/tiling注册无soc维度,误入A2环境会遮蔽canndev内置实现并因tiling数据结构不匹配导致算子不可用。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE图模式调用 | [test_geir_apply_came_part4](./examples/test_geir_apply_came_part4.cpp) | 通过[算子IR](op_graph/apply_came_part4_proto.h)构图方式调用ApplyCamePart4算子。    |
