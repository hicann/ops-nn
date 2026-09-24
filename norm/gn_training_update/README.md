# GNTrainingUpdate

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：GroupNorm训练前向更新融合算子。消费配套算子GNTrainingReduce输出的组内`sum` / `square_sum`，现算每组均值与（有偏）方差，对`x`做归一化与可选仿射，并旁路输出`batch_mean` / `batch_variance`供反向GroupNormGrad复用。

- 计算公式（`M = (C/num_groups)·H·W`为每组元素个数）：

  $$
  mean = \frac{sum}{M},\quad variance = \frac{square\_sum}{M} - mean^2
  $$

  $$
  y = \frac{x - mean}{\sqrt{variance + epsilon}} \times scale + offset
  $$

  `scale` / `offset`同时缺省时退化为纯归一化；`mean` / `variance`为官方IR保留输入，不参与计算（统计量一律由`sum` / `square_sum`现算）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 910px"><colgroup>
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
    <tr><td>x</td><td>输入</td><td>待归一化特征图，4D [N,C,H,W]（NCHW）或 [N,H,W,C]（NHWC）。</td><td>FLOAT16、FLOAT</td><td>ND、NCHW、NHWC</td></tr>
    <tr><td>sum</td><td>输入</td><td>GNTrainingReduce输出的组内元素和，5D，NCHW时 [N,G,1,1,1]、NHWC时 [N,1,1,G,1]。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>square_sum</td><td>输入</td><td>组内平方和，shape同sum。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>scale</td><td>可选输入</td><td>仿射 γ，5D，NCHW时 [1,G,1,1,1]、NHWC时 [1,1,1,G,1]；与offset同时缺失时退化为纯归一化。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>offset</td><td>可选输入</td><td>仿射 β，shape同scale。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>mean</td><td>可选输入</td><td>官方IR保留位（shape同sum），不参与计算。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>variance</td><td>可选输入</td><td>官方IR保留位（shape同sum），不参与计算。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>num_groups</td><td>可选属性</td><td>组数G，默认值为2，必须整除C。</td><td>INT</td><td>-</td></tr>
    <tr><td>epsilon</td><td>可选属性</td><td>加在variance上防除零的小值，默认值为0.0001。</td><td>FLOAT</td><td>-</td></tr>
    <tr><td>y</td><td>输出</td><td>归一化（+可选仿射）结果，shape和数据类型均与x相同。</td><td>FLOAT16、FLOAT</td><td>ND、NCHW、NHWC</td></tr>
    <tr><td>batch_mean</td><td>输出</td><td>现算组内均值（sum/M），shape同sum。</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>batch_variance</td><td>输出</td><td>现算组内有偏方差（square_sum/M − mean²，未加epsilon），shape同sum。</td><td>FLOAT</td><td>ND</td></tr>
  </tbody>
</table>

## 约束说明

- `G == num_groups`且`C % num_groups == 0`（Host硬校验）。
- 除`x` / `y`外的张量仅支持ND；`x` / `y`为ND时布局由统计量组维位置推断，显式NCHW/NHWC标签与统计量组维位置冲突时拒绝。
- `N=0`空batch合法：三个输出均为空且不下发Kernel；C / H / W / G为0视为非法。
- 动态Shape：支持未知维（-1），未知维在InferShape透传、Kernel编译按静态统一处理（DynamicCompileStatic）；动态Rank：接口开启DynamicRankSupport，但当x或sum任一未知rank时静态可判的形状校验一律延后到运行时，非法输入的拦截时点随之延后。
- 全程float32主链计算（`x`为float16时输出按float16舍入）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_gn_training_update.cpp](examples/arch35/test_geir_gn_training_update.cpp) | 通过[GNTrainingUpdate IR](op_graph/gn_training_update_proto.h)构图调用。 |

样例的构建与运行依赖CANN开发环境与自定义算子包安装（详见仓库根README的环境准备章节）：

1. 完成CANN Toolkit安装并`source`其`set_env`环境脚本；
2. 在本仓执行`bash build.sh --pkg --soc=ascend950 --ops=gn_training_update`构建自定义算子包并安装（`--install-path`指定安装路径），`source`安装后包内`bin/set_env.bash`；
3. 构建并运行样例：`bash build.sh --run_example gn_training_update graph cust --soc=ascend950`（cust表示使用自定义算子包）；
4. 预期输出：三个用例（NCHW ×2 + NHWC ×1）全部打印`PASS`，并打印y/batch_mean/batch_variance的实际数值。
