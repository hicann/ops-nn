# GNTrainingReduce

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：完成GroupNorm训练前向的统计归约。按通道分组数`num_groups`把输入特征图`x`的通道维划分为若干组，在每组$(n,g)$内对$D \cdot H \cdot W$个元素求$\Sigma x$与$\Sigma x^2$，输出两个fp32张量；该算子只产出原始矩，与配套算子GNTrainingUpdate成对使用。
- 计算公式：令$D = C / num\_groups$，$M = D \cdot H \cdot W$为每组归约元素数。输入`x`非fp32时先转换为fp32（fp32路径不转换）。

  - NCHW输入：把$x[N,C,H,W]$重排为$x_g[N,G,D,H,W]$，在轴$(2,3,4)$上归约：

    $$
    sum[n,g] = \sum_{d=0}^{D-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} x_g[n,g,d,h,w]
    $$

    $$
    square\_sum[n,g] = \sum_{d=0}^{D-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} \left(x_g[n,g,d,h,w]\right)^2
    $$

  - NHWC输入：把$x[N,H,W,C]$重排为$x_g[N,H,W,G,D]$，在轴$(1,2,4)$上归约，公式形式同上。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr><td>x</td><td>输入</td><td>GroupNorm输入特征图，即公式中的x。</td><td>FLOAT16、FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>sum</td><td>输出</td><td>每个(n,g)分组的Σx，即公式中的sum。</td><td>FLOAT</td><td>ND</td></tr>
<tr><td>square_sum</td><td>输出</td><td>每个(n,g)分组的Σx²，即公式中的square_sum。</td><td>FLOAT</td><td>ND</td></tr>
<tr><td>num_groups</td><td>可选属性</td><td>通道分组数G，即公式中的num_groups；默认值为2，须与配套算子GNTrainingUpdate保持一致。</td><td>INT64</td><td>-</td></tr>
</tbody>
</table>

## 约束说明

- 通道维`C`必须能被`num_groups`整除；`num_groups`的取值必须不小于1。
- 输出`sum`与`square_sum`的shape恒为rank-5：NCHW输入时，输出的`sum`与`square_sum`shape为[N,num_groups,1,1,1]，NHWC输入时，输出shape为[N,1,1,num_groups,1]。
- 空Tensor合法：N=0时输出为空Tensor；N>0且C=0、H=0或W=0时为空分组，归约为空和，`sum`与`square_sum`均写出0。
- 本算子不支持输入广播。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_gn_training_reduce.cpp](examples/arch35/test_geir_gn_training_reduce.cpp) | 通过[算子IR](op_graph/gn_training_reduce_proto.h)构图方式调用GNTrainingReduce算子。 |
