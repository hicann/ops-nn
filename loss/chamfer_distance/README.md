# ChamferDistance

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：计算两组二维点集之间的倒角距离（Chamfer Distance）。对第一组中的每个点，在第二组中找到距它最近的点，输出该最近距离与最近点的下标；再对称地对第二组做一遍。距离为平方欧氏距离，不开根。

- 计算公式：

  设点集按坐标平面分离存放，$xyz[0]$为全部x坐标、$xyz[1]$为全部y坐标，$B$为batch数、$N$为每个点集的点数。记第$b$个batch中第$i$个点为$p_{b,i}=(x_{b,i}, y_{b,i})$，则：

  $$
  d(b, i, j) = (x^1_{b,i} - x^2_{b,j})^2 + (y^1_{b,i} - y^2_{b,j})^2
  $$

  $$
  \text{dist1}_{b,i} = \min_{j} d(b, i, j), \quad \text{idx1}_{b,i} = \arg\min_{j} d(b, i, j)
  $$

  $$
  \text{dist2}_{b,j} = \min_{i} d(b, i, j), \quad \text{idx2}_{b,j} = \arg\min_{i} d(b, i, j)
  $$

  其中：

  - batch之间相互独立，两组点集的点数$N$必须相同。
  - 存在并列最小值时，取下标最小者。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型        | 数据格式 |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| xyz1   | 输入           | 第一组二维点集，shape为$(2, B, N)$：xyz1[0]为全部x坐标、xyz1[1]为全部y坐标。 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| xyz2   | 输入           | 第二组二维点集，shape与数据类型均与xyz1一致。                 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| dist1  | 输出           | xyz1中每个点到xyz2的最小平方距离，shape为$(B, N)$，数据类型与xyz1一致。 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| dist2  | 输出           | xyz2中每个点到xyz1的最小平方距离，shape为$(B, N)$，数据类型与xyz1一致。 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| idx1   | 输出           | dist1对应的最近点下标，取值范围为$[0, N)$，shape为$(B, N)$。   | INT32           | ND       |
| idx2   | 输出           | dist2对应的最近点下标，取值范围为$[0, N)$，shape为$(B, N)$。   | INT32           | ND       |

## 约束说明

- xyz1的shape固定为3维且首维为2，首维是x、y两个坐标平面，不是点的坐标维度。
- xyz2的shape与数据类型必须与xyz1一致，即两组点集的batch数与点数相同。
- dist1、dist2的数据类型与xyz1一致；idx1、idx2的数据类型固定为INT32。
- 输出的shape由xyz1的第1、2维决定，即$(B, N)$。
- BFLOAT16仅<term>Ascend 950PR/Ascend 950DT</term>支持，其余产品的数据类型支持FLOAT16、FLOAT。
- xyz1的$B$或$N$为0时（两组点集同时为空），dist1、dist2、idx1、idx2输出对应的空Tensor，算子正常返回。
- 坐标取值含inf或nan时，按IEEE规则参与比较与传播，不做拦截。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_chamfer_distance.cpp](examples/test_geir_chamfer_distance.cpp) | 通过[算子IR](./op_graph/chamfer_distance_proto.h)构图方式调用ChamferDistance算子。 |
