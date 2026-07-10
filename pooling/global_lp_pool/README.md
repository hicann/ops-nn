# GlobalLpPool

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：对输入张量的空间维度进行全局Lp池化计算，公式为`y = (sum(|x|^p))^(1/p)`，保持空间维度为1（keepdims=True）。支持4D(NCHW)和5D(NCD0D1D2)输入。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>待进行global_lp_pool计算的输入张量，4D(NCHW)或5D(NCD0D1D2)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>属性</td>
      <td>Lp范数的指数p。默认值为2.0（即L2范数）。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>全局Lp池化结果。空间维度归约为1，与输入x同dtype。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量维度仅支持4D(NCHW)或5D(NCD0D1D2)。
- p的取值范围建议为正浮点数（p>0）。p=0时行为未定义，不建议使用。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_geir_global_lp_pool](examples/arch35/test_geir_global_lp_pool.cpp) | 通过[算子IR](./op_graph/global_lp_pool_proto.h)构图方式调用GlobalLpPool算子。 |
