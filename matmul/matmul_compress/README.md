# MatMulCompress

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|×|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|×|

## 功能说明

- **算子功能**：进行矩阵乘计算时，可先通过msModelSlim工具对右矩阵进行无损压缩，减少内存占用，然后通过本接口完成无损解压缩、矩阵乘计算。
- **计算公式**：
  
  ```
  x2_unzip = unzip(x2, compressIndex)
  result = x1 @ x2_unzip + bias
  ```
  
  其中x2表示右矩阵经过msModelSlim工具压缩后的一维数据，compressIndex表示压缩算法相关的信息，x2_unzip是接口内部进行无损解压缩后的数据（与原始右矩阵数据一致）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 869px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 343px">
<col style="width: 114px">
<col style="width: 102px">
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
    <td>输入张量</td>
    <td>矩阵乘的左输入，2维张量。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入张量</td>
    <td>压缩后的矩阵乘右输入，1维张量。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>compressIndex</td>
    <td>输入张量</td>
    <td>矩阵乘右输入的压缩索引表，1维张量。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入张量</td>
    <td>偏置项，支持空指针传入。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>计算结果输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- x1和x2_unzip的Reduce维度大小必须相等。
- 所有输入张量不支持非连续的Tensor。

