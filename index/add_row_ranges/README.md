# AddRowRanges

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：对矩阵x的每一行r，将矩阵src中由indices指定的行范围`[start, end)`进行按列求和，并累加到x的对应行上。该算子源自Kaldi语音识别框架中的`AddRowRanges`操作，主要用于语音处理中的特征矩阵行范围累加。

- 计算公式：

$$
x_{out}(r, c) = x(r, c) + \sum_{j=indices[r,0]}^{indices[r,1]-1} src(j, c)
$$

特殊处理：
- 当`indices[r, 0] >= indices[r, 1]`时，该行为空范围，跳过累加，`x_out(r, c) = x(r, c)`
- 当`indices[r, 0] == -1`或`indices[r, 1] == -1`时，该行为空范围，跳过累加，`x_out(r, c) = x(r, c)`

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>输入矩阵，2D张量，shape为(M, N)，M为行数，N为列数。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>输入</td>
      <td>源矩阵，2D张量，shape为(K, N)，K为行数。src的列数必须与x的列数相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>行范围索引，2D张量，shape为(M, 2)。indices[r, 0]为起始行，indices[r, 1]为结束行（不包含）。值为-1表示空范围。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>输出矩阵，2D张量，shape和dtype与输入x相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **输入维度约束**：x、src、indices必须为2D张量（ND格式），不支持其他维度。
- **列数约束**：src的列数（src.shape[1]）必须等于x的列数（x.shape[1]）。
- **indices shape约束**：indices的shape必须为(M, 2)，其中M等于x的行数（x.shape[0]）。
- **dtype约束**：x和src必须使用相同的dtype，且为float32或float16。indices必须为int32。
- **空范围处理**：当`indices[r, 0] >= indices[r, 1]`或`indices[r, 0] == -1`或`indices[r, 1] == -1`时，该行为空范围，跳过累加，输出等于输入x对应行。
- **索引越界**：indices值超出[0, K)范围时，按numpy切片语义裁剪：负索引解析为K+value（裁剪到0），正索引裁剪到K。裁剪后若start >= end则视为空范围。
- **空tensor处理**：当x行数为0或列数为0时，直接返回空输出，不报错。
- **确定性**：该算子为确定性计算，每个输出元素由单个线程独立计算，无atomicAdd，累加顺序固定为start到end。
- **精度说明**：float16场景使用float32中间变量累加（Kahan补偿求和），累加完成后转回float16，避免float16累加精度损失。float32场景使用Kahan补偿求和。
- **NaN/Inf传播**：NaN参与累加结果为NaN，Inf参与累加遵循IEEE 754标准（Inf+Inf=Inf, Inf+(-Inf)=NaN）。
- **数据范围**：输出元素总数（M × N）不得超过INT32_MAX（约21亿）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_add_row_ranges.cpp">test_geir_add_row_ranges</a></td>
    <td>通过<a href="./op_graph/add_row_ranges_proto.h">算子IR</a>构图方式调用AddRowRanges算子</td>
  </tr>
</tbody>
</table>
