# WeightQuantBatchMatmul


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。
- 计算公式：

  $$
  result = self @ ANTIQUANT(mat2) + bias
  $$

  公式中的$mat2$为伪量化场景的输入，其反量化公式$ANTIQUANT(mat2)$为

  $$
  ANTIQUANT(mat2) = (mat2 + antiquantOffset) * antiquantScale
  $$

  当需要对输出进行量化处理时，其量化公式为

  $$
  \begin{aligned}
  result &= QUANT(self @ ANTIQUANT(mat2) + bias) \\
  &= (self @ ANTIQUANT(mat2) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当不需要对输出再进行量化操作时，其计算公式为

  $$
  result = self @ ANTIQUANT(mat2) + bias
  $$

## 参数说明

<table class="tg"><thead>
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
      <td>公式中的输入self。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入mat2。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>输入</td>
      <td>反量化参数中的缩放因子，对应公式中的antiquantScale。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>输入</td>
      <td>反量化参数中的偏置因子，对应公式中的antiquantOffset。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>输入</td>
      <td>量化参数中的缩放因子，对应公式中的quantScale。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset</td>
      <td>输入</td>
      <td>量化参数中的偏置因子，对应公式中的quantOffset。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的result。</td>
      <td>FLOAT16,INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_weight_quant_batch_mat_mul](examples/test_aclnn_weight_quant_batch_mat_mul.cpp) | 通过<br>[aclnnWeightQuantBatchMatmul](docs/aclnnWeightQuantBatchMatmul.md)<br> 调用WeightQuantBatchMatmul算子。 |