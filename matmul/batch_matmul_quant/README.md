# BatchMatmulQuant


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Ascend 950PR/Ascend 950DT |    ×     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品 |    ×     |
| Atlas 推理系列产品 |    ×     |
| Atlas 训练系列产品 |    ×     |

## 功能说明

- 算子功能：
实现输入Tensor的dtype是float16, 输出的dtype是int8的矩阵乘计算。

- 计算公式：

  $$
  out = Quant(x1@x2 + bias)
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
      <td>公式中的输入x1。</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>输入</td>
      <td>硬件完成量化计算的量化参数，可以通过 <a href="../../quant/trans_quant_param/docs/aclnnTransQuantParam.md">aclnnTransQuantParam</a> 接口获取。</td>
      <td>UINT64</td>
      <td>NC1HWC0</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>用于描述x1是否转置。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>用于描述x2是否转置。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_batch_matmul_quant](examples/test_aclnn_batch_matmul_quant.cpp) | 通过<br>[aclnnBatchMatmulQuant](docs/aclnnBatchMatmulQuant.md)<br>调用BatchMatmulQuant算子。 |