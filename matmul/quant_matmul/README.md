# QuantMatmul


##  产品支持情况

| 产品  | 是否支持 |
| ---- | :----: |
| Ascend 950PR/Ascend 950DT               |    ×     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品               |    ×     |
| Atlas 推理系列产品                        |    ×     |
| Atlas 训练系列产品                        |    ×     |

## 功能说明

- 算子功能：完成量化的矩阵乘计算。
- 计算公式：

$$
out = (x1@x2 + bias) * deqScale
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
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>输入</td>
      <td>表示量化参数，公式中的输入deqScale。</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>adjX1</td>
      <td>输入</td>
      <td>表示x1的输入shape是否包含transpose。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX2</td>
      <td>输入</td>
      <td>表示x2的输入shape是否包含transpose。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quant_matmul_v2](examples/test_aclnn_quant_matmul_v2.cpp) | 通过<br>[aclnnQuantMatmul](docs/aclnnQuantMatmul.md)<br>[aclnnQuantMatmulV2](docs/aclnnQuantMatmulV2.md)<br>等方式调用QuantMatmul算子。 |
