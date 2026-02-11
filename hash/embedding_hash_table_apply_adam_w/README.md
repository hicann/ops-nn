# EmbeddingHashTableApplyAdamW

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ✗     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ✗     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ✗     |
| <term>Atlas 推理系列产品</term> |      ✗     |
| <term>Atlas 训练系列产品</term> |      ✗     |

## 功能说明

- 算子功能：查询hash表是否存在key，如果存在，则更新其value，如果不存在，则插入key。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>table_handles</td>
      <td>输入</td>
      <td>输入hash表handle句柄，里面包含了hash表的表头地址等。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>输入</td>
      <td>插入key序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>输入</td>
      <td>AdamW算法中修正的一阶矩阵。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输入</td>
      <td>AdamW算法中修正的二阶矩阵。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1_power</td>
      <td>输入</td>
      <td>前置beta1的次方结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2_power</td>
      <td>输入</td>
      <td>前置beta2的次方结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>输入</td>
      <td>权重衰减系数。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>输入</td>
      <td>用于计算一阶矩阵的系数。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>输入</td>
      <td>用于计算二阶矩阵的系数。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>除0防护，提高数值稳定性。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>输入梯度。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>输入</td>
      <td>AMSGrad算法中用于限制norm梯度最大值。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dims</td>
      <td>输入属性</td>
      <td>hash表桶深度。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bucket_sizes</td>
      <td>输入属性</td>
      <td>hash表桶数量。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>输入属性</td>
      <td>判断是否使用AMSGrad算法。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>输入属性</td>
      <td>判断是否最大化参数。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>m</td>
      <td>输出</td>
      <td>AdamW算法，修正后的一阶矩阵。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输出</td>
      <td>AdamW算法，修正后的二阶矩阵。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1_power</td>
      <td>输出</td>
      <td>更新后的beta1的次方结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2_power</td>
      <td>输出</td>
      <td>更新后的beta2的次方结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>输出</td>
      <td>更新后的梯度参数。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 无 | 无 | 无 |
