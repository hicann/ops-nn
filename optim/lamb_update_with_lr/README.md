# LambUpdateWithLr

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：BERT LAMB优化器图融合算子（信任比权重更新，含裁剪）：计算信任比并用minimum_y/greater_y做上下界裁剪后更新参数input_sub。

- 计算公式：

  $ratio = where(input\_greater1>greater\_y,\ input\_greater\_realdiv/input\_realdiv,\ select\_e)$

  $clip = max(min(ratio,\ minimum\_y),\ greater\_y)$

  $y = input\_sub - clip \times input\_mul0 \times input\_mul1$

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
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input_greater1</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的input_greater1（权重范数），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_greater_realdiv</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的input_greater_realdiv（信任比分子），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的input_realdiv（信任比分母），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul0</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的input_mul0（学习率），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul1</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul1（update），主张量，shape需与input_sub满足broadcast关系，其broadcast结果决定输出y的shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_sub</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_sub（参数），主张量，shape需与input_mul1满足broadcast关系，其broadcast结果决定输出y的shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_y</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的greater_y（阈值），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_e</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的select_e（回退值），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>minimum_y</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的minimum_y（裁剪上界），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y（更新后的参数），shape取input_mul1与input_sub的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_update_with_lr](./examples/test_geir_lamb_update_with_lr.cpp) | 通过算子IR构图方式调用LambUpdateWithLr算子。 |
