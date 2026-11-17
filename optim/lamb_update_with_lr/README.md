# LambUpdateWithLr

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | × |
|  <term>Kirin 9030处理器系列产品</term> | × |

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
      <td>支持空Tensor。公式中的input_greater1（权重范数），shape支持0-8维，与input_sub共同决定输出shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_greater_realdiv</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_greater_realdiv（信任比分子），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_realdiv（信任比分母），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul0</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul0（学习率），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul1</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul1（update），按input_sub的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_sub</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_sub（参数），与input_greater1的broadcast结果决定输出y的shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_y</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的greater_y（阈值），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_e</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的select_e（回退值），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>minimum_y</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的minimum_y（裁剪上界），shape支持0-8维；融合场景下按input_greater1的shape使用，不参与输出shape广播。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y（更新后的参数），shape取input_greater1与input_sub的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输出shape只由input_greater1与input_sub的broadcast结果决定；其余输入按融合图约定与对应主输入同形使用，不参与输出shape广播。
- 所有输入及输出的维度数为0~8。当input_greater1与input_sub均为0维时，输出的shape为(1,)。
- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_update_with_lr](./examples/test_geir_lamb_update_with_lr.cpp) | 通过算子IR构图方式调用LambUpdateWithLr算子。 |
