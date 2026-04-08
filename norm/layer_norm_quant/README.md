# LayerNormQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：LayerNorm算子是大模型常用的归一化操作。LayerNormQuant算子将LayerNorm归一化输出和下游的量化算子融合起来，减少搬入搬出操作。
- 计算公式：

  * LayerNorm操作：
  
    $$
    y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
    $$
    
    $$
    E(x) = {\frac{1}{n} \sum_{i=1}^{n} x_i }
    $$
    
    $$
    Var(x) = {\frac{1}{n} \sum_{i=1}^{n} (x_i-E(x))^2 }
    $$
  
  * quantMode为0时，量化模式为静态量化，输出scaleOut无实际意义：
    
    $$
    res = y / scale + zeroPointsOptional
    $$

  * quantMode为1时，量化模式为动态量化：
    
    $$
    tmp = y * scale
    $$
    
    $$
    scaleOut = row\_max(abs(tmp))/dtypeMax
    $$
    
    $$
    res = round(y / scaleOut )
    $$

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
      <td>x</td>
      <td>输入</td>
      <td>表示层归一化中的x参数。对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数。对应公式中的`gamma`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>对应LayerNorm计算公式中的beta，表示层归一化中的beta参数。对应公式中的`beta`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示被融合的量化计算中的scale输入。对应公式中的`scale`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zeroPointsOptional</td>
      <td>可选输入</td>
      <td>表示被融合的量化计算中的zeroPointsOptional输入。仅在quantMode为0时有效，对应公式中的`zeroPointsOptional`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>可选属性</td>
      <td><ul><li>量化模式，用于确定融合算子融合的时静态还是动态量化算子。对应公式中的`quantMode`。取值为0（静态量化）或1（动态量化）。</li><li>默认值为0。</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示对应LayerNorm中的epsilon，添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>res</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出y被量化后的结果。对应公式中的`res`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>表示动态量化计算的scaleOut结果输出，对应公式中的`scaleOut`，仅在quantMode等于1时有效。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_layer_norm_quant](examples/arch35/test_aclnn_layer_norm_quant.cpp) | 通过[aclnnLayerNormQuant](docs/aclnnLayerNormQuant.md)接口方式调用LayerNormQuant算子。 |
