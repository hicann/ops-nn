# AddLayerNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |


## 功能说明

- 算子功能：LayerNorm是一种归一化方法，可以将网络层输入数据归一化到[0, 1]之间。LayerNormGrad算子是深度学习中用于反向传播阶段的一个关键算子，主要用于计算LayerNorm操作的梯度。AddLayerNormGrad算子是将Add和LayerNormGrad融合起来，减少搬入搬出操作。

- 计算公式：

  - 正向公式：（D为reduce轴大小）

    $$
    x= inputx1 + inputx2
    $$

    $$
    \operatorname{LayerNorm}(x)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*\gamma + \beta
    $$

    $$
    其中\operatorname{E}(x_i)=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x_i)=\frac{1}{D}\sum_{1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - 反向公式：
    $$
    x= inputx1 + inputx2
    $$

    $$
    dxOut = \sum_{j}{inputdy_i * \gamma_j * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}} + dsumOptional
    $$

    $$
    dgammaOut = \sum_{j}{inputdy_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}}
    $$

    $$
    dbetaOut = \sum_{j}{inputdy_i}
    $$

    其中：
    - $\hat{x_j}$：

      $$
      \hat{x_j}=({x_i-\operatorname{E}(x)}) * {rstd}
      $$

    - $rstd$：

      $$
      rstd=\frac {1}{\sqrt{\operatorname{Var}(x)}}
      $$
    
    - $\frac{{\rm d}\hat{x_j}}{{\rm d}x_i}$：
    
      $$
      \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}  x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x_i)}}-\frac{1}{\operatorname{Var}(x_i)}  (x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}
      $$

      其中，当i=j时，$\delta_{ij}$=1；当i!=j时，$\delta_{ij}$=0。
 
    - $\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}$：
    
      $$
      \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
      $$

      其中，D为x中参加均值计算的数量。

    - $\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}$：
      
      $$
      \frac{\rm d \operatorname{Var}(x_i)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var}  (x_i)}}(x_i-\operatorname{E}(x))
      $$

    - 化简后的$dxOut$：

      $$
      dxOut = rstd * ({inputdy_i * \gamma_j} - \frac{1}{D} * (\sum_{j}{inputdy_i * \gamma_j} + \hat      {x_j} * \sum_{j}{inputdy_i * \gamma_j * \hat{x_j}})) + dsumOptional
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
      <td>dy</td>
      <td>输入</td>
      <td>主要的grad输入，对应公式中的`inputdy`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>为正向融合算子的输入x1，对应公式中的`inputx1`。shape、数据类型与`dy`的shape、数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>为正向融合算子的输入x2，对应公式中的`inputx2`。shape、数据类型与`dy`的shape、数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>表示正向输入x1、x2之和的标准差的倒数，对应公式中的`rstd`。shape需要与`dy`满足broadcast关系（前几维的维度和`dy`前几维的维度相同，前几维指`dy`的维度减去`gamma`的维度，表示不需要norm的维度）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示正向输入x1、x2之和的均值，对应公式中的`E(x)`。shape需要与`dy`满足broadcast关系（前几维的维度和`dy`前几维的维度相同，前几维指`dy`的维度减去`gamma`的维度，表示不需要norm的维度）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>可选输入</td>
      <td>表示需要norm的维度，对应公式中的`γ`。shape维度和`dy`后几维的维度相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dsum</td>
      <td>可选输入</td>
      <td>额外的反向梯度累加输入，对应公式中的输入`dsumOptional`。shape、数据类型与`dy`的shape、数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>
        <ul>
          <li>表示输入x1+x2之和的梯度，对应公式中的`dxOut`。shape、数据类型与`dy`的shape、数据类型保持一致。</li>
          <li>数据类型、shape与输入x1的保持一致。</li>
        </ul>
      </td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td>
        <ul>
          <li>表示入参gamma的梯度，对应公式中的`dgammaOut`。shape与输入`gamma`一致。</li>
          <li>数据类型、shape与输入gamma的保持一致。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>输出</td>
      <td>
        <ul>
          <li>表示正向入参beta的反向梯度，对应公式中的`dbetaOut`。shape与输入`gamma`一致。</li>
          <li>数据类型、shape与输入gamma的保持一致。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> ：输入参数`dy`、`x1`、`x2`和输出参数`dsum`、`dx`的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_layer_norm_grad](examples/test_aclnn_add_layer_norm_grad.cpp) | 通过[aclnnAddLayerNormGrad](docs/aclnnAddLayerNormGrad.md)接口方式调用AddLayerNormGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/add_layer_norm_grad_proto.h)构图方式调用AddLayerNormGrad算子。         |