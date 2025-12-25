# ApplyFusedEmaAdam

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- **算子功能**：实现FusedEmaAdam融合优化器功能，结合了Adam优化器和指数移动平均（EMA）技术。
- **计算公式**：

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$


  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$


  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$


  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$


  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$


  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$


  $$
  denom=\sqrt{v_{next}}+eps
  $$


  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$


  $$
  var_{out}=var-lr*update
  $$


  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
<col style="width: 155px">
<col style="width: 162px">
<col style="width: 380px">
<col style="width: 276px">
<col style="width: 107px">
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
    <td>grad</td>
    <td>输入</td>
    <td>待更新参数对应的梯度，对应公式中的grad。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>varRef</td>
    <td>输入/输出</td>
    <td>待更新参数，对应公式中的var。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mRef</td>
    <td>输入/输出</td>
    <td>待更新参数对应的一阶动量，对应公式中的m。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>vRef</td>
    <td>输入/输出</td>
    <td>待更新参数对应的二阶动量，对应公式中的v。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sRef</td>
    <td>输入/输出</td>
    <td>待更新参数对应的EMA权重，对应公式中的s。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>step</td>
    <td>输入</td>
    <td>优化器当前的更新次数，对应公式中的step。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>属性</td>
    <td>学习率，对应公式中的lr，取值范围是(0,1)，默认为1e-3。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>emaDecay</td>
    <td>属性</td>
    <td>指数移动平均（EMA）的衰减速率，对应公式中的emaDecay，取值范围是(0,1)，默认为0.9999。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>属性</td>
    <td>计算一阶动量的系数，对应公式中的β1，取值范围是(0,1)，默认为0.9。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta2</td>
    <td>属性</td>
    <td>计算二阶动量的系数，对应公式中的β2，取值范围是(0,1)，默认为0.999。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>属性</td>
    <td>加到分母上的项，用于数值稳定性，对应公式中的eps，默认为1e-8。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>属性</td>
    <td>控制应用L2正则化还是权重衰减，对应公式中的mode，1为adamw，0为L2，默认为1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>biasCorrection</td>
    <td>属性</td>
    <td>控制是否进行偏差校正，对应公式中的biasCorrection，true表示进行校正，false表示不做校正，默认为True。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightDecay</td>
    <td>属性</td>
    <td>权重衰减，对应公式中的weightDecay，取值范围是(0,1)，默认为0.0。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入grad、var、m、v、s的数据类型和shape必须保持一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_apply_fused_ema_adam](./examples/test_aclnn_apply_fused_ema_adam.cpp) | 通过[aclnnApplyFusedEmaAdam](docs/aclnnApplyFusedEmaAdam.md)接口方式调用ApplyFusedEmaAdam算子。 |