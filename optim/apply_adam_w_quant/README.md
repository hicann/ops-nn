# aclnnApplyAdamWQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：
  
  对优化器输入的m和v作为索引，取出各自qmap中的值，乘以每个blockSize对应的absmax进行反量化，而后实现adamW优化器功能，更新后的m和v每blockSize中取一个最大值，每blockSize个m和v对应一个absmax，进行一次norm归一化，利用二分法找到对应m和v对应qmap中的索引作为输出，absmax也作为下一轮量化的输入

- 计算公式：
  
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$


  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$


  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$


  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$


  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>var</td>
      <td>输入/输出</td>
      <td>公式中的theta，待计算的权重输入同时也是输出。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>公式中的输入gt。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>输入/输出</td>
      <td>公式中m参数量化前的索引值。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>v</td>
      <td>输入/输出</td>
      <td>公式中v参数量化前的索引值。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qmap_m</td>
      <td>输入</td>
      <td>公式中m参数量化映射表升序排列。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qmap_v</td>
      <td>输入</td>
      <td>公式中v参数量化映射表升序排列。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>absmax_m</td>
      <td>输入/输出</td>
      <td>计算输入为本次反量化阶段将分位数反归一化（乘以absmaxMRef），计算输出为本次量化和下一轮反量化的参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>absmax_v</td>
      <td>输入/输出</td>
      <td>计算输入为本次反量化阶段将分位数反归一化（乘以absmaxVRef），计算输出为本次量化和下一轮反量化的参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>属性</td>
      <td><ul><li>迭代次数，公式中的t。</li><li>取值范围是大于1的正整数。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>属性</td>
      <td><ul><li>学习率，公式中的eta。</li><li>取值范围是0~1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>属性</td>
      <td><ul><li>公式中beta1参数。</li><li>取值范围是0~1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>属性</td>
      <td><ul><li>公式中beta2参数。</li><li>取值范围是0~1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>属性</td>
      <td><ul><li>权重衰减系数，公式中lambda参数。</li><li>取值范围是0~1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td><ul><li>公式中epsilon参数，加在分母中用来防止除0。</li><li>取值范围是1e-8。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gnorm_scale</td>
      <td>属性</td>
      <td><ul><li>对输入参数grad进行缩放的参数。</li><li>取值范围是0~1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td><ul><li>每个block的大小。</li><li>取值范围固定为256。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td><ul><li>保留参数。</li><li>保留参数。</li></ul></td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_adam_w_quant](./examples/test_aclnn_apply_adam_w_quant.cpp) | 通过[aclnnApplyAdamWQuant](./docs/aclnnApplyAdamWQuant.md)接口方式调用ApplyAdamWQuant算子。    |