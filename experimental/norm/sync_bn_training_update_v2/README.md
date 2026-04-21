# SyncBnTrainingUpdateV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 计算批次 mean/var 并以 momentum 更新 running_mean/var；输出归一化结果。

- 计算公式：
  符号定义：
    - 设分布式训练设备数为 K，第 k 个设备输入张量形状为 [N_k, C, H, W]
    - N_k：设备k的样本数；C：通道数；H/W：空间维度；


$$
N_c = \sum_{k=0}^{K-1} N_k \cdot H \cdot W \quad (\text{通道} \, c \, \text{的全局总元素数})
$$


$$
sum_c = \sum_{k=0}^{K-1} sum_{k,c} \quad (\text{通道} \, c \, \text{的全局元素和})
$$


$$
sum\_sq_c = \sum_{k=0}^{K-1} sum\_sq_{k,c} \quad (\text{通道} \, c \, \text{的全局元素平方和})
$$


$$
\mu_c = \frac{sum_c}{N_c} \quad (\text{通道} \, c \, \text{的全局均值})
$$


$$
\sigma_c^2 = \frac{sum\_sq_c}{N_c} - \mu_c^2 \quad (\text{通道} \, c \, \text{的全局方差，平方和公式简化计算})
$$


$$
\hat{x}_{k,c,h,w} = x_{k,c,h,w} - \mu_c \quad (\text{设备}k \, \text{中通道}c \, \text{位置}(h,w) \, \text{的中心化结果})
$$


$$
\sigma_{\varepsilon, c} = \sqrt{\sigma_c^2 + \varepsilon} \quad (\text{通道}c \, \text{的稳定化标准差，} \varepsilon\approx1e-5)
$$


$$
\tilde{x}_{k,c,h,w} = \frac{\hat{x}_{k,c,h,w}}{\sigma_{\varepsilon, c}} \quad (\text{设备}k \, \text{中通道}c \, \text{位置}(h,w) \, \text{的归一化结果})
$$


$$
\text{running\_mean}_c = \text{momentum} \times \text{running\_mean}_c + (1 - \text{momentum}) \times \mu_c \quad (\text{通道}c \, \text{移动均值更新})
$$


$$
\text{running\_var}_c = \text{momentum} \times \text{running\_var}_c + (1 - \text{momentum}) \times \sigma_c^2 \quad (\text{通道}c \, \text{移动方差更新})
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x1</td>
      <td>输入</td>
      <td>该批次数据，对应公式中n_i。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>全局中均值，对应公式中running_mean。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>全局方差，对应公式中running_var。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>该批次数据归一化后的值。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>更新后的均值和方差，对应公式中new_running_mean。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>输出</td>
      <td>更新后的均值和方差，对应公式中nvar。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 只支持四维张量（N,C,H,W）,按C轴批处理化

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sync_bn_training_update_v2_example](./examples/test_aclnn_sync_bn_training_update_v2.cpp) | 通过aclnnSyncBnTrainingUpdateV2接口方式调用SyncBnTrainingUpdateV2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Shi xiangyang | 个人开发者 | SyncBnTrainingUpdateV2 | 2025/12/26 |SyncBnTrainingUpdateV2算子适配开源仓 |