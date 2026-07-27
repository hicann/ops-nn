# Sleep

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |    √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：在当前执行流中插入设备侧可控延时片段。该算子通过SIMT`clock()`忙等待实现精确的延时控制，语义与CUDA`spin_kernel`/`torch.cuda._sleep`一致。

- 计算公式：

  对于给定的时钟周期数cycles，Sleep算子执行以下计算：

  1. 获取当前clock计数器值作为起始时间：

     $$
     start = clock()
     $$

  2. 忙等待直到经过的cycle数达到指定值：

     $$
     while(clock() - start < cycles): spin
     $$

  3. 实际休眠时间（秒）与clock频率相关：

     $$
     t_{sleep} = \frac{cycles}{f_{clock}}
     $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|--------------|------|---------|---------|
| cycles | 输入 | 休眠的时钟周期数，必须为正整数（cycles > 0）。受AICore超时限制，Ascend 950PR/Ascend 950DT主频1.65GHz下最大约1.782e12（约18分钟）。 | INT64 | ND |

## 约束说明

- cycles参数必须为正整数（cycles > 0）。
- cycles以aclIntArray*传入，数组包含1个元素即休眠周期数。
- AICore默认执行超时时间为18分钟。Ascend 950PR/Ascend 950DT主频1.65GHz下，cycles最大值约1.782e12。如需更长时间，可通过`aclrtSetOpExecuteTimeOut`接口修改AICore超时配置。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sleep](./examples/arch35/test_aclnn_sleep.cpp) | 通过[aclnnSleep](./docs/aclnnSleep.md)接口方式调用Sleep算子。 |
