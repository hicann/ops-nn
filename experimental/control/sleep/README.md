# Sleep

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |    √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：让 NPU 设备休眠指定的时钟周期数。该算子通过忙等待（busy-spin）AI Core 的 SYS_CNT 硬件计数器实现精确的延时控制，语义与 CUDA 的 `spin_kernel` 一致。

- 计算公式：

  对于给定的时钟周期数 cycles，Sleep 算子执行以下计算：

  1. 获取当前 SYS_CNT 计数器值作为起始时间：

     $$
     start = \text{GetSystemCycle}()
     $$

  2. 计算目标计数器值：

     $$
     target = start + cycles
     $$

  3. 忙等待直到计数器达到目标值：

     $$
     \text{while}(\text{GetSystemCycle}() < target): \text{spin}
     $$

  4. 实际休眠时间（秒）与设备频率相关：

     $$
     t_{sleep} = \frac{cycles}{f_{device}}
     $$

     其中 $f_{device}$ 为设备频率（例如 1650 MHz 设备，$f_{device} = 1.65 \times 10^9$ Hz）。

- 典型用途：
  - 性能测试中模拟计算延迟
  - 同步测试和时序控制
  - 基准测试中的延迟注入

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>cycles</td>
      <td>属性</td>
      <td>休眠的时钟周期数，必须为非负整数。当 cycles = 0 时，算子立即返回；当 cycles > 0 时，NPU 将忙等待指定的周期数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- cycles 参数必须为非负整数（cycles >= 0）
- 该算子无输入/输出张量
- 该算子使用忙等待机制，在休眠期间会占用 NPU 计算资源
- 实际休眠精度受设备频率影响，不同设备型号可能存在差异

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sleep](./examples/test_aclnn_sleep.cpp) | 通过[aclnnSleep](./docs/aclnnSleep.md)接口方式调用Sleep算子。 |
| aclnn调用 (Python ctypes) | [test_sleep_aclnn.py](./examples/test_sleep_aclnn.py) | 通过Python ctypes调用ACLNN接口。 |
| PyTorch接口 | [test_sleep_pybind.py](./examples/test_sleep_pybind.py) | 通过PyTorch NPU扩展调用，提供torch.npu.sleep(cycles)接口。 |
