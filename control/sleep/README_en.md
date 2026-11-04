# Sleep

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |    √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Description: inserts a device-side controllable delay segment into the current execution stream. The operator implements precise delay control through the busy waiting of the SIMT `clock()`, and its semantics are consistent with the CUDA `spin_kernel`/`torch.cuda._sleep`.

- Formula:

  For the given number of clock cycles cycles, the Sleep operator performs the following computation:

  1. Obtain the current clock counter value as the start time:

     $$
     start = clock()
     $$

  2. Busy-wait until the number of elapsed cycles reaches the specified value:

     $$
     while(clock() - start < cycles): spin
     $$

  3. The actual sleep time (in seconds) is related to the clock frequency:

     $$
     t_{sleep} = \frac{cycles}{f_{clock}}
     $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|--------|--------------|------|---------|---------|
| cycles | Input | The number of clock cycles to sleep, which must be a positive integer (cycles > 0). Limited by the AI Core timeout, the maximum value is about 1.782e12 (about 18 minutes) at the main frequency of 1.65GHz of the Ascend 950PR&950DT products. | INT64 | ND |

## Constraints

- The cycles parameter must be a positive integer (cycles > 0).
- cycles is passed in as an aclIntArray*, and the array contains 1 element, that is, the number of sleep cycles.
- The default AI Core execution timeout is 18 minutes. At the main frequency of 1.65GHz of the Ascend 950PR&950DT products, the maximum value of cycles is about 1.782e12. If you need a longer time, modify the AI Core timeout configuration through the `aclrtSetOpExecuteTimeOut` interface.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sleep](./examples/arch35/test_aclnn_sleep.cpp) | Invoke the Sleep operator through the [aclnnSleep](./docs/aclnnSleep_en.md) interface. |
