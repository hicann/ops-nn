# ThresholdGradV2D

## 产品支持情况

| 产品　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否支持 |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR/Ascend 950DT</term>　　　　　　　　　 | √　　　　|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √　　　　|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √　　　　|
| <term>Atlas 200I/500 A2 推理产品</term>　　　　　　　　　| ×　　　　|
| <term>Atlas 推理系列产品</term>　　　　　　　　　　　　　| √　　　　|
| <term>Atlas 训练系列产品</term>　　　　　　　　　　　　　| √　　　　|

## 功能说明

- 算子功能：完成threshold正向操作的反向传播梯度计算。当threshold==0时等价于ReluGrad。

- 计算公式：

  $$
      out_i =
      \begin{cases}
      gradOutput_i, \quad self_i > threshold\\
      0,  \quad self_i \leq threshold
      \end{cases}
  $$

## 参数说明

| 参数名     | 输入/输出 | 描述                                                         | 数据类型                                       | 数据格式 |
| :--------- | :-------- | :----------------------------------------------------------- | :--------------------------------------------- | :------- |
| gradOutput | 输入      | 上游梯度，与self广播兼容。                                  | FLOAT16、FLOAT、BF16、INT32、INT8、UINT8        | ND       |
| self       | 输入      | 正向输入，决定门控掩码。                                      | FLOAT16、FLOAT、BF16、INT32、INT8、UINT8        | ND       |
| threshold  | 属性      | 阈值标量，OPTIONAL，默认1.0。threshold==0时等价ReluGrad。  | FLOAT                                           | -        |
| out        | 输出      | 反向梯度，dtype与self一致，shape为广播后形状。           | FLOAT16、FLOAT、BF16、INT32、INT8、UINT8        | ND       |

## 约束说明

- gradOutput、self、out三者 dtype一致。
- threshold==0走ReluGrad路径（regbase上额外支持INT64）。

## 调用说明

| 调用方式   | 调用样例                                                | 说明                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| aclnn调用 | [test_aclnn_threshold_backward.cpp](examples/test_aclnn_threshold_backward.cpp) | 通过aclnnThresholdBackward接口调用 |
