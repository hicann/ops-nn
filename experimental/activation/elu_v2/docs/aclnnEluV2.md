# EluV2

## 产品支持情况

| 产品                                           | 是否支持 |
|:-------------------------------------------- |:----:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √    |

## 功能说明

计算指数线性单元激活函数Elu的结果。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEluV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEluV2”接口执行计算。

```Cpp
aclnnStatus aclnnEluV2GetWorkspaceSize(
    const aclTensor* self, 
    const aclScalar* alpha, 
    const aclScalar* scale, 
    const aclScalar* inputScale, 
    aclTensor* out,
    uint64_t* workspaceSize, 
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnEluV2(
    void* workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream stream);
```

## **参数说明**

### aclnnEluV2GetWorkspaceSize

| 参数名           | 输入/输出 | 描述                         | 使用说明                 | 类型              | 数据格式    | 维度      | 非连续Tensor |
| ------------- | ----- | -------------------------- | -------------------- | --------------- | ------- | ------- | --------- |
| self          | 输入    | 输入数据张量                     | 数据类型支持fp16，fp32，bf16 | aclTensor*      | ND      | 1-8     | √         |
| alpha         | 输入    | 输入参数                       | 数据类型fp32，，默认值为1.0    | aclScalar*      |         |         |           |
| scale         | 输入    | 输入参数                       | 数据类型fp32，，默认值为1.0    | aclScalar*      |         |         |           |
| inputScale    | 输入    | 输入参数                       | 数据类型fp32，，默认值为1.0    | aclScalar*      |         |         |           |
| out           | 输出    | 输出数据张量                     | 数据类型同输入self          | aclTensor*      | 同输入self | 同输入self | √         |
| workspaceSize | 输出    | 返回需要在Device侧申请的workspace大小 |                      | uint64_t*       |         |         |           |
| executor      | 输出    | 返回op执行器，包含了算子计算流程          |                      | aclOpExecutor** |         |         |           |

### aclnnEluV2

| 参数名           | 输入/输出 | 描述                                                                 |
| ------------- | ----- | ------------------------------------------------------------------ |
| workspace     | 输入    | 在Device侧申请的workspace内存地址                                           |
| workspaceSize | 输入    | 在Device侧申请的workspace大小，由第一段接口aclnnEluV2GetWorkspaceSize获取 |
| executor      | 输入    | op执行器，包含了算子计算流程，由第一段接口 aclnnEluV2GetWorkspaceSize 获取  |
| stream        | 输入    | 指定执行任务的Stream流                                                     |

## 返回值

| 返回值                             | 说明                                                        |
| ------------------------------- | --------------------------------------------------------- |
| ACLNN_SUCCESS (0)               | 成功                                                        |
| ACLNN_ERR_PARAM_NULLPTR         | 输入参数中存在空指针                                                |
| ACLNN_ERR_PARAM_INVALID         | 输入参数不合法（dtype 不匹配、shape 不一致、不支持的 dtype、不支持的 format、维度超限等） |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 内部创建执行器失败                                                 |
| ACLNN_ERR_INNER_NULLPTR         | 内部操作返回空指针                                                 |

## 约束说明

无

## 调用示例

示例代码参考[examples/test_aclnn_elu_v2.cpp](../examples/test_aclnn_elu_v2.cpp)，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

 
