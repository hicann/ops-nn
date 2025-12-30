# ScatterNd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明
算子功能：拷贝data的数据至out，同时在指定indices处根据updates更新out中的数据。

## 函数原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnScatterNdGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterNd”接口执行计算。

* `aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *data,const aclTensor *indices,const aclTensor *updates, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterNdGetWorkspaceSize
- **参数说明：**
  * data(aclTensor*,计算输入)：Device侧的aclTensor, 数据类型与updates、out一致，shape满足1<=rank(data)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL

  * indices(aclTensor*,计算输入)：Device侧的aclTensor，数据类型支持INT32、INT64。indices.shape[-1] <= rank(data)，且1<=rank(indices)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。仅支持非负索引。indices中的索引数据不支持越界。

  * updates(aclTensor*,计算输入)：Device侧的aclTensor, 数据类型与data、out一致。shape要求rank(updates)=rank(data)+rank(indices)-indices.shape[-1] -1, 且满足1<=rank(updates)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL

  * out(aclTensor*，计算输出)：Device侧的aclTensor，数据类型与data、out一致，shape与data一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL

  * workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。

  * executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1.传入的data、indices、updates、out中有空指针
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. 数据类型不在支持的范围之内;
                                      2. shape不满足要求：1<=rank(data)<=8,  1<=rank(indices)<=8,rank(updates)=rank(data)+rank(indices)- indices.shape[-1] -1
                                      3. shape不满足要求：1<=rank(indices)<=8, indices.shape[-1] <= rank(data)
                                      4. shape不满足要求：1<=rank(updates)<=8, updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]
                                      5. shape不满足要求：data.shape == out.shape
  ```
## aclnnScatterNd
- **参数说明：**
  * workspace(void *, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnScatterNdGetWorkspaceSize获取。
  * executor(aclOpExecutor *, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
无