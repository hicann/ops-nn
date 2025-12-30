# ScatterElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |   ×     |
| <term>Atlas 推理系列产品 </term>                             |     ×      |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |
## 功能说明

算子功能：返回输入张量中的唯一元素。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUniqueGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUnique”接口执行**计算。

- `aclnnStatus aclnnUniqueGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, aclTensor* valueOut, aclTensor* inverseOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUnique(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnUniqueGetWorkspaceSize

* **参数说明**：
  - self（aclTensor*，计算输入）：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度不大于8。
    - <term>Atlas 训练系列产品</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  
  - sorted（bool，计算输入）: 表示是否对 valueOut 按升序进行排序。true时排序，false时乱序。
  - returnInverse（bool，计算输入）: 表示是否返回输入数据中各个元素在 valueOut 中的下标。
  - valueOut（aclTensor*，计算输出）: Device侧的aclTensor，第一个输出张量，输入张量中的唯一元素，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  - inverseOut（aclTensor*，计算输出）: 第二个输出张量，当returnInverse为True时有意义，返回self中各元素在valueOut中出现的位置下标，数据类型支持INT64，shape与self保持一致。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 self或valueOut或inverseOut是空指针时。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. self 或valueOut 的数据类型不在支持的范围之内。
                                      2. self为非连续张量。
                                      3. returnInverse为True，且inverseOut与self shape不一致。
                                      4. returnInverse为True，inverseOut的数据类型不为INT64。
  ```

## aclnnUnique

* **参数说明**：
  * workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64\_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnUniqueGetWorkspaceSize获取。
  * executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的Stream。

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnUnique默认确定性实现。

  * <term>Ascend 950PR/Ascend 950DT</term>：
    - 由于去重算法实现差异，当满足下列所有条件时，算子将无视 sorted 入参的值，固定对输出结果进行升序排序：
      - self 输入为 1D
      - self 的数据类型为下列类型：FLOAT、FLOAT16、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16
  * <term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：在输入self包含0的情况下，算子的输出中会包含正0和负0，而非只输出一个0。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。