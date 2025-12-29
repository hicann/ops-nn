# aclnnBaddbmm&aclnnInplaceBaddbmm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/batch_mat_mul_v3)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 接口功能：
计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和。
注意：batch1、batch2必须是三维Tensor，两个shape仅在aclnnBaddbmm支持做broadcast,两个shape在aclnnInplaceBaddbmm做broadcast则会被拦截；
self必须要支持和batch1@batch2的结果做broadcast。（broadcast，广播机制，是指较小的shape扩展至较大的shape，使两者shape互相兼容，当前仅支持（1，n）的broadcast，即两个Tensor对应的每一维度必须相同或其中一个为1。）

- 计算公式：

  $$
  out = βself+α(batch1@batch2)
  $$
  注意：如果β为0，则self会被忽略，不参与计算。

- 示例：

  self的shape是[1, M, K], batch1@batch2的shape是[A, M, 1]，计算输出out的shape是[A, M, K]。每一维度的数字需要相同或其中一个为1。此处若self的shape是[2, M, K]，则不满足broadcast条件，报错。

## 函数原型

- aclnnBaddbmm和aclnnInplaceBaddbmm实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnBaddbmm：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceBaddbmm：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBaddbmmGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBaddbmm”接口执行计算。
```cpp
aclnnStatus aclnnBaddbmmGetWorkspaceSize(
  const aclTensor*    self, 
  const aclTensor*    batch1, 
  const aclTensor*    batch2, 
  const aclScalar*    beta, 
  const aclScalar*    alpha, 
  aclTensor*          out, 
  int8_t              cubeMathType, 
  uint64_t*           workspaceSize, 
  aclOpExecutor**     executor)
```
```cpp
aclnnStatus aclnnBaddbmm(
  void*           workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor*  executor, 
  aclrtStream     stream)
```

```cpp
aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(
  const aclTensor*    selfRef, 
  const aclTensor*    batch1, 
  const aclTensor*    batch2, 
  const aclScalar*    beta, 
  const aclScalar*    alpha, 
  int8_t              cubeMathType, 
  uint64_t*           workspaceSize, 
  aclOpExecutor**     executor)
```
```cpp
aclnnStatus aclnnInplaceBaddbmm(
  void*             workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor*    executor, 
  aclrtStream       stream)
```

## aclnnBaddbmmGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td><ul><li>数据类型需要与out保持一致。</li>
      <li>shape需要与batch1@batch2的后两维的shape满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>（注意：只能是self通过broadcast变成和batch1@batch2的后两维一样的shape，不可以batch1@batch2的后两维通过broadcast变成和self一样，举例：self：[2, 3, 5]，batch1@batch2：[1, 1, 1]，会报错。）。</li>
      <li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>输入</td>
      <td>公式中的输入batch1。</td>
      <td><ul><li>数据类型与self、batch2的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>shape需要与batch2满足bmm输入约束关系。</li><li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>输入</td>
      <td>公式中的输入batch2。</td>
      <td><ul><li>数据类型与self、batch1的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>shape需要与batch1满足bmm输入约束关系。</li>。<li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的输入β。</td>
      <td>数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>公式中的输入α。</td>
      <td>数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td><ul><li>数据类型需要与self保持一致。</li><li>shape要求与batch1@batch2的后两维shape保持一致。</li><li>format需要与self、batch1@batch2保持一致。</li>
      </ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>用于指定Cube单元的计算逻辑。</td>
      <td>如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：<ul>
        <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
        <li>1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。</li>
        <li>2：USE_FP16，支持将输入降精度至FLOAT16计算。</li>
        <li>3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - 不支持BFLOAT16数据类型；
    - 当输入数据类型为FLOAT32时不支持cubeMathType=0；
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
    - 不支持cubeMathType=3。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - cubeMathType=0，当输入数据类型为FLOAT16或BFLOAT16时，矩阵乘计算会使用FLOAT16 / BFLOAT16输入、FLOAT32输出的方式传递，当输入为其他数据类型时不做处理；
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
    - cubeMathType=2，当输入数据类型是FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
    - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理。
  - <term>昇腾910_95 AI处理器</term>：
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
    - cubeMathType=2，当输入数据类型是FLOAT32时，会转换为FLOAT16计算；当输入为其他数据类型时不做处理；
    - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、batch1或batch2是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>self、batch1、batch2或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self、batch1、batch2或out的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self、batch1、batch2或out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self、batch1、batch2或out的数据格式不一致。</td>
    </tr>
    <tr>
      <td>self不能与batch1@batch2做broadcast操作。</td>
    </tr>
    <tr>
      <td>self与batch1@batch2必须同时为空tensor或同时不为空tensor。</td>
    </tr>
    <tr>
      <td>self、batch1、batch2或out的数据类型不满足推导规则。</td>
    </tr>
  </tbody>
  </table>

## aclnnBaddbmm

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBaddbmmGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceBaddbmmGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>输入|输出</td>
      <td>输入输出tensor，即公式中的输入self与out。</td>
      <td><ul>
      <li>数据类型与batch1@batch2的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。<li>shape需要与batch1@batch2的后两维的shape一致。</li>
      <li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>输入</td>
      <td>公式中的输入batch1。</td>
      <td><ul><li>数据类型与self、batch2的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>shape需要与batch2满足bmm输入约束关系。</li><li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>输入</td>
      <td>公式中的输入batch2。</td>
      <td><ul><li>数据类型与self、batch1的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>shape需要与batch1满足bmm输入约束关系。</li>。<li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的输入β。</td>
      <td>数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>公式中的输入α。</td>
      <td>数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>用于指定Cube单元的计算逻辑。</td>
      <td>如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：<ul>
        <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
        <li>1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。</li>
        <li>2：USE_FP16，支持将输入降精度至FLOAT16计算。</li>
        <li>3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - 不支持BFLOAT16数据类型；
    - 当输入数据类型为FLOAT32时不支持cubeMathType=0；
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
    - 不支持cubeMathType=3。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
    - cubeMathType=2，当输入数据类型是FLOAT32，会转换为FLOAT16计算；当输入为其他数据类型时不做处理；
    - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的selfRef、batch1或batch2是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>selfRef、batch1或batch2的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>selfRef、batch1或batch2的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>selfRef、batch1或batch2的数据格式不一致。</td>
    </tr>
    <tr>
      <td>selfRef不能与batch1@batch2做broadcast操作。</td>
    </tr>
    <tr>
      <td>batch1和batch2的第一维度不相等。</td>
    </tr>
    <tr>
      <td>batch1和batch2的维度不是三维。</td>
    </tr>
    <tr>
      <td>batch1的最后一维和batch2的倒数第二维不相等。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceBaddbmm

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceBaddbmmGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnBaddbmm&aclnnInplaceBaddbmm默认确定性实现。
  - <term>昇腾910_95 AI处理器</term>: aclnnBaddbmm&aclnnInplaceBaddbmm默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：Cube单元不支持FLOAT32计算。当输入为FLOAT32，可通过设置cubeMathType=1（ALLOW_FP32_DOWN_PRECISION）来允许接口内部cast到FLOAT16进行计算。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_baddbmm.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> batch1Shape = {1, 2, 3};
  std::vector<int64_t> batch2Shape = {1, 3, 4};
  std::vector<int64_t> outShape = {1, 2, 4};
  void* selfDeviceAddr = nullptr;
  void* batch1DeviceAddr = nullptr;
  void* batch2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* batch1 = nullptr;
  aclTensor* batch2 = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> batch1HostData = {1, 1, 1, 2, 2, 2};
  std::vector<float> batch2HostData = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<float> outHostData(8, 0);
  int8_t cubeMathType = 1;
  float alphaValue = 1.2f;
  float betaValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建batch1 aclTensor
  ret = CreateAclTensor(batch1HostData, batch1Shape, &batch1DeviceAddr, aclDataType::ACL_FLOAT, &batch1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建batch2 aclTensor
  ret = CreateAclTensor(batch2HostData, batch2Shape, &batch2DeviceAddr, aclDataType::ACL_FLOAT, &batch2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建beta aclScalar
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnBaddbmm接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnBaddbmm第一段接口
  ret = aclnnBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBaddbmm第二段接口
  ret = aclnnBaddbmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceBaddbmm接口调用示例
  // 3. 调用CANN算子库API
  LOG_PRINT("\ntest aclnnInplaceBaddbmm\n");
  // 调用aclnnInplaceBaddbmm第一段接口
  ret = aclnnInplaceBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceBaddbmm第二段接口
  ret = aclnnInplaceBaddbmm(inplaceWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(batch1);
  aclDestroyTensor(batch2);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(batch1DeviceAddr);
  aclrtFree(batch2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtFree(inplaceWorkspaceAddr);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
