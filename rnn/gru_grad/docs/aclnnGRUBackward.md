# aclnnGRUBackward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品 </term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

* 算子功能：GRU的反向传播，计算正向输入input、权重params、初始状态hx的梯度。
* 正向计算公式：
  * 重置门 $r_t= \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{t-1} + b_{hr})$
  * 更新门 $z_t = \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{t-1} + b_{hz})$
  * 候选隐藏状态 $n_t = \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn}))$
  * 隐藏状态 $h_t = (1-z) \odot n_t + z_t \odot h_{t-1}$
* 反向计算公式：
  * $\text{上游总梯度：} \quad dh_t = dy_t + dh_{next} \quad $($dy_t$为上层梯度，$dh_{next}$为t+1时刻传回的梯度)
  * $\text{更新门梯度：} \quad dz_t = dh_t * (h_{t-1} - \tilde{h}_t) * z_t * (1 - z_t)$
  * $\text{候选态梯度：} \quad dh_{\tilde{h}t} = dh_t * (1 - z_t) * (1 - \tilde{h}_t^2)$
  * $\text{重置门梯度：} \quad dr_t = dh_{\tilde{h}t} * lin_{hh}[2*hidden\_size:3*hidden\_size] * r_t * (1 - r_t)$
  * $\text{线性变换梯度拆分：}$
    $\quad dlin_{ih} = [dz_t; dr_t; dh_{\tilde{h}t}], \quad dlin_{hh} = [dz_t; dr_t; dh_{\tilde{h}t} * r_t]$
  * $\text{输入梯度（传给下层）：} \quad dx_t = W_{ih}^T @ dlin_{ih}$
  * $\text{前一时刻隐藏态梯度（传给t-1）：} \quad dh_{prev} = W_{hh}^T @ dlin_{hh} + dh_t * z_t$
  * $\text{权重/偏置梯度累加：}$
    $\quad dW_{ih} += dlin_{ih} @ x_t^T, \quad dW_{hh} += dlin_{hh} @ h_{t-1}^T$
    $\quad db_{ih} += dlin_{ih}.sum(dim=1), \quad db_{hh} += dlin_{hh}.sum(dim=1)$

## 函数原型

 每个算子分为两段式接口，必须先调用“aclnnGRUBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGRUBackward”接口执行计算。

```Cpp
 aclnnStatus aclnnGRUBackwardGetWorkspaceSize(
   const aclTensor     *input,
   const aclTensorList *params,
   const aclTensor     *hx,
   const aclTensor     *dy,
   const aclTensor     *dh,
   const aclTensorList *r,
   const aclTensorList *z,
   const aclTensorList *n,
   const aclTensorList *h_n,
   const aclTensorList *h,
   const aclTensor     *batchSizesOptional,
   bool                hasBias,
   int64_t             numLayers,
   bool                bidirection,
   bool                batchFirst,
   aclTensor           *dxOut,
   aclTensor           *dhPrevOut,
   aclTensorList       *dparamsOut,
   uint64_t            *workspaceSize,
   aclOpExecutor       **executor)
```

```Cpp
 aclnnStatus aclnnGRUBackward(
   void            *workspace,
   uint64_t        workspaceSize,
   aclOpExecutor   *executor,
   aclrtStream     stream)
```

## aclnnGRUBackwardGetWorkspaceSize

* **参数说明：**

<table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
   <col style="width: 149px">
   <col style="width: 121px">
   <col style="width: 264px">
   <col style="width: 253px">
   <col style="width: 262px">
   <col style="width: 148px">
   <col style="width: 135px">
   <col style="width: 146px">
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
       <th>非连续Tensor</th>
     </tr></thead>
   <tbody>
     <tr>
       <td>input</td>
       <td>输入</td>
       <td>GRU的输入序列，对应公式中的x。</td>
       <td><ul><li>batch_size表示序列组数；time_step表示时间维度；input_size表示输入的特征数量。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>若传入有效batchSizesOptional，为[time_step * batch_size, input_size];</li>
       <li>若传入空指针batchSizesOptional，为[time_step, batch_size, input_size] 或 [batch_size, time_step, input_size]</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>params</td>
       <td>输入</td>
       <td>GRU每层的权重和偏置张量列表，对应公式中的w与b。</td>
       <td><ul><li>bidirection为True时 `D = 2`，否则 `D = 1`，hasBiases为True时 `B = 2`，否则 `B = 1`。列表长度为 D * B * num_layers * 2。</li><li>当bidirection和hasBias均为True时排布为：[weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]。</li>
       <li>hasBias为False时无bias项；bidirection为False时无reverse项。</li><li>多层时逐层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>weight_ih: [3*hidden_size, cur_input_size]<br>weight_hh: [3*hidden_size, hidden_size]<br>bias_ih: [3*hidden_size]<br>bias_hh: [3*hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>hx</td>
       <td>输入</td>
       <td>GRU每层的初始hidden状态。对应0时刻的h(t-1)。</td>
       <td><ul><li>多层双向时每个tensor数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>每个tensor shape为[numLayers * D, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dy</td>
       <td>输入</td>
       <td>GRU正向最后一层输出hidden的梯度。对应公式中的∂L/∂h^(l)。</td>
       <td><ul><li>双向时数据沿最后一维按前后向排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>[time_step, batch_size, hidden_size * D] 或 [batch_size, time_step, hidden_size * D]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dh</td>
       <td>输入</td>
       <td>GRU正向每层输出hidden在T时刻从下一个时间步传来的梯度。对应δh_next。</td>
       <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>[numLayers * D, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>r</td>
       <td>输入</td>
       <td>GRU正向中每层每个时刻重置门的激活值。对应公式中的r。</td>
       <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>列表中每个shape支持三维[time_step, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>z</td>
       <td>输入</td>
       <td>GRU正向中每层每个时刻更新门的激活值。对应公式中的z。</td>
       <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>列表中每个shape支持三维[time_step, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>n</td>
       <td>输入</td>
       <td>GRU正向中每层每个时刻候选隐藏状态的激活值。对应公式中的n。</td>
       <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>列表中每个shape支持三维[time_step, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>h_n</td>
       <td>输入</td>
       <td>GRU正向中每层每个时刻候选隐藏状态的中间值。对应公式中的$W_{hn}h_{t-1} + b_{hn}$。</td>
       <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>列表中每个shape支持三维[time_step, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>h</td>
       <td>输入</td>
       <td>GRU正向中每层每个时刻的隐藏状态。对应公式中的h。</td>
       <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>列表中每个shape支持三维[time_step, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>batchSizesOptional</td>
       <td>输入</td>
       <td>变长GRU输入序列各个时刻的有效序列batch数。当前仅支持传入空tensor。</td>
       <td><ul><li>变长序列时支持。</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>[time_step]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>hasBias</td>
       <td>输入</td>
       <td>表示是否有偏置b。</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>numLayers</td>
       <td>输入</td>
       <td>表示GRU层数。</td>
       <td><ul><li>值大于0。</li></ul></td>
       <td>INT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>bidirection</td>
       <td>输入</td>
       <td>表示是否是双向。</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>batchFirst</td>
       <td>输入</td>
       <td>表示输入数据input、y、dy格式是否是batch在第一维。</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>dxOut(grad_x_t)</td>
       <td>输出</td>
       <td>输入input上的梯度，对应公式中的δx。</td>
       <td><ul><li>shape与input一致。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>[time_step, batch_size, input_size] 或 [batch_size, time_step, input_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dhPrevOut(grad_h_prev)</td>
       <td>输出</td>
       <td>GRU每层初始hidden的梯度，对应t=0时的δh_prev。</td>
       <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>[D * num_layers, batch_size, hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dparamsOut(total_grad_w_ih, total_grad_w_hh)</td>
       <td>输出</td>
       <td>权重和偏置的梯度张量列表。对应公式中的δw和δb。</td>
       <td><ul><li>列表长度为 D * B * num_layers。</li><li>排布与输入params一致。</li><li>数据类型与input一致。</li></ul></td>
       <td>FLOAT32、FLOAT16</td>
       <td>ND</td>
       <td>dweight_ih: [3*hidden_size, cur_input_size]<br>dweight_hh: [3*hidden_size, hidden_size]<br>dbias: [3*hidden_size]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>workspaceSize</td>
       <td>输出</td>
       <td>返回需要在Device侧申请的workspace大小。</td>
       <td>Host侧出参。</td>
       <td>UINT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>executor</td>
       <td>输出</td>
       <td>返回op执行器，包含了算子计算流程。</td>
       <td>Host侧出参。</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
   </tbody></table>

* **返回值：**

 aclnnStatus: 返回状态码，具体参见[aclnn返回码]。

 第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
   <col style="width: 267px">
   <col style="width: 124px">
   <col style="width: 775px">
   </colgroup>
   <thead>
     <tr>
       <th>返回码</th>
       <th>错误码</th>
       <th>描述</th>
     </tr></thead>
   <tbody>
     <tr>
       <td>ACLNN_ERR_PARAM_NULLPTR</td>
       <td>161001</td>
       <td>如果传入参数为aclTensor或aclTensorList且非batchSizesOptional，是空指针。</td>
     </tr>
     <tr>
       <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
       <td rowspan="12">161002</td>
       <td>如果传入参数为aclTensor或aclTensorList，数据类型不在支持的范围之内。</td>
     </tr>
     <tr>
       <td>如果传入参数类型为aclTensor或aclTensorList，数据类型不同。</td>
     </tr>
     <tr>
       <td>如果传入参数类型为aclTensor或aclTensorList，shape不满足对应的shape要求。</td>
     </tr>
     <tr>
       <td>numLayers不满足>0。</td>
     </tr>
   </tbody>
   </table>

## aclnnGRUBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGRUBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码]。

## 约束说明

- 确定性计算：

  - aclnnGRUBackward默认确定性实现。
  - 支持FP16/FP32，所有输入的数据类型需保持一致
  - 输入numLayers大于等于3层时，可能会导致超时。

## 调用示例

 示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(const std::string& name, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret);
              return );
    LOG_PRINT("=== %s shape=[", name.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    LOG_PRINT("] ===\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
    LOG_PRINT("\n");
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType,
                        aclTensorList** tensor, T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t timeStep = 2;
    int64_t batchSize = 3;
    int64_t inputSize = 4;
    int64_t hiddenSize = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    bool hasBias = false;
    bool batchFirst = false;
    bool bidirection = false;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> dyShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> dhShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (hasBias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    // gate lists: r, z, n, hn, h each has ldScale tensors of [T, B, H]
    std::vector<std::vector<int64_t>> gateListShape;
    for (int64_t i = 0; i < ldScale; i++) {
        gateListShape.push_back({timeStep, batchSize, hiddenSize});
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dyDeviceAddr = nullptr;
    void* dhDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rDeviceAddr;
    std::vector<void*> zDeviceAddr;
    std::vector<void*> nDeviceAddr;
    std::vector<void*> hnDeviceAddr;
    std::vector<void*> hDeviceAddr;

    // output
    void* dxDeviceAddr = nullptr;
    std::vector<void*> dparamsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dhPrevDeviceAddr = nullptr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* dh = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* r = nullptr;
    aclTensorList* z = nullptr;
    aclTensorList* n = nullptr;
    aclTensorList* hn = nullptr;
    aclTensorList* h = nullptr;

    aclTensor* dxOut = nullptr;
    aclTensor* dhPrevOut = nullptr;
    aclTensorList* dparamsOut = nullptr;

    // 构造输入
    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params,
                                     1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dyHostData(GetShapeSize(dyShape), 0.5);
    ret = CreateAclTensor<float>(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhHostData(GetShapeSize(dhShape), 0.1);
    ret = CreateAclTensor<float>(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hxHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造 gate lists (r, z, n, hn, h) - 前向计算的中间值
    rDeviceAddr.resize(ldScale, nullptr);
    zDeviceAddr.resize(ldScale, nullptr);
    nDeviceAddr.resize(ldScale, nullptr);
    hnDeviceAddr.resize(ldScale, nullptr);
    hDeviceAddr.resize(ldScale, nullptr);

    ret = CreateAclTensorList<float>(gateListShape, rDeviceAddr.data(), aclDataType::ACL_FLOAT, &r, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, zDeviceAddr.data(), aclDataType::ACL_FLOAT, &z, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, nDeviceAddr.data(), aclDataType::ACL_FLOAT, &n, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hnDeviceAddr.data(), aclDataType::ACL_FLOAT, &hn, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hDeviceAddr.data(), aclDataType::ACL_FLOAT, &h, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出
    std::vector<float> dxHostData(GetShapeSize(inputShape), 0.0);
    ret = CreateAclTensor<float>(dxHostData, inputShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhPrevHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(dhPrevHostData, hxShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrevOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, dparamsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &dparamsOut,
                                     0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnGRUBackward第一段接口
    ret = aclnnGRUBackwardGetWorkspaceSize(
        input, params, hx, dy, dh, r, z, n, hn, h, nullptr, hasBias, numLayers, bidirection, batchFirst,
        dxOut, dhPrevOut, dparamsOut, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnGRUBackward第二段接口
    ret = aclnnGRUBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧
    PrintOutResult("dxOut", inputShape, &dxDeviceAddr);
    PrintOutResult("dhPrevOut", hxShape, &dhPrevDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        PrintOutResult("dparamsOut[" + std::to_string(i) + "]", paramsListShape[i], &dparamsListDeviceAddr[i]);
    }

    // 6. 释放aclTensor和aclTensorList
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(dy);
    aclDestroyTensor(dh);
    aclDestroyTensor(hx);

    aclDestroyTensorList(r);
    aclDestroyTensorList(z);
    aclDestroyTensorList(n);
    aclDestroyTensorList(hn);
    aclDestroyTensorList(h);

    aclDestroyTensor(dxOut);
    aclDestroyTensor(dhPrevOut);
    aclDestroyTensorList(dparamsOut);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(dyDeviceAddr);
    aclrtFree(dhDeviceAddr);
    aclrtFree(hxDeviceAddr);

    for (size_t i = 0; i < rDeviceAddr.size(); i++) {
        aclrtFree(rDeviceAddr[i]);
        aclrtFree(zDeviceAddr[i]);
        aclrtFree(nDeviceAddr[i]);
        aclrtFree(hnDeviceAddr[i]);
        aclrtFree(hDeviceAddr[i]);
    }

    aclrtFree(dxDeviceAddr);
    aclrtFree(dhPrevDeviceAddr);
    for (size_t i = 0; i < dparamsListDeviceAddr.size(); i++) {
        aclrtFree(dparamsListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
