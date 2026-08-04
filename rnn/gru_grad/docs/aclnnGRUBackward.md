# aclnnGRUBackward

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                          |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
| <term>Atlas 推理系列产品 </term>                                                |    ×    |
| <term>Atlas 训练系列产品</term>                                                 |    ×    |

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
   const aclTensorList *hx,
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
   bool                bidirectional,
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
       <td><ul><li>bidirection为True时 `D = 2`，否则 `D = 1`，hasBiases为True时 `B = 2`，否则 `B = 1`。列表长度为 D * B * num_layers。</li><li>当bidirection和hasBias均为True时排布为：[weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]。</li>
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
       <td>表示输入数据input、y、dy、dxOut格式是否是batch在第一维。</td>
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 单层单向、带偏置(bias)的GRU反向样例
    int64_t timeStep = 2;
    int64_t batchSize = 1;
    int64_t inputSize = 4;
    int64_t hiddenSize = 4;
    int64_t gateNum = 3; // GRU: reset / update / new
    int64_t numLayers = 1;
    bool hasBias = true;
    bool bidirectional = false;
    bool batchFirst = false;

    // 形状定义
    std::vector<int64_t> xShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> wiShape = {gateNum * hiddenSize, inputSize};
    std::vector<int64_t> whShape = {gateNum * hiddenSize, hiddenSize};
    std::vector<int64_t> bShape = {gateNum * hiddenSize};
    // 初始hidden状态: [batchSize, hiddenSize]，列表长度 D * numLayers = 1
    std::vector<int64_t> initHShape = {batchSize, hiddenSize};
    std::vector<int64_t> dyShape = {timeStep, batchSize, hiddenSize};
    std::vector<int64_t> dhShape = {numLayers, batchSize, hiddenSize};
    // 每步门激活值/中间值: [timeStep, batchSize, hiddenSize]
    std::vector<int64_t> gateShape = {timeStep, batchSize, hiddenSize};

    // 反向传播输出张量形状
    std::vector<int64_t> dxShape = xShape;           // 与input相同
    std::vector<int64_t> dhPrevShape = dhShape;      // [D * numLayers, batchSize, hiddenSize]
    std::vector<int64_t> dwiShape = wiShape;         // 与wi相同
    std::vector<int64_t> dwhShape = whShape;         // 与wh相同
    std::vector<int64_t> dbShape = bShape;           // 与b相同

    // 设备地址指针
    void* xDeviceAddr = nullptr;
    void* wiDeviceAddr = nullptr;
    void* whDeviceAddr = nullptr;
    void* biDeviceAddr = nullptr;
    void* bhDeviceAddr = nullptr;
    void* initHDeviceAddr = nullptr;
    void* dyDeviceAddr = nullptr;
    void* dhDeviceAddr = nullptr;
    void* rDeviceAddr = nullptr;
    void* zDeviceAddr = nullptr;
    void* nDeviceAddr = nullptr;
    void* hnDeviceAddr = nullptr;
    void* hDeviceAddr = nullptr;

    // 反向传播输出设备地址指针
    void* dxDeviceAddr = nullptr;
    void* dhPrevDeviceAddr = nullptr;
    void* dwiDeviceAddr = nullptr;
    void* dwhDeviceAddr = nullptr;
    void* dbiDeviceAddr = nullptr;
    void* dbhDeviceAddr = nullptr;

    // ACL Tensor 指针
    aclTensor* x = nullptr;
    aclTensor* wi = nullptr;
    aclTensor* wh = nullptr;
    aclTensor* bi = nullptr;
    aclTensor* bh = nullptr;
    aclTensor* initH = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* dh = nullptr;
    aclTensor* r = nullptr;
    aclTensor* z = nullptr;
    aclTensor* n = nullptr;
    aclTensor* hn = nullptr;
    aclTensor* h = nullptr;

    // 反向传播输出 ACL Tensor 指针
    aclTensor* dx = nullptr;
    aclTensor* dhPrev = nullptr;
    aclTensor* dwi = nullptr;
    aclTensor* dwh = nullptr;
    aclTensor* dbi = nullptr;
    aclTensor* dbh = nullptr;

    // 输入数据。注意: r/z 为 sigmoid 输出、n 为 tanh 输出，必须取非饱和值(如 0.3/0.5/0.2)，
    // 否则激活导数为0会导致反向梯度全为0，看不出计算结果。
    std::vector<float> xHostData(GetShapeSize(xShape), 0.5f);
    std::vector<float> wiHostData(GetShapeSize(wiShape), 0.1f);
    std::vector<float> whHostData(GetShapeSize(whShape), 0.1f);
    std::vector<float> biHostData(GetShapeSize(bShape), 0.0f);
    std::vector<float> bhHostData(GetShapeSize(bShape), 0.0f);
    std::vector<float> initHHostData(GetShapeSize(initHShape), 0.5f);
    std::vector<float> dyHostData(GetShapeSize(dyShape), 1.0f);
    std::vector<float> dhHostData(GetShapeSize(dhShape), 0.5f);
    std::vector<float> rHostData(GetShapeSize(gateShape), 0.3f);  // reset gate(sigmoid), 非饱和
    std::vector<float> zHostData(GetShapeSize(gateShape), 0.5f);  // update gate(sigmoid), 非饱和
    std::vector<float> nHostData(GetShapeSize(gateShape), 0.2f);  // candidate(tanh), 非饱和
    std::vector<float> hnHostData(GetShapeSize(gateShape), 0.3f); // W_hn*h_{t-1}+b_hn 中间值
    std::vector<float> hHostData(GetShapeSize(gateShape), 0.5f);  // 隐藏状态 h

    // 反向传播输出数据(初始化为0)
    std::vector<float> dxHostData(GetShapeSize(dxShape), 0.0f);
    std::vector<float> dhPrevHostData(GetShapeSize(dhPrevShape), 0.0f);
    std::vector<float> dwiHostData(GetShapeSize(dwiShape), 0.0f);
    std::vector<float> dwhHostData(GetShapeSize(dwhShape), 0.0f);
    std::vector<float> dbiHostData(GetShapeSize(dbShape), 0.0f);
    std::vector<float> dbhHostData(GetShapeSize(dbShape), 0.0f);

    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 params(w_ih, w_hh, b_ih, b_hh) aclTensorList
    ret = CreateAclTensor(wiHostData, wiShape, &wiDeviceAddr, aclDataType::ACL_FLOAT, &wi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(whHostData, whShape, &whDeviceAddr, aclDataType::ACL_FLOAT, &wh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biHostData, bShape, &biDeviceAddr, aclDataType::ACL_FLOAT, &bi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bhHostData, bShape, &bhDeviceAddr, aclDataType::ACL_FLOAT, &bh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* paramsArray[] = {wi, wh, bi, bh};
    auto paramsList = aclCreateTensorList(paramsArray, sizeof(paramsArray) / sizeof(paramsArray[0]));

    // 创建 hx(initH) aclTensorList，列表长度 D * numLayers = 1
    ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT, &initH);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* hxArray[] = {initH};
    auto hxList = aclCreateTensorList(hxArray, sizeof(hxArray) / sizeof(hxArray[0]));

    // 创建 dy / dh aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 r / z / n / h_n / h aclTensorList，列表长度 D * numLayers = 1
    ret = CreateAclTensor(rHostData, gateShape, &rDeviceAddr, aclDataType::ACL_FLOAT, &r);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* rArray[] = {r};
    auto rList = aclCreateTensorList(rArray, sizeof(rArray) / sizeof(rArray[0]));

    ret = CreateAclTensor(zHostData, gateShape, &zDeviceAddr, aclDataType::ACL_FLOAT, &z);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* zArray[] = {z};
    auto zList = aclCreateTensorList(zArray, sizeof(zArray) / sizeof(zArray[0]));

    ret = CreateAclTensor(nHostData, gateShape, &nDeviceAddr, aclDataType::ACL_FLOAT, &n);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* nArray[] = {n};
    auto nList = aclCreateTensorList(nArray, sizeof(nArray) / sizeof(nArray[0]));

    ret = CreateAclTensor(hnHostData, gateShape, &hnDeviceAddr, aclDataType::ACL_FLOAT, &hn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* hnArray[] = {hn};
    auto hnList = aclCreateTensorList(hnArray, sizeof(hnArray) / sizeof(hnArray[0]));

    ret = CreateAclTensor(hHostData, gateShape, &hDeviceAddr, aclDataType::ACL_FLOAT, &h);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* hArray[] = {h};
    auto hList = aclCreateTensorList(hArray, sizeof(hArray) / sizeof(hArray[0]));

    // 创建反向传播输出张量
    ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dhPrevHostData, dhPrevShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrev);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 dparams(dw_ih, dw_hh, db_ih, db_hh) aclTensorList
    ret = CreateAclTensor(dwiHostData, dwiShape, &dwiDeviceAddr, aclDataType::ACL_FLOAT, &dwi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dwhHostData, dwhShape, &dwhDeviceAddr, aclDataType::ACL_FLOAT, &dwh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dbiHostData, dbShape, &dbiDeviceAddr, aclDataType::ACL_FLOAT, &dbi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dbhHostData, dbShape, &dbhDeviceAddr, aclDataType::ACL_FLOAT, &dbh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* dparamsArray[] = {dwi, dwh, dbi, dbh};
    auto dparamsList = aclCreateTensorList(dparamsArray, sizeof(dparamsArray) / sizeof(dparamsArray[0]));

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnGRUBackward第一段接口
    ret = aclnnGRUBackwardGetWorkspaceSize(x, paramsList, hxList, dy, dh, rList, zList, nList, hnList, hList, nullptr,
                                           hasBias, numLayers, bidirectional, batchFirst, dx, dhPrev, dparamsList,
                                           &workspaceSize, &executor);
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

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult("dx", dxShape, &dxDeviceAddr);
    PrintOutResult("dhPrev", dhPrevShape, &dhPrevDeviceAddr);
    PrintOutResult("dw_ih", dwiShape, &dwiDeviceAddr);
    PrintOutResult("dw_hh", dwhShape, &dwhDeviceAddr);
    PrintOutResult("db_ih", dbShape, &dbiDeviceAddr);
    PrintOutResult("db_hh", dbShape, &dbhDeviceAddr);

    // 6. 释放aclTensor和aclTensorList
    aclDestroyTensor(x);
    aclDestroyTensor(wi);
    aclDestroyTensor(wh);
    aclDestroyTensor(bi);
    aclDestroyTensor(bh);
    aclDestroyTensor(initH);
    aclDestroyTensor(dy);
    aclDestroyTensor(dh);
    aclDestroyTensor(r);
    aclDestroyTensor(z);
    aclDestroyTensor(n);
    aclDestroyTensor(hn);
    aclDestroyTensor(h);
    aclDestroyTensor(dx);
    aclDestroyTensor(dhPrev);
    aclDestroyTensor(dwi);
    aclDestroyTensor(dwh);
    aclDestroyTensor(dbi);
    aclDestroyTensor(dbh);

    aclDestroyTensorList(paramsList);
    aclDestroyTensorList(hxList);
    aclDestroyTensorList(rList);
    aclDestroyTensorList(zList);
    aclDestroyTensorList(nList);
    aclDestroyTensorList(hnList);
    aclDestroyTensorList(hList);
    aclDestroyTensorList(dparamsList);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(wiDeviceAddr);
    aclrtFree(whDeviceAddr);
    aclrtFree(biDeviceAddr);
    aclrtFree(bhDeviceAddr);
    aclrtFree(initHDeviceAddr);
    aclrtFree(dyDeviceAddr);
    aclrtFree(dhDeviceAddr);
    aclrtFree(rDeviceAddr);
    aclrtFree(zDeviceAddr);
    aclrtFree(nDeviceAddr);
    aclrtFree(hnDeviceAddr);
    aclrtFree(hDeviceAddr);
    aclrtFree(dxDeviceAddr);
    aclrtFree(dhPrevDeviceAddr);
    aclrtFree(dwiDeviceAddr);
    aclrtFree(dwhDeviceAddr);
    aclrtFree(dbiDeviceAddr);
    aclrtFree(dbhDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
