# aclnnInplaceMaskedScatter

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/masked_scatter)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

根据掩码(mask)张量中元素为True的位置，复制(source)中的元素到(selfRef)对应的位置上。

## 函数原型

每个算子分为[两段式接口](../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceMaskedScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceMaskedScatter”接口执行计算。

```Cpp
aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
 aclTensor*         selfRef,
 const aclTensor*   mask,
 const aclTensor*   source,
 uint64_t*          workspaceSize,
 aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnInplaceMaskedScatter(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnInplaceMaskedScatterGetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1477px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 255px">
    <col style="width: 270px">
    <col style="width: 143px">
    <col style="width: 164px">
    <col style="width: 145px">
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
        <td>selfRef</td>
        <td>输入</td>
        <td>输入Tensor。</td>
        <td>-</td>
        <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>输入</td>
        <td>输入Tensor。</td>
        <td>shape不能大于selfRef，且需要和selfRef满足<a href="../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>source</td>
        <td>输入</td>
        <td>输入Tensor。</td>
        <td>元素数量需要大于等于mask中元素为true的数量。</td>
        <td>与selfRef相同</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的selfRef、mask、source是空指针。</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>selfRef和mask的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td>selfRef和mask的shape无法做broadcast。</td>
      </tr>
      <tr>
      <td>mask的shape维度大于selfRef。</td>
      </tr>
      <tr>
      <td>source的数据类型和selfRef的数据类型不同。</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceMaskedScatter

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceMaskedScatterGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_masked_scatter](./examples/test_aclnn_masked_scatter.cpp) | 通过[aclnnInplaceMaskedScatter](docs/aclnnInplaceMaskedScatter.md)接口方式调用MaskedScatter算子。 |