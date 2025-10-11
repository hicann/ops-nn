# aclnnEdgeSoftmax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     x    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入图的每个节点的所有入边特征，按节点分组进行 softmax 归一化。
- 计算公式：对于一个节点 $i$，

  $$
  a_{ij} = \frac{\exp(z_{ij})}{\sum_{j \in \mathcal{N}(i)} \exp(z_{ij})}
  $$

  其中 $z_{ij}$ 是边 $j \rightarrow i$ 的信号，在 softmax 的上下文中也称为 logits。$\mathcal{N}(i)$ 是指向节点 $i$ 的所有节点的集合。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnEdgeSoftmaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEdgeSoftmax”接口执行计算。
```Cpp
aclnnStatus aclnnEdgeSoftmaxGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *idx,
    int64_t n,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnEdgeSoftmax(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnEdgeSoftmaxGetWorkspaceSize

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1458px"><colgroup>
  <col style="width: 154px">
  <col style="width: 120px">
  <col style="width: 276px">
  <col style="width: 308px">
  <col style="width: 212px">
  <col style="width: 107px">
  <col style="width: 136px">
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
      <td>x</td>
      <td>输入</td>
      <td>边特征张量</td>
      <td>形状为 (E, F) 或 (E,)</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-2</td>
      <td>x</td>
    </tr>
      <tr>
      <td>idx</td>
      <td>输入</td>
      <td>目标节点索引</td>
      <td>形状为 (E,)</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr> 
      <tr>
      <td>n</td>
      <td>输入</td>
      <td>节点总数</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr> 
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>归一化后边特征</td>
      <td>形状为 (E, F) 或 (E,)</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
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
  </tbody>
  </table>
  
  

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
## aclnnEdgeSoftmax

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnEdgeSoftmaxGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无。