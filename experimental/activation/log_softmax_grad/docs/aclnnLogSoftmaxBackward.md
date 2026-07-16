# aclnnLogSoftmaxBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>     |     √    |

## 功能说明

- 接口功能：完成 LogSoftmax 的反向传播，计算 LogSoftmax 正向输入的梯度（gradInput）。
- 计算公式：

  其中 $output$ 为 LogSoftmax 正向输出（即 $\text{log\_softmax}(x)$），$gradOutput$ 为上游梯度：

  $$
  gradInput = gradOutput - \exp(output) \cdot \text{reduce\_sum}(gradOutput, dim)
  $$

  由于 $\exp(output)$ 即为 softmax 概率分布，上述计算等价于：

  $$
  gradInput = gradOutput - softmax(x) \cdot \text{reduce\_sum}(gradOutput, dim)
  $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLogSoftmaxBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLogSoftmaxBackward”接口执行计算。

```Cpp
aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* output,
  int64_t          dim,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnLogSoftmaxBackward(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnLogSoftmaxBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1310px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>LogSoftmax正向输出的梯度，公式中的gradOutput。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、output与out的shape一致。</li><li>gradOutput、output与out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输入</td>
      <td>LogSoftmax正向输出，公式中的output（已归一化的log_softmax结果）。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、output与out的shape一致。</li><li>gradOutput、output与out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>指定进行reduce的维度。</td>
      <td>取值范围为[-dimNum, dimNum-1]，其中dimNum为gradOutput的维度数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>backward计算的输出，为LogSoftmax正向输入的梯度值，公式中的gradInput。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、output与out的shape一致。</li><li>gradOutput、output与out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOutput、output或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>gradOutput或output的数据类型不在支持的范围（FLOAT、FLOAT16、BFLOAT16）之内。</td>
    </tr>
    <tr>
      <td>gradOutput与output的数据类型不同。</td>
    </tr>
    <tr>
      <td>gradOutput、output与out的shape不同。</td>
    </tr>
    <tr>
      <td>dim不在gradOutput的维度范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput或output的维度超过8。</td>
    </tr>
    <tr>
      <td>当前硬件不支持bfloat16向float32的Cast时，使用BFLOAT16数据类型。</td>
    </tr>
  </tbody></table>

## aclnnLogSoftmaxBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLogSoftmaxBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 输入gradOutput、output与输出out的shape必须一致，不支持广播。
- 指定维度dim需在gradOutput的维度范围内，范围[-dimNum, dimNum-1]。
- 支持的数据类型为FLOAT、FLOAT16、BFLOAT16；当硬件不支持bfloat16向float32的Cast时，使用BFLOAT16会报错。
- 输入tensor的维度数不能超过8。

## 调用示例

完整可编译的调用示例代码见 [test_aclnn_log_softmax_grad.cpp](../examples/test_aclnn_log_softmax_grad.cpp)。

具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。
