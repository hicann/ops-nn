<!-- codespell:ignore Silu -->
<!-- cspell:ignore Silu -->

# aclnnFusedMatmulSilu

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 910B</term> | √ |

## 功能说明

- 接口功能：完成Matmul与SiLU激活的融合计算，公式为`y = silu(x @ weight^T + bias)`。

- 精度说明：Cube完成FP32累加后，Matmul结果会先转换为BF16并写入GM。AIV读取该BF16中间结果并转换为FP32，执行bias Add和SiLU后，再转换为BF16写回输出。因此结果包含一次Matmul中间结果的BF16截断。

- 计算公式：

  $$
  linear_{m,n} = \sum_{k=0}^{K-1} x_{m,k} \cdot weight_{n,k} + bias_n
  $$

  $$
  y_{m,n} = \frac {linear_{m,n}} {1 + e^{-linear_{m,n}}}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnFusedMatmulSiluGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedMatmulSilu”接口执行计算。

```Cpp
aclnnStatus aclnnFusedMatmulSiluGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* weight,
  const aclTensor* bias,
  aclTensor*       y,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedMatmulSilu(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnFusedMatmulSiluGetWorkspaceSize

- **参数说明：**

  <table>
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>Matmul计算的左矩阵。</td>
        <td>shape为[M, K]，K需要为64的整数倍，且K <= 4096。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2维</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>Matmul计算的右矩阵，计算时按转置参与矩阵乘。</td>
        <td>shape为[N, K]，weight的第1维需要与x的第1维一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2维</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>Matmul计算的偏置。</td>
        <td>shape为[N]，长度需要与weight的第0维一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>1维</td>
        <td>√</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>计算输出。</td>
        <td>shape为[M, N]。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2维</td>
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
        <td>返回op执行器，包含算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码说明。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table>
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
        <td>传入的x、weight、bias、y为空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入或输出的数据类型、数据格式、维度、shape不满足约束。</td>
      </tr>
    </tbody>
  </table>

## aclnnFusedMatmulSilu

- **参数说明：**

  <table>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码说明。

## 约束说明

- 仅支持Ascend 910B。
- 仅支持BFLOAT16数据类型与ND数据格式。
- 支持动态shape，但`x`、`weight`、`bias`、`y`的rank分别固定为2、2、1、2；运行时实际`K`需满足下述约束。
- 当前实现要求K为64的整数倍，且K <= 4096。
- 当前实现使用`MatmulImpl`执行Cube矩阵乘主路径，并使用`MultiCoreMatmulTiling`进行多核切分；bias与SiLU由AIV后处理完成。
- Cube完成FP32累加后会将Matmul结果转换为BF16写入GM，AIV读取该BF16中间结果并转换为FP32执行bias Add和SiLU，最后转换为BF16写回输出。

## 调用示例

示例代码请参考[examples](../examples)目录。
