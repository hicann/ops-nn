# FusedMatMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>    |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：矩阵乘与通用向量计算融合。
- 计算公式：

  $$
  y = OP((x1 @ x2 + bias), x3)
  $$

  sigmoid运算:

  $$
  y = sigmoid(x1 @ x2 + bias) = \frac{1}{1 + \exp(-(x1 @ x2 + bias))}
  $$

  16cast32运算:

  $$
  y = cast\_float32(x1 @ x2 + bias)
  $$

  通过aclnnFusedMatmulV2传入非默认alpha或beta，且fusedOpType为"add"时：

  $$
  y = alpha * (x1 @ x2) + beta * x3
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td><ul><li>数据类型需要与x2满足数据类型推导规则（参见<a href="../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td><ul><li>数据类型需要与x1满足数据类型推导规则（参见<a href="../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</li></ul></td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>公式中的输入bias。</td>
        <td><ul><li>仅当fusedOpType为""、"16cast32"、"relu"、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"sigmoid"时生效，shape必须为(N,)或(1, N)，N为x2逻辑shape的最后一维；不使用bias时传入空指针。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>输入</td>
        <td>公式中的输入x3。</td>
        <td><ul><li>仅当fusedOpType为"add"、"mul"时生效，其他情况传入空指针即可。</li></ul></td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>属性</td>
        <td>用于缩放矩阵乘结果的可选系数，默认值为1.0。</td>
        <td><ul><li>仅由支持缩放系数的接口设置；非默认值仅支持fusedOpType为"add"的场景。</li></ul></td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta</td>
        <td>属性</td>
        <td>用于缩放x3的可选系数，默认值为1.0。</td>
        <td><ul><li>仅由支持缩放系数的接口设置；非默认值仅支持fusedOpType为"add"的场景。</li></ul></td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>输入</td>
        <td>公式中的输入OP。</td>
        <td><ul><li>融合模式取值必须是""（表示不做融合）、"16cast32"、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"relu"、"sigmoid"中的一种。</li><li>"scale_add"仅供算子内部使用，不支持用户直接传入；使用缩放系数时，运行日志中可能出现该取值。</li></ul></td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>公式中的输出y。</td>
        <td><ul><li>数据类型需要与x1和x2推导后的数据类型一致（参见<a href="../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）；当fusedOpType为"16cast32"时，y的数据类型固定为FLOAT32。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
  </tbody></table>

## 约束说明

- 所有支持 bias 的融合模式均要求 bias 的 shape 为 (N,) 或 (1, N)，其中 N 为 x2 逻辑 shape 的最后一维。bias 的数据类型必须与矩阵输入一致或为 FLOAT32；传入 bias 时，矩阵乘的 K 轴必须大于 0。
- GELU（gelu_erf、gelu_tanh）和 Sigmoid（sigmoid）的计算顺序为先计算矩阵乘并加 bias，再逐元素执行激活；未传入 bias 时不加偏置。
- 当fusedOpType取值为"sigmoid"时，仅支持Ascend 950PR&950DT系列产品，x1、x2和y必须为二维，数据类型必须同为FLOAT16或同为BFLOAT16；x1的shape为(M, K)，x2的shape为(K, N)，y的shape为(M, N)，M、N和K均必须大于0，x3必须传入空指针。
- 当fusedOpType取值为"sigmoid"时，alpha和beta只能取1或不设置，不支持空Tensor。

- 当fusedOpType取值为"gelu_erf"、"gelu_tanh"时，x1、x2的数据类型必须为BFLOAT16、FLOAT16;当fusedOpType为""、"relu"时, x1、x2的数据类型必须为FLOAT32（仅支持开启HFLOAT32场景）、BFLOAT16、FLOAT16；当fusedOpType取值为"16cast32"时，x1、x2的数据类型必须为BFLOAT16、FLOAT16；当fusedOpType为"add"、"mul"时, x1、x2、x3的数据类型必须为FLOAT32（仅支持开启HFLOAT32场景）、BFLOAT16、FLOAT16。
- 当fusedOpType取值为"16cast32"时，输出y的数据类型必须为FLOAT32。
- alpha或beta为非默认值时，仅支持Ascend 950PR&950DT系列产品上的三维非转置add场景，不支持bias和batch轴广播，x1、x2、x3和y必须为相同的FLOAT16、BFLOAT16或FLOAT32数据类型（FLOAT32时仅支持开启HF32，以HF32模式计算）。

## 调用说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 170px">
    <col style="width: 300px">
    <col style="width: 430px">
    </colgroup>
    <thead>
      <tr>
        <th>调用方式</th>
        <th>样例代码</th>
        <th>说明</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>aclnn接口</td>
        <td><a href="examples/arch35/test_aclnn_fused_mat_mul.cpp">test_aclnn_fused_mat_mul</a></td>
        <td>通过<a href="docs/aclnnFusedMatmul.md">aclnnFusedMatmul</a>、<a href="docs/aclnnFusedMatmulV2.md">aclnnFusedMatmulV2</a>接口方式调用FusedMatmul算子。该样例使用V1接口。</td>
      </tr>
      <tr>
        <td>torch接口</td>
        <td><a href="torch_extension/fused_mat_mul.py">fused_matmul</a></td>
        <td>通过<a href="docs/torchapi_fused_matmul.md">fused_matmul</a>接口方式调用FusedMatmul算子。</td>
      </tr>
  </tbody></table>
