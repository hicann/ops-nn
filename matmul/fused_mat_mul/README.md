# FusedMatMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：矩阵乘与通用向量计算融合。
- 计算公式：

  $$
  y = OP((x1 @ x2 + bias), x3)
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
        <td><ul><li>仅当fusedOpType为""、"16cast32"、"relu"、"add"、"mul"时生效，其他情况传入空指针即可。</li></ul></td>
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
        <td><ul><li>融合模式取值必须是""（表示不做融合）、"16cast32"、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"relu"中的一种。</li><li>"scale_add"仅供算子内部使用，不支持用户直接传入；使用缩放系数时，运行日志中可能出现该取值。</li></ul></td>
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

- 当fusedOpType取值为"gelu_erf"、"gelu_tanh"时，x1、x2的数据类型必须为BFLOAT16、FLOAT16;当fusedOpType为""、"relu"时, x1、x2的数据类型必须为FLOAT32（仅支持开启HFLOAT32场景）、BFLOAT16、FLOAT16；当fusedOpType取值为"16cast32"时，x1、x2的数据类型必须为BFLOAT16、FLOAT16；当fusedOpType为"add"、"mul"时, x1、x2、x3的数据类型必须为FLOAT32（仅支持开启HFLOAT32场景）、BFLOAT16、FLOAT16。
- 当fusedOpType取值为"16cast32"时，输出y的数据类型必须为FLOAT32。
- alpha或beta为非默认值时，仅支持Ascend 950PR/Ascend 950DT上的三维非转置add场景，不支持bias和batch轴广播，x1、x2、x3和y必须为相同的FLOAT16或BFLOAT16数据类型。

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
