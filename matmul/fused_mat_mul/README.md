# FusedMatMul


##  产品支持情况

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 600px">
    <col style="width: 120px">
    </colgroup>
    <thread>
      <tr>
        <th>产品</th>
        <th>是否支持</th>
      </tr></thread>
    <tbody>
      <tr>
        <td><term>Ascend 950PR/Ascend 950DT</term></td>
        <td>√</td>
      </tr>
      <tr>
        <td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></td>
        <td>×</td>
      </tr>
  </tbody></table> 


## 功能说明

- 算子功能：矩阵乘与通用向量计算融合。
- 计算公式：

  $$
  y = OP((x1 @ x2 + bias), x3)
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
    <thread>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td><ul><li>数据类型需要与x2满足数据类型推导规则（参见<a href="../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td><ul><li>数据类型需要与x1满足数据类型推导规则（参见<a href="../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>公式中的输入bias。</td>
        <td><ul><li>仅当fusedOpType为""、"relu"、"add"、"mul"时生效，其他情况传入空指针即可。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>输入</td>
        <td>公式中的输入x3。</td>
        <td><ul>-</td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>输入</td>
        <td>公式中的输入OP。</td>
        <td><ul><li>融合模式取值必须是""（表示不做融合）、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"relu"中的一种。</li></td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>公式中的输出y。</td>
        <td><ul><li>数据类型需要与x1和x2推导后的数据类型一致（参见<a href="../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
  </tbody></table>


## 约束说明

- 当fusedOpType取值为"gelu_erf"、"gelu_tanh"时，x1、x2、x3的数据类型必须为BFLOAT16、FLOAT16;当fusedOpType为""、"relu"、"add"、"mul"时, x1、x2、x3的数据类型必须为FLOAT32(仅支持使能HFLOAT32场景)、BFLOAT16、FLOAT16。
## 调用说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 170px">
    <col style="width: 300px">
    <col style="width: 430px">
    </colgroup>
    <thread>
      <tr>
        <th>调用方式</th>
        <th>样例代码</th>
        <th>说明</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>aclnn接口</td>
        <td><a href="examples/arch35/test_aclnn_fused_mat_mul.cpp">test_aclnn_fused_mat_mul</a></td>
        <td>通过<a href="docs/aclnnFusedMatmul.md">aclnnFusedMatmul</a>接口方式调用FusedMatmul算子</td>
      </tr>
  </tbody></table> 