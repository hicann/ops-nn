# GeluGrad
## 贡ls献说明
| 贡献者   | 贡献方              | 贡献算子     | 贡献时间      | 贡献内容         |
|-------|------------------|----------|-----------|--------------|
| skywang2 | 个人开发者 | GeluGrad | 2025/12/31 | 新增GeluGrad算子 |

## 支持的产品型号
- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

  `GeluGrad`算子用于计算Gelu函数的梯度。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GeluGrad</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
    <tr><td align="center">dy</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
    </tr> 
    <tr><td rowspan="2" align="center">算子输入</td>
    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
    </tr> 
    <tr><td rowspan="1" align="center">算子输入</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">z</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>     
  </table>

## 约束与限制
- dy,x,y,z,out的数据类型仅支持float32,float16,bfloat16，数据格式仅支持ND

### 运行验证
```bash
bash build.sh --run_example gelu_grad eager cust --vendor_name=custom --experimental
```
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_gelu_grad.cpp"> test_aclnn_gelu_grad.cpp</td><td>通过aclnn调用的方式调用GeluGrad算子。</td>
    </tr>
</table>
