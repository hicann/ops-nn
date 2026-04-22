# Swish
## 贡献说明
| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Swish | 2025/12/31 | 新增Swish算子 |

## 支持的产品型号

- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Swish`算子实现Swish激活函数，是一种由输入与其经过Sigmoid函数结果相乘得到的平滑、非线性函数。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Swish</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
     
    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">算子属性</td>
    <td align="center">scale</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">swish</td></tr>  
  </table>

## 约束与限制
- x,y的数据类型仅支持float32,float16,bfloat1，数据格式仅支持ND

### 运行验证
编译
```bash
bash build.sh --pkg --soc=ascend910b --experimental --ops=swish
```
运行
```bash
bash build.sh --run_example swish eager cust --vendor_name=custom --experimental
```
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_swish.cpp">test_aclnn_swish.cpp</td><td>通过aclnn调用的方式调用Swish算子。</td>
    </tr>
</table>
