# ForeachBinaryOp

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_binary_op)

> 说明：ForeachBinaryOp 是一个图融合（fused）的图内部算子，仅通过图模式（GE IR / 图融合 Pass）使用，**不对外提供 aclnn 单算子接口**。本文档按图模式（GEIR）方式描述算子定义与约束。

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term>（arch35/ascend950） | √ |
| 其它产品 | × |

## 功能说明

- 接口功能：对两个 Tensor 列表 `x1`、`x2` 逐 Tensor、逐元素做二元运算，运算类型由属性 `op_code` 在编译期选择（0=add、1=sub、2=mul、3=div）。该算子将多种二元 foreach 运算统一为一个算子，便于图融合。

- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}],\ x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}],\ y = [{y_0}, {y_1}, ... {y_{n-1}}]
  $$

  $$
  y_i = x1_i \odot x2_i \quad (i=0,1,...,n-1)
  $$

  其中 $\odot$ 由 `op_code` 决定：

  | op_code | 运算 | 公式 |
  |:---:|:---:|:---|
  | 0 | add | $y_i = x1_i + x2_i$ |
  | 1 | sub | $y_i = x1_i - x2_i$ |
  | 2 | mul | $y_i = x1_i \times x2_i$ |
  | 3 | div | $y_i = x1_i / x2_i$ |

  > 整型（INT32）除法对除数为 0 的元素结果置 0（规避未定义行为）；浮点除以 0 时遵循 IEEE 语义产生 inf/nan。

## 算子定义

REG_OP(ForeachBinaryOp)，详见 [op_graph/foreach_binary_op_proto.h](../op_graph/foreach_binary_op_proto.h)。

- **输入**

  | 参数名 | 类型 | 描述 | 数据类型 | 数据格式 |
  |:---|:---|:---|:---|:---:|
  | x1 | DYNAMIC_INPUT（TensorList） | 第一个输入张量列表，对应公式中的 `x1`。列表内所有 Tensor 的数据类型一致。 | FLOAT16、FLOAT、INT32、BFLOAT16 | ND |
  | x2 | DYNAMIC_INPUT（TensorList） | 第二个输入张量列表，对应公式中的 `x2`。数据类型、shape 与 `x1` 一致。 | FLOAT16、FLOAT、INT32、BFLOAT16 | ND |

- **属性**

  | 属性名 | 类型 | 必选 | 描述 |
  |:---|:---|:---:|:---|
  | op_code | Int | 是（REQUIRED_ATTR） | 选择二元运算：0=add、1=sub、2=mul、3=div。取值范围 [0, 4)。 |

- **输出**

  | 参数名 | 类型 | 描述 | 数据类型 | 数据格式 |
  |:---|:---|:---|:---|:---:|
  | y | DYNAMIC_OUTPUT（TensorList） | 输出张量列表，对应公式中的 `y`，`y_i = x1_i <op> x2_i`。数据类型、shape 与 `x1` 一致。 | FLOAT16、FLOAT、INT32、BFLOAT16 | ND |

## 约束说明

- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>（arch35/ascend950），SIMT kernel 实现。
- `x1`、`x2`、`y` 三个列表长度（Tensor 个数）一致，且一一对应的 Tensor shape 一致。
- 列表 Tensor 个数上限为 256（`MAX_TENSOR_NUM_FOREACH_BINARY_OP`）。
- `x1`、`x2`、`y` 中每个 Tensor 的数据类型一致，且属于 FLOAT16/FLOAT/INT32/BFLOAT16。
- 支持空 Tensor（总元素数为 0 时不启核）。
- `op_code` 必须落在 [0, 4) 区间，否则 tiling 报错。
- TilingKey 规则：`tilingKey = op_code * 4 + dtypeIdx`，其中 dtypeIdx：FP16=0、FP32=1、INT32=2、BF16=3，共 16 种调度模式 [0, 15]。

## 调用示例

ForeachBinaryOp 为图内部融合算子，通过 GE IR 图模式构图调用。示例代码参见 [examples/arch35/test_geir_foreach_binary_op.cpp](../examples/arch35/test_geir_foreach_binary_op.cpp)，核心构图片段如下：

```Cpp
auto op1 = op::ForeachBinaryOp("foreachBinaryOp1");
const int N = 2;                  // 列表内 Tensor 个数
op1.set_attr_op_code(0);          // 0=add, 1=sub, 2=mul, 3=div
op1.create_dynamic_input_x1(N);
for (int i = 0; i < N; ++i) { op1.set_dynamic_input_x1(i, dataX1[i]); }
op1.create_dynamic_input_x2(N);
for (int i = 0; i < N; ++i) { op1.set_dynamic_input_x2(i, dataX2[i]); }
op1.create_dynamic_output_y(N);
```
