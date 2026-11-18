# AscendDequant

## 产品支持情况

|产品|是否支持|
|:----------|:----------:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3系列产品</term>|×|
|<term>Atlas A2系列产品</term>|×|
|<term>Atlas 200I/500 A2推理产品</term>|×|
|<term>Atlas推理系列产品</term>|×|
|<term>Atlas训练系列产品</term>|×|

## 功能说明

- 算子功能：对INT32输入`x`按反量化系数`deq_scale`逐元素计算，输出FLOAT16或FLOAT32结果`y`，`deq_scale`支持右对齐广播。

- 计算公式：

    $$
    s = reinterpret\_f32(low\_32\_bits(deq\_scale))
    $$

    $$
    acc = cast\_to\_fp32(x) \times s
    $$

    $$
    acc = relu\_flag\ ?\ \max(acc, 0) : acc
    $$

    $$
    acc = sqrt\_mode\ ?\ acc \times s : acc
    $$

    $$
    y = cast\_to\_dst\_type(acc)
    $$

  说明：

  - `deq_scale`为UINT64时，低32位承载FLOAT32系数的IEEE-754位模式，高32位不参与计算。
  - `sqrt_mode`为true时在结果上再乘一次同一系数`s`，即乘`s`的平方，与开方无关。
  - `relu_flag`的ReLU作用于两次乘法之间。
  - 输出FLOAT16时只做一次舍入（RINT）；输出FLOAT32时不做窄化转换。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待反量化的输入张量，对应公式中的`x`。支持1~8维，shape与输出`y`一致，支持空Tensor（任一维为0）。 | INT32 | ND |
| deq_scale | 输入 | 反量化系数，对应公式中的`s`。形状需可按右对齐广播到`x`，即秩不大于`x`的秩，且右对齐后每一维等于`x`的对应维或为1。 | UINT64 | ND |
| sqrt_mode | 可选属性 | 取true时在结果上再乘一次同一系数`s`；默认false。 | BOOL | - |
| relu_flag | 可选属性 | 取true时在两次乘法之间取`max(acc, 0)`；默认false。 | BOOL | - |
| dtype | 可选属性 | 输出数据类型，0表示FLOAT32，1表示FLOAT16；默认1，需与输出`y`的数据类型一致。 | INT | - |
| y | 输出 | 反量化结果，对应公式中的`y`。shape与输入`x`完全一致。 | FLOAT16、FLOAT32 | ND |

## 数据类型组合

| x | deq_scale | y | dtype |
| :--- | :--- | :--- | :--- |
| INT32 | UINT64 | FLOAT16 | 1 |
| INT32 | UINT64 | FLOAT32 | 0 |

## 约束说明

- `deq_scale`采用右对齐广播语义（与numpy广播一致）：秩不超过`x`，右对齐后每一维要么等于`x`的对应维，要么为1；不满足时报shape相关错误。
- 输出`y`的shape必须与输入`x`完全一致，数据类型必须与属性`dtype`一致。
- 支持空Tensor：`x`任一维为0时不启动计算（`usedCoreNum=0`、blockDim=1，kernel直接返回）。
- 算子为纯逐元素计算，按ND位置语义处理，不做format相关的布局推导。算子定义中注册的NHWC、NCHW、NCDHW、NDHWC格式仅为接口注册，实际计算与ND一致。
- `sqrt_mode`与`deq_scale`的数据类型无互斥限制，UINT64场景同样支持`sqrt_mode=true`。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过[算子IR](./op_graph/ascend_dequant_proto.h)构图方式调用AscendDequant算子。 |

本算子不提供aclnn接口（`ACLNNTYPE aclnn_exclude`），当前仅支持GE图模式调用。
