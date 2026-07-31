# SwigluClamp

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

SwigluClamp 激活函数,SwiGlu 家族变体。将 silu + clamp + mul 融合为单个算子,用于 Step-3.7-Flash 等 MoE 模型的 FFN 专家层。

对输入 `x` shape `[..., 2N]`,沿末维切 gate(前 N)/ up(后 N):

$$
gate = x[..., :N], \quad up = x[..., N:]
$$

$$
out = silu(gate).clamp(max=limit) \times up.clamp(min=-limit, max=limit)
$$

其中 $silu(g) = g \times sigmoid(g)$。**关键:gate 路 silu 在 clamp 之前**(SwigluClamp 顺序),区别于 clamp-then-silu 的 SwigluOAI / clipped_swiglu。

## 参数说明

|参数|输入/输出/属性|描述|数据类型|数据格式|
|:---|:---|:---|:---|:---|
|x|输入|输入张量,shape `[..., 2N]`,末维为偶数|FLOAT16 / FLOAT / BFLOAT16|ND|
|limit|可选属性|门限值,必须 > 0,默认 7.0|FLOAT|-|
|y|输出|输出张量,shape `[..., N]`(末维为 x 一半)|FLOAT16 / FLOAT / BFLOAT16|ND|

## 约束说明

- `x` 末维必须为偶数(2N)。
- `limit` 必须 > 0。
- 输入不支持包含 ±inf 或 nan。

## 调用说明

|调用方式|调用样例|说明|
|:---|:---|:---|
|aclnn 调用|./examples/test_aclnn_swiglu_clamp.cpp|通过 aclnnSwigluClamp 接口调用|
|PTA 调用|torch.ops.cann_ops_nn.swiglu_clamp(x, limit)|通过 torch_extension(PyTorch Adapter)调用|
