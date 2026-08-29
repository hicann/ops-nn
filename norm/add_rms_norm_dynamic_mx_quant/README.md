# AddRmsNormDynamicMxQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：将Add算子、RmsNorm归一化（相比LayerNorm算子去掉了减去均值的部分）与DynamicMxQuant动态MX量化融合为一次计算，减少搬入搬出操作：先对输入做加法，再做RmsNorm归一化，最后在输入尾轴上按blocksize=32分组进行动态MX量化，输出量化结果y、加法结果x、量化尺度mxscale及标准差倒数rstd。mxscale的计算算法由scale_alg指定（dst_type为FP8时支持OCP与cuBLAS两种实现，FP4时仅支持OCP），具体计算过程见下方计算公式。
- V2接口（CANN >= 9.2.0）：新增可选输入x3，当x3不为空时，计算公式变为 x = (x3 + x1) + x2，支持三路残差加法。详见[aclnnAddRmsNormDynamicMxQuantV2](docs/aclnnAddRmsNormDynamicMxQuantV2.md)。
- 计算公式：

  当x3为空时（V1接口或V2接口x3传空）：
  $$
  x=x_{1}+x_{2}
  $$

  当x3不为空时（仅V2接口）：
  $$
  x=x_{3}+x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  当scale_alg为0时：

  - 将RmsNorm输出y在尾轴维度上按k = 32个数分组，一组k个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \quad i = 1, 2, \ldots, k
    $$

  - emax: 对应数据类型的最大正规数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  当scale_alg为1时，只涉及FP8类型：
  - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。
  - 找到该块中数值的最大绝对值：

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

  - 将FP32映射到目标数据类型FP8可表示的范围内：

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

  - 转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
  - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
  - 为保证量化时不溢出，对指数进行向上取整：

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$

  - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
  - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - 应用到量化的最终步骤：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的x1。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的x2。shape和数据类型需要与x1一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的gamma。shape需要与x1最后一维一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，对应公式中的beta。shape必须与gamma一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>可选输入</td>
      <td>表示加法计算中的可选输入，残差路径，对应公式中的x3。shape和数据类型需要与x1一致；不传入时计算x = x1 + x2。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到分母中的值，以确保数值稳定。对应公式中的epsilon。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td><ul><li>表示mxscale的计算方法，对应公式中的scale_alg。</li><li>支持取值0和1，取值为0表示Open Compute Project(OCP)实现，取值为1表示cuBLAS实现。当dst_type为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li><li>默认值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示数据转换的模式，对应公式中的round_mode。</li><li>当dst_type为35/36时，仅支持{"rint"}。</li><li>当dst_type为40/41时，支持{"rint", "floor", "round"}。</li><li>默认值为"rint"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示指定数据转换后y的类型，对应公式中的DType。</li><li>输入范围为{35, 36, 40, 41}，分别对应{FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2}。</li><li>默认值为40。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_rstd</td>
      <td>可选属性</td>
      <td><ul><li>表示指定是否输出有效的rstd_out。</li><li>支持True和False。</li><li>默认值为False。</li><li>当output_rstd为False时，rstd为无效占位输出。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示归一化并量化后的结果，对应公式中的Pi和di，shape与x1一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td><ul><li>表示x1和x2的和，对应公式中的x。shape和数据类型与x1一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>输出</td>
      <td><ul><li>表示每个分组对应的量化尺度，对应公式中的mxscale和Sb，shape见约束说明。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td><ul><li>表示归一化后的标准差的倒数，对应公式中Rms(x)的倒数。</li><li>当output_rstd为True时，维度数与x1保持一致，不需要norm的维度（x1的维度减去gamma的维度）与x1对应维度一致，需要norm的维度为1。如x1 shape为(2,3,4,8)、gamma shape为(8,)时，rstd shape为(2,3,4,1)。</li><li>当output_rstd为False时，rstd为无效占位输出。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：

  mxscale的shape约束说明如下：
  - rank(mxscale) = rank(x1) + 1。
  - mxscale.shape[-2] = ceil(ceil(x1.shape[-1] / 32) / 2)，即x1尾轴按blocksize=32分组后的块数向上取整到偶数，再除以2。
  - mxscale.shape[-1] = 2。
  - 其他维度与输入x1一致。

- 当输出y的数据类型为FLOAT4_E2M1或FLOAT4_E1M2，x1尾轴的值必须为偶数。

- 输入gamma、可选输入beta的数据类型只能和x1的数据类型保持一致或者为FLOAT32。

- **边界值场景说明**
  - 当输入是Inf时：1、输出y为0；2、输出x为Inf；3、输出mxscale为255，偶数pad填充值为0；4、输出rstd为0。
  - 当输入是NaN时：1、输出y为0；2、输出x为Nan；3、输出mxscale为255，偶数pad填充值为0；4、输出rstd为NaN。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_rms_norm_dynamic_mx_quant](examples/test_aclnn_add_rms_norm_dynamic_mx_quant.cpp) | 通过[aclnnAddRmsNormDynamicMxQuant](docs/aclnnAddRmsNormDynamicMxQuant.md)接口方式调用AddRmsNormDynamicMxQuant算子。 |
| aclnnV2接口  | [test_aclnn_add_rms_norm_dynamic_mx_quant_v2](examples/test_aclnn_add_rms_norm_dynamic_mx_quant_v2.cpp) | 通过[aclnnAddRmsNormDynamicMxQuantV2](docs/aclnnAddRmsNormDynamicMxQuantV2.md)接口方式调用，支持可选输入x3。 |
| GE图模式 | -  | 通过[算子IR](op_graph/add_rms_norm_dynamic_mx_quant_proto.h)构图方式调用AddRmsNormDynamicMxQuant算子。         |
| PyTorch API | [test_torch_extension](tests/ut/torch_extension/test_torch_extension.py) | 通过[cann_ops_nn.add_rms_norm_dynamic_quant](docs/torchapi_add_rms_norm_dynamic_quant.md)接口调用AddRmsNormDynamicMxQuant算子，需安装`cann_ops_nn` whl包，支持单算子模式和图模式（torch.compile + torchair），接口使用方式详见链接文档。 |

## Torch Extension 使用说明

通过 torch_extension（`cann_ops_nn`，接口名 `add_rms_norm_dynamic_quant`）调用算子时，部分输出的数据类型和形状与 ACLNN/GE 层不同，需要特别关注。接口的完整参数说明、返回值说明与调用示例见[torchapi_add_rms_norm_dynamic_quant](docs/torchapi_add_rms_norm_dynamic_quant.md)，本节仅对输出差异点做详细解释。

### 问题背景

ACLNN 层和 GE 图算子层面，FP4 输出的 `aclDataType` 为 `DT_FLOAT4_E2M1` 或 `DT_FLOAT4_E1M2`，每个 FP4 值占据完整字节（未打包），输出 shape 与输入 x1 相同。

但 PyTorch 没有原生 FP4 dtype，无法直接表达 FP4 tensor。因此 torch_extension 层需要将两个 FP4 值打包为一个 `uint8`（每 4 bit 存一个 FP4），输出 shape 的尾轴减半。

### C++ 层打包实现（csrc）

`torch_extension/cann_ops_nn/ops/norm/add_rms_norm_dynamic_quant/csrc/add_rms_norm_dynamic_quant.cpp`（位于 ops-nn 仓库根目录）中的处理：

1. **分配输出 tensor，FP4 时 shape 减半**：

   ```cpp
   auto y_shape = x1.sizes().vec();
   if (IsFp4DstType(dst_type)) {
       y_shape.back() /= 2;  // 两个 FP4 打包为一个 uint8
   }
   at::ScalarType y_scalar = GetScalarTypeFromDstType(dst_type);  // FP4 时为 at::kByte
   at::Tensor y = at::empty(y_shape, x1.options().dtype(y_scalar));
   ```

2. **TensorWrapper 覆盖 aclDataType**：

   ```cpp
   TensorWrapper y_wrapper = {y, y_acltype};  // 将 y 的 aclDataType 覆盖为 dst_type 对应类型
   ```

   `TensorWrapper` 将 y 的 aclDataType 覆盖为 dst_type 对应的 ACL 类型（FP4 时为 `ACL_FLOAT4_E2M1`/`ACL_FLOAT4_E1M2`），使 ACLNN 在 `uint8` 内存上按 FP4 位格式写入数据，实现"底层 FP4 存储、上层 uint8 表达"。

### 输出对照表

| dst_type | ACLNN/GE 输出 dtype | ACLNN/GE 输出 shape | torch_extension 输出 dtype | torch_extension 输出 shape |
|----------|---------------------|---------------------|---------------------------|---------------------------|
| 40 (FP4_E2M1) | DT_FLOAT4_E2M1 | 与 x1 相同 | **uint8** | **x1.shape[-1] / 2** |
| 41 (FP4_E1M2) | DT_FLOAT4_E1M2 | 与 x1 相同 | **uint8** | **x1.shape[-1] / 2** |
| 36 (FP8_E4M3FN) | DT_FLOAT8_E4M3FN | 与 x1 相同 | float8_e4m3fn | 与 x1 相同 |
| 35 (FP8_E5M2) | DT_FLOAT8_E5M2 | 与 x1 相同 | float8_e5m2 | 与 x1 相同 |

> **注意**：FP8 输出使用 PyTorch 原生 `float8_e4m3fn` / `float8_e5m2` dtype，无需打包，shape 与输入一致。仅 FP4 需要打包为 uint8。

### mxscale 输出说明

`mxscale` 的 dtype 在 ACLNN/GE 层为 `DT_FLOAT8_E8M0`，torch_extension 层的表达方式为：

- 单算子模式（csrc）：使用 PyTorch 原生 `float8_e8m0fnu` dtype 直接表达；
- 图模式（torchair）：`graph_convert` 中将 mxscale 的 GE dtype 置为 `DT_UINT8`，以 `uint8` 表达。

两者字节布局完全一致（每个元素 1 字节，E8M0 编码），可通过 `view(torch.uint8)` 互转。

### 图模式 FP4 打包后处理（graph_convert）

图模式（`torch.compile` + `torchair`）通过 `graph_convert` 中的 `_pack_fp4_output_to_uint8` 函数处理 FP4 输出打包：

- GE 算子的 infershape 将 y 的 shape 设为与 x1 相同（未打包，R 个 FP4 元素）。
- `graph_convert` 在 GE 算子输出后链式添加三个 GE 图算子完成打包：
  1. `Reshape`：(…, R) → (…, R//2, 2)
  2. `Bitcast`：DT_FLOAT4 → DT_UINT8
  3. `Reshape`：(…, R//2, 2) → (…, R//2)
- 该实现参照 `swiglu_group_quant` 的 graph_convert 中的同类处理。

对于 `output_rstd=False` 场景，`graph_convert` 始终向 GE 算子传递 `output_rstd=True`（确保 rstd 输出 shape 正确），rstd tensor 照常返回但包含未定义数据，与单算子模式（csrc）行为一致。

### 使用建议

| 调用方式 | FP4 (dst_type=40/41) | FP8 (dst_type=35/36) | output_rstd=False |
|----------|----------------------|----------------------|-------------------|
| 单算子模式（`torch.ops`） | 完全支持 | 完全支持 | 完全支持 |
| 图模式（`torch.compile`） | 完全支持 | 完全支持 | 完全支持 |
