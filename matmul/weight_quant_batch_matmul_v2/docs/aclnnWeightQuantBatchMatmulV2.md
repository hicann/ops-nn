# aclnnWeightQuantBatchMatmulV2

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |


## 功能说明

- **接口功能**：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  当需要对输出进行量化处理时，其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当不需要对输出再进行量化操作时，其计算公式为

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnWeightQuantBatchMatmulV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnWeightQuantBatchMatmulV2”接口执行计算。

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
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
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th><a href="../../../docs/zh/context/数据格式.md" target="_blank">数据格式</a></th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th><a href="../../../docs/zh/context/非连续的Tensor.md" target="_blank">非连续的Tensor</a></th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>矩阵乘的左输入矩阵，公式中的输入`x`，device侧的aclTensor。</td>
        <td></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>维度支持两维，shape支持(m, k)，m表示矩阵第1维的大小，k表示矩阵的第2维的大小，其中Reduce维度k需要与`weight`的Reduce维度k大小相等。</td>
        <td>非连续的Tensor仅支持转置场景。</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>矩阵乘的右输入矩阵，公式中的输入`weight`，device侧的aclTensor。</td>
        <td></td>
        <td>INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、INT32、FLOAT、FLOAT4_E2M1</td>
        <td>ND、FRACTAL_NZ</td>
        <td>支持(k, n)，其中k表示矩阵第1维的大小，n表示矩阵第2维的大小。k、n大小在[1, 65535]范围内。k, n要求32B对齐。其中若数据类型为INT4或FLOAT4_E2M1，则weight的内轴应为偶数。</td>
        <td>非连续的Tensor仅支持转置场景。</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>输入</td>
        <td>实现输入反量化计算的反量化scale参数，反量化公式中的输入`antiquantScale`。</td>
        <td></td>
        <td>FLOAT16、BFLOAT16或FLOAT8_E8M0（仅涉及mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>），非mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，数据类型要求和输入`x`的数据类型保持一致。</td>
        <td>ND</td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>输入</td>
        <td>实现输入反量化计算的反量化offset参数，反量化公式中的`antiquantOffset`，device侧的aclTensor。</td>
        <td>可选输入, 当不需要时为空指针。</td>
        <td></td>
        <td>ND</td>
        <td>要求与`antiquantScale`一致。</td>
        <td></td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>输入</td>
        <td>实现输出量化计算的量化参数，device侧的aclTensor。</td>
        <td>由量化公式中的`quantScale`和`quantOffset`的数据通过`aclnnTransQuantParam`接口转化得到。</td>
        <td></td>
        <td>ND</td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>输入</td>
        <td>实现输出量化计算的量化offset参数，量化公式中的`quantOffset`，device侧的aclTensor。</td>
        <td></td>
        <td></td>
        <td>ND</td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>偏置输入，公式中的`bias`，device侧的aclTensor。</td>
        <td>可选输入, 当不需要时为空指针。</td>
        <td></td>
        <td>ND</td>
        <td></td>
        <td>不支持</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>输入</td>
        <td>表示在伪量化pergroup和mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，对输入`weight`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。当伪量化算法不为pergroup和mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>时传入0；当伪量化算法为pergroup<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>时传入值的范围为[32, k-1]且值要求是32的倍数；在mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>，仅支持32。</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>计算输出，公式中的`y`，device侧的aclTensor。</td>
        <td></td>
        <td></td>
        <td>ND</td>
        <td></td>
        <td>不支持</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>

  - x（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`x`矩阵为非转置时，m大小在[1, 2^31-1]范围内；转置时，m大小在[1, 65535]范围内。
    - <term>Ascend 950PR/Ascend 950DT</term>：`x`矩阵m大小在[1, 2^31-1]范围内，不支持转置场景。

  - weight（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：维度支持2维，Reduce维度k需要与`x`的Reduce维度k大小相等。数据类型支持INT8、INT4、INT32，当`weight`[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ且数据类型为INT4或INT32时，或者当`weight`[数据格式](../../../docs/zh/context/数据格式.md)为ND且数据类型为INT32时，仅在INT4Pack场景支持，需配合`aclnnConvertWeightToINT4Pack`接口完成从INT32到INT4Pack的转换，以及从ND到FRACTAL_NZ的转换，[详情可参考样例](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)，若数据类型为INT4，则`weight`的内轴应为偶数。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。shape支持(k, n)，其中k表示矩阵第1维的大小，n表示矩阵第2维的大小。
      对于不同伪量化算法模式，`weight`的[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ仅在如下场景下支持:
      - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：
        - `weight`的数据类型为INT8，y的数据类型为非INT8。
        - `weight`的数据类型为INT4/INT32，`weight`转置，y的数据类型为非INT8。
      - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32，`weight`非转置，`x`非转置，antiquantGroupSize为64或128，k为antiquantGroupSize对齐，n为64对齐，y的数据类型为非INT8。
    - <term>Ascend 950PR/Ascend 950DT</term>：维度支持2维，Reduce维度k需要与`x`的Reduce维度k大小相等。数据类型支持INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、INT32、FLOAT、FLOAT4_E2M1。shape支持(k, n)，其中k表示矩阵第1维的大小，n表示矩阵第2维的大小。k、n大小在[1, 65535]范围内。k、n要求32B对齐。其中若数据类型为INT8，FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8，则k、n大小在[1,2^31-1]范围内，k、n不要求32B对齐。其中若数据类型为INT4或FLOAT4_E2M1，则weight的内轴应为偶数。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。
        - 数据类型为INT32/FLOAT时，必须配合aclnnConvertWeightToINT4Pack接口完成从INT32/FLOAT到紧密排布的INT4/FLOAT4_E2M1的转换，[详情可参考样例](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。
        - 数据类型为FLOAT/FLOAT4_E2M1时，仅支持pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)，pergroup[量化模式](../../../docs/zh/context/量化介绍.md)下，要求`antiquantScale`与`x`的数据类型相同，mx[量化模式](../../../docs/zh/context/量化介绍.md)下，要求`antiquantScale`类型为FLOAT8_E8M0类型，且不支持antiquantOffsetOptional输入，且antiquantGroupSize仅支持32。
        - 数据类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，只支持perchannel[量化模式](../../../docs/zh/context/量化介绍.md)。

      对于不同伪量化算法模式，`weight`的[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ仅在如下场景下支持:
      - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32，`weight`非转置，`x`非转置，y的数据类型为非INT8。
      - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32/FLOAT4_E2M1/FLOAT，`weight`非转置，`x`非转置，k为64对齐，n为64对齐，y的数据类型为非INT8。
      - mx[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为FLOAT4_E2M1/FLOAT，`weight`非转置，`x`非转置，k为64对齐，n为64对齐，y的数据类型为非INT8。

  - antiquantScale（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、UINT64、INT64（当FLOAT16、BFLOAT16时，数据类型要求和输入`x`保持一致；当为UINT64、INT64时，`x`仅支持FLOAT16，不转置，`weight`仅支持INT8，ND转置，模式仅支持perchannel[量化模式](../../../docs/zh/context/量化介绍.md)，quantScaleOptional和quantOffsetOptional必须传入空指针，m仅支持[1, 96]，k和n要求64对齐，需要首先配合aclnnCast接口完成FLOAT16到FLOAT32的转换，详情请参考[Cast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md)，再配合aclnnTransQuantParamV2接口完成FLOAT32到UINT64的转换，详情请参考[TransQuantParamV2](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)）。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。
      对于不同伪量化算法模式，`antiquantScale`支持的shape如下:
      - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1,)或(1, 1)。
      - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
      - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16或FLOAT8_E8M0（仅涉及mx[量化模式](../../../docs/zh/context/量化介绍.md)），非mx[量化模式](../../../docs/zh/context/量化介绍.md)下，数据类型要求和输入`x`的数据类型保持一致。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景并且连续性要求和weight保持一致。
       对于不同伪量化算法模式，`antiquantScale`支持的shape如下:
        - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：(1,)或(1,1)。
        - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
        - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小。
        - mx[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小，仅支持32。

  - antiquantOffsetOptional（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、INT32，数据类型为FLOAT16、BFLOAT16时，数据类型要求和输入`x`的数据类型保持一致；数据类型为INT32类型时，数据范围限制为[-128, 127]，x仅支持FLOAT16，weight仅支持INT8，`antiquantScale`仅支持UINT64/INT64。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16，数据类型要求和输入`x`的数据类型保持一致。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景并且连续性要求和weight保持一致。当x是FLOAT16或者BFLOAT16，同时weight是FLOAT、FLOAT4_E2M1、FLOAT8_E5M2、FLOAT8_E4M3FN或者HIFLOAT8时，不支持该参数，填空指针。

  - quantScaleOptional（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。可选输入，当不需要时为空指针；对于不同的伪量化算法模式，支持的shape如下:
      - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1,)或(1, 1)。
      - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
    - <term>Ascend 950PR/Ascend 950DT</term>：预留参数，暂未使用，固定传入空指针。

  - quantOffsetOptional（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。可选输入, 当不需要时为空指针；存在时shape要求与`quantScaleOptional`一致。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Ascend 950PR/Ascend 950DT</term>：预留参数，暂未使用，固定传入空指针。

  - biasOptional（aclTensor *, 计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：维度支持1维或2维，shape支持(n,)或(1, n)。数据类型支持FLOAT16、FLOAT。当`x`的数据类型为BFLOAT16时，本参数要求为FLOAT；当`x`的数据类型为FLOAT16时，本参数要求为FLOAT16。
    - <term>Ascend 950PR/Ascend 950DT</term>：维度支持1维或2维，shape支持(n,)或(1, n)。数据类型支持FLOAT16、FLOAT、BFLOAT16。当`x`的数据类型为BFLOAT16时，本参数要求为FLOAT或BFLOAT16；当`x`的数据类型为FLOAT16时，本参数要求为FLOAT16。当`x`的数据类型为BFLOAT16时，同时`weight`类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，本参数要求为BFLOAT16
  
  - antiquantGroupSize（int, 计算输入）：表示在伪量化pergroup和mx[量化模式](../../../docs/zh/context/量化介绍.md)下，对输入`weight`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。当伪量化算法不为pergroup和mx[量化模式](../../../docs/zh/context/量化介绍.md)时传入0；当伪量化算法为pergroup[量化模式](../../../docs/zh/context/量化介绍.md)时传入值的范围为[32, k-1]且值要求是32的倍数；在mx[量化模式](../../../docs/zh/context/量化介绍.md)，仅支持32。

  - y（aclTensor *, 计算输出）：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：维度支持2维，shape支持(m, n)。数据类型支持FLOAT16、BFLOAT16、INT8。当`quantScaleOptional`存在时，数据类型为INT8；当`quantScaleOptional`不存在时，数据类型支持FLOAT16、BFLOAT16，且与输入`x`的数据类型一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：维度支持2维，shape支持(m, n)。数据类型支持FLOAT16、BFLOAT16。当`quantScaleOptional`不存在时，与输入`x`的数据类型一致。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1030px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 650px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的shape维度不符合要求。
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的数据类型不在支持的范围之内。
- x、weight的reduce维度(k)不相等。
- antiquantOffsetOptional存在输入时，shape与antiquantScale不相同。
- quantOffsetOptional存在输入时，shape与quantScale不相同。
- x的shape不符合要求。
- biasOptional的shape不符合要求。
- antiquantGroupSize值不符合要求。
- quantOffsetOptional存在时，quantScaleOptional是空指针。
- m、n、k取值不在支持范围内。
- 不支持空tensor场景。
- 输入tensor的数据格式不在支持范围内。
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的连续性不符合要求。
- x为bfloat16，weight为float4_e2m1或者float32时，bias数据类型只支持bfloat16。
        </td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>产品型号不支持。</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV2

- **参数说明**
  <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口`aclnnWeightQuantBatchMatmulV2GetWorkspaceSize`获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明
- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnWeightQuantBatchMatmulV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
  - <term>Ascend 950PR/Ascend 950DT</term>: aclnnWeightQuantBatchMatmulV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

性能优化建议：
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
  - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：推荐使用非转置的weight输入。
  - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。m范围为[65, 96]时，推荐使用数据类型为UINT64或INT64的antiquantScale。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
  - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
  - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)：推荐使用非转置的`weight`输入。

## 调用示例
  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {16, 32};
  std::vector<int64_t> weightShape = {32, 16};
  std::vector<int64_t> yShape = {16, 16};
  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* y = nullptr;
  std::vector<float> xHostData(512, 1);
  std::vector<int8_t> weightHostData(512, 1);
  std::vector<float> yHostData(256, 0);

  std::vector<int64_t> antiquantScaleShape = {16};
  void* antiquantScaleDeviceAddr = nullptr;
  aclTensor* antiquantScale = nullptr;
  std::vector<float> antiquantScaleHostData(16, 1);

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建antiquantScale aclTensor
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建xFp16 aclTensor
  void* xFp16DeviceAddr = nullptr;
  aclTensor* xFp16 = nullptr;
  ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建antiquantScale aclTensor
  void* antiquantScaleFp16DeviceAddr = nullptr;
  aclTensor* antiquantScaleFp16 = nullptr;
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建yFp16 aclTensor
  void* yFp16DeviceAddr = nullptr;
  aclTensor* yFp16 = nullptr;
  ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  void* workspaceAddr = nullptr;

  // 调用cast生成FP16的输入
  ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnWeightQuantBatchMatmulV2第一段接口
  ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnWeightQuantBatchMatmulV2第二段接口
  ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
  ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(antiquantScale);
  aclDestroyTensor(y);
  aclDestroyTensor(xFp16);
  aclDestroyTensor(antiquantScaleFp16);
  aclDestroyTensor(yFp16);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(antiquantScaleDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(xFp16DeviceAddr);
  aclrtFree(antiquantScaleFp16DeviceAddr);
  aclrtFree(yFp16DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
