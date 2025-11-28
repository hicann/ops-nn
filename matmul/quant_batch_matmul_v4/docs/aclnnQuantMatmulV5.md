# aclnnQuantMatmulV5

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>昇腾910_95 AI处理器</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×    |

## 功能说明

- 接口功能：完成量化的矩阵乘计算。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：兼容aclnnQuantMatmulV3、aclnnQuantMatmulV4接口功能。完成量化的矩阵乘计算，最小支持输入维度为1维，最大支持输入维度为2维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）。

- 计算公式：
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64，yOffset为FLOAT32：

      $$
      out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
      $$

    - 无x1Scale无bias：
      $$
      out = x1@x2 * x2Scale + x2Offset
      $$

    - bias INT32：
      $$
      out = (x1@x2 + bias) * x2Scale + x2Offset
      $$

    - bias BFLOAT16/FLOAT32（此场景无offset）：
      $$
      out = x1@x2 * x2Scale + bias
      $$

    - x1Scale无bias：
      $$
      out = x1@x2 * x2Scale * x1Scale
      $$

    - x1Scale， bias INT32（此场景无offset）：
      $$
      out = (x1@x2 + bias) * x2Scale * x1Scale
      $$

    - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无offset）：
      $$
      out = x1@x2 * scale * x1Scale + bias
      $$
    
    - x1，x2为INT8，x1Scale, x2Scale为FLOAT32，bias为FLOAT32，out为FLOAT16/BFLOAT16  (pertoken-pergroup量化):
      $$
      out = (x1 @ x2) * x1Scale * x2Scale + bias
      $$

    - x1，x2为INT4，x1Scale, x2Scale为FLOAT32，x2Offset为FLOAT16, out为FLOAT16/BFLOAT16 (pertoken-pergroup非对称量化):
      $$
      out = x1Scale * x2Scale @ (x1 @ x2 - x1 @ x2Offset)
      $$

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulV5GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulV5”接口执行计算。

```c++
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScale,
  const aclTensor *x1Offset,
  const aclTensor *x2Offset,
  const aclTensor *yOffset,
  const aclTensor *bias,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```
```c++
aclnnStatus aclnnQuantMatmulV5(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulV5GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
 <col style="width: 198px">
 <col style="width: 121px">
 <col style="width: 220px">
 <col style="width: 397px">
 <col style="width: 220px">
 <col style="width: 115px">
 <col style="width: 138px">
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
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>-</td>
      <td>INT4、INT8、INT32</td>
      <td>ND</td>
      <td>2-6</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>
        <ul>
          <li>AI处理器亲和数据排布格式。</li>
          <li>ND格式下支持最后两根轴转置情况下的非连续tensor，其他场景的<a href="../../../docs/zh/context/非连续的Tensor.md">非连续的Tensor</a>不支持。</li>
          <li>仅最后m和k轴转置情况下支持<a href="../../../docs/zh/context/非连续的Tensor.md">非连续的Tensor</a>，其他轴方向不支持非连续的Tensor。</li>
        </ul>
      </td>
      <td>INT4、INT8、INT32</td>
      <td>ND</td>
      <td>2-6</td>
      <td>x</td>
    </tr>
    <tr>
      <td>x1Scale</td>
      <td>输入</td>
      <td>公式中的输入x1Scale。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-6</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2Scale</td>
      <td>输入</td>
      <td>表示量化参数，公式中的输入x2Scale。</td>
      <td>-</td>
      <td>UINT64、INT64、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>1-6</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yScale</td>
      <td>输入</td>
      <td>输出y的反量化scale参数。</td>
      <td>预留参数，当前版本不支持，需要传入nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x1Offset</td>
      <td>输入</td>
      <td>公式中的输入x1Offset。</td>
      <td>预留参数，当前版本不支持，需要传入nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2Offset</td>
      <td>输入</td>
      <td>公式中的输入x2Offset。</td>
      <td>shape是1维（t, ），t = 1或n，其中n与x2的n一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset</td>
      <td>输入</td>
      <td>公式中的输入yOffset。</td>
      <td>预留参数，当前版本不支持，需要传入nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>表示x1的输入shape是否包含transpose。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>表示x2的输入shape是否包含transpose。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupSize</td>
      <td>输入</td>
      <td>用于输入m、n、k方向上的量化分组大小。</td>
      <td>由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式见表格下方公式一。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>-</td>
      <td>FLOAT16、INT8、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 </tbody></table>
 
  - 公式一：

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **返回值：**


  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x1、x2、x1Scale、x2Scale、yOffset或out是空指针。</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>x1、x2、bias、x1Scale、x2Scale、x2Offset或out的shape不满足校验条件。</td>
      </tr>
      <tr>
        <td>x1、x2、bias、x1Scale、x2Scale、x2Offset或out的数据类型和数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>x1、x2、bias、x2Scale、x2Offset或out是空tensor。</td>
      </tr>
      <tr>
        <td>x1与x2的最后一维大小超过65535，x1的最后一维指transposeX1为true时的m或transposeX1为false时的k，x2的最后一维指transposeX2为true时的k或transposeX2为false时的n。</td>
      </tr>
      <tr>
        <td>传入的groupSize不满足校验条件，或传入的groupSize为0时， x1、x2与x1Scale，x2Scale的shape关系无法推断groupSize。</td>
      </tr>
    </tbody></table>


## aclnnQuantMatmulV5

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
    </colgroup>
    <thead>
      <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV5GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

输入和输出支持以下数据类型组合：
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT32                     | FLOAT32     | UINT64          | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
  | INT8                      | INT8                      | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
  | INT8                      | INT8                      | null        | UINT64/INT64    | null/FLOAT32| null     | null/INT32   | null       | INT8                                   |
  | INT8                      | INT8                      | null/FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                  |
  | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16| null        | null     | null/INT32   | null       | INT32                                |
  | INT8                      | INT8                      | FLOAT32        | FLOAT32| null        | null     | FLOAT32   | null       | BFLOAT16                                |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4                | INT4                | FLOAT32     | FLOAT32         | FLOAT16        | null     | null    | null       | BFLOAT16               |

不同的[量化模式](../../../docs/zh/context/量化介绍.md)支持的x1、 x2、x1Scale和x2Scale的输入dtype组合以及支持的平台为：
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  x1、x2、x1Scale、x2Scale、yOffset和groupSize在不同量化场景下dtype、shape的取值等方面相互影响，关系如下：
    | x1数据类型                 | x2数据类型                 | x1Scale数据类型| x2Scale数据类型| x1 shape | x2 shape| x1Scale shape| x2Scale shape| yOffset shape| [groupSizeM，groupSizeN，groupSizeK]取值|
    | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------------ | ------------ | ------------ |
    | INT8                    |INT32                   |FLOAT32              |UINT64             | （m, k）|（k, n // 8）|（m, 1）|（（k // 256），n）| (n) | [0, 0, 256]|
    | INT8                    |INT8                   |FLOAT32              |FLOAT32             | （m, k）|（n, k）|（m, k // 128）|（（k // 128），（n // 128））| null | [1, 128, 128]|
    | INT4                    |INT4                   |FLOAT32              |FLOAT32             | （m, k）|（n, k）|（m, 1）| (k // 128, n) | (k // 128, n) | [0, 0, 256]|
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - x1数据类型支持INT8、INT32、INT4。
    - 当数据类型为INT32、INT4时，为INT4量化场景、transposeX1为false情况。
    - 当数据类型为INT4时，维度为：（batch，m，k），要求k为偶数。
    - 当数据类型为INT32时，每个INT32数据存放8个INT4数据，对应维度表示：（batch，m，k // 8），要求k为8的倍数。
    - 启用A8W8 perblock对称量化时，数据类型支持INT8，目前n需与256对齐, k与128对齐且为4 * 128的倍数，transposeX1为false，形状为(m, k)。
    - 启用A4W4 pergroup非对称量化时，数据类型支持INT4，目前k需与1024对齐，transposeX1为false，形状为(m, k)。
    - 在transposeX1为false情况下，形状为（batch, m, k）。
    - 在transposeX1为true情况下，形状为（batch, k, m），其中batch代表前0~4维，0维表示bacth不存在。
    - AI处理器亲和数据排布格式下，shape支持4~8维。
    - transposeX2为true时维度为：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32， n0 = 16， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。
    - transposeX2为false时维度为：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1。
    - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到AI处理器亲和数据排布格式的转换。
  - x2数据类型支持INT8、INT32、INT4。
    - 当数据类型为INT32、INT4时，为INT4量化场景，当前仅支持2维ND格式。ND格式下，shape支持2~6维，
      - transposeX2为false情况下各个维度表示：（batch，k，n）。
      - transposeX2为true情况下各个维度表示：（batch，n，k）。
      - batch可不存在，其中k与x1的shape中的k一致。
    - 数据类型为INT4时：
      - transposeX2为true时维度为：（n，k），要求k为偶数。
      - transposeX2为false时维度为：（k，n），要求n为偶数。
    - 数据类型为INT32时，每个INT32数据存放8个INT4数据，
      - transposeX2为true时维度为：（n，k // 8），要求k为8的倍数。
      - transposeX2为false时维度为：（k，n // 8），要求n为8的倍数。
    - 可使用aclnnConvertWeightToINT4Pack接口完成x2从INT32（1个int32在0~3bit位存储1个int4）到INT32（1个int32存储8个int4）或INT4（1个int4表示1个int4）的数据格式转换，具体参见[aclnnConvertWeightToINT4Pack接口](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。
    - 启用A8W8 perblock对称量化时，数据类型支持INT8，transposeX2为true，形状为(n, k)，目前n需与256对齐，k与128对齐且为4 * 128的倍数。
    - 启用A4W4 pergroup非对称量化时，数据类型支持INT4，transposeX2为true，形状为(n, k)，目前k需与1024对称，目前n需与256对齐。
  - x1Scale约束如下：
    - shape支持2维，形状为（m，1）。数据类型支持FLOAT32。
    - 启用A8W8 perblock对称量化时，数据类型为float32, 形状为(m, ceil(k / 128))。
    - 启用A4W4 pergroup非对称量化时，数据类型为float32, 形状为(m, 1)。
  - x2Scale的约束如下：
    - shape支持2维，形状为（k / groupSize，n）其中n与x2的n一致。
    - 数据类型支持UINT64、INT64、FLOAT32、BFLOAT16。
    - 当原始输入类型不满足[约束说明](#约束说明)中类型组合时，由于TransQuantParamV2只支持1维，需要将x2_scale view成一维(k / groupSize * n)，再调用TransQuantParamV2算子的aclnn接口来将x2Scale转成UINT64数据类型，再将输出view成二维（k / groupSize, n）。
    - 启用A8W8 perblock对称量化时，数据类型为float32, 当x2的transpose为true时，形状为(ceil(n / 128), ceil(k / 128))，当x2的transpose为false时，形状为(ceil(k / 128), ceil(n / 128))。
    - 启用A4W4 pergroup非对称量化时，数据类型为float32，当x2的transpose为false时，形状为(ceil(k / 256), n)。
  - 当前版本不支持yScale，需要传入nullptr。
  - x2Offset的约束如下：
    - 可选量化参数，数据类型支持FLOAT32。
    - 当out数据类型为INT8时，offset可以存在，其他输入类型需要传入nullptr。
    - A4W4 pergroup非对称量化时: 支持输入类型为FLOAT16，形状为（ceil(k, 256)，n）。
  - bias的约束如下：
    - 当前版本不支持，需要传入nullptr。
    - A8W8 perblock对称量化时，数据类型支持float32，shape支持一维(n, )。
  - transposeX1：x1和x2为INT32、INT4时，transposeX1仅支持false，各个维度表示：（m, k）。
  - transposeX2的约束如下：
    - ND格式下，为false时维度为：（batch，k，n），为true时维度为：（batch，n，k），batch可不存在，其中k与x1的shape中的k一致。
    - AI处理器亲和数据排布格式下：
      - 为true时维度为：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32，n0 = 16，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。
      - 为false时维度为：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1。
  - groupSize的约束如下：
    - 常规情况下以及A4W4 pergroup非对称量化模式时，[groupSizeM，groupSizeN，groupSizeK]取值组合仅支持仅支持[0, 0, 256]，即groupSizeK仅支持256。
    - 在A8W8 perblock对称量化模式时，[groupSizeM，groupSizeN，groupSizeK]取值组合仅支持[1, 128, 128]。
  - out的约束如下：
    - shape支持2维，（m，n）。数据类型支持FLOAT16、INT8、BFLOAT16、INT32。
    - A8W8 perblock对称量化模式时，目前输出支持bfloat16。
    - A4W4 pergroup非对称量化模式时，目前输出支持bfloat16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), ans);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // 实际上是float16半精度方式

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yoffset aclTensor
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```