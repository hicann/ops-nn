# aclnnQuantMatmulActivationQuant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：融合MX（Microscaling，微缩放）量化矩阵乘、激活和动态MX量化，输出量化结果及对应的缩放因子（scale）。支持GELU（Gaussian Error Linear Unit，高斯误差线性单元）的tanh/erf形式、SiLU（Sigmoid Linear Unit，Sigmoid线性单元）及SwiGLU（Swish Gated Linear Unit，Swish门控线性单元）；x1、x2均为ND格式。数据类型及转置组合见[约束说明](#约束说明)。

<!-- npu="950" id7 -->

- 计算流程：先完成MX量化矩阵乘及偏置相加，再计算激活，最后执行动态MX量化。

以下公式省略batch下标，$X_1$、$X_2$表示按转置状态解释后的矩阵，形状分别为$(M,K)$、$(K,N)$。M为左矩阵的行数，K为公共归约维度，N为右矩阵的列数。

### MX量化矩阵乘

沿K轴每32个元素划为一组，每组矩阵乘结果乘以对应的输入scale，再与偏置累加，得到$C$：

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

其中，$g_M$、$g_N$、$g_K$分别为M、N、K方向的分组大小，当前固定为$(1,1,32)$；$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$。$S_1$、$S_2$分别表示`x1ScaleOptional`、`x2Scale`中第$j$组的scale，实际存储形状见[形状](#形状)。矩阵乘采用FP32累加，偏置在激活前相加；未传入偏置时，$b_n=0$。

### 激活

对矩阵乘结果$C$应用指定的激活函数，得到$A$：

- `activationType="gelu_tanh"`：GELU的tanh近似。

  $$
  A=\frac{C}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(C+0.044715C^3\right)\right)\right]
  $$

- `activationType="gelu_erf"`：GELU的erf形式。

  $$
  A=\frac{C}{2}\left[1+\operatorname{erf}\left(\frac{C}{\sqrt{2}}\right)\right]
  $$

- `activationType="silu"`：逐元素计算SiLU。

  $$
  A=\operatorname{SiLU}(C)=\frac{C}{1+e^{-C}}
  $$

- `activationType="swiglu"`：将$C$沿最后一维等分为左半部分$G$和右半部分$U$，对$G$计算SiLU后与$U$逐元素相乘。

  $$
  C=[G\mid U],\qquad A=\operatorname{SiLU}(G)\odot U=\frac{G}{1+e^{-G}}\odot U
  $$

上述激活均在FP32中计算。记激活后的输出宽度为$H$：GELU和SiLU的$H=N$，SwiGLU的$H=N/2$。

### 动态MX量化

将激活结果按最近偶数舍入模式（Round to Nearest, ties to Even，RNE）转换为BF16，即舍入到最近的可表示值，中间值取尾数最低位为0的值，再沿最后一维每32个元素划为一组。最后一组不足32个元素时补0参与scale计算，补齐元素不写入输出。

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

每组共用一个scale $s$，按`roundMode`转换为输出数据类型$D$：

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{为FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{为FP4}
\end{cases},\qquad (yOut)_i=\operatorname{Cast}_{D,\,\mathrm{roundMode}}(Q_i)
$$

`yScaleOut`使用FLOAT8_E8M0存储scale。第$j$组写入`yScaleOut[..., m, j // 2, j % 2]`，输出形状为`(..., M, ceil(H/64), 2)`。

#### scale的计算方法

以下公式适用于$a>0$且有限的情况，令$p=\lfloor\log_2a\rfloor$。

- `scaleAlg=0`：OCP（Open Compute Project，开放计算项目）共享指数算法，支持FP8和FP4。

  $$
  e=\max(-127,p-e_{\max})
  $$

  FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1的$e_{\max}$分别为8、15、2。

- `scaleAlg=1`：按目标类型的最大有限值计算scale，仅支持FP8。

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/F_{\max})\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

  FLOAT8_E4M3FN、FLOAT8_E5M2的$F_{\max}$分别为448、57344。

- `scaleAlg=2`：按`dstTypeMax`指定的范围计算scale，仅支持FP4。令$d=\operatorname{FP32}(\mathrm{dstTypeMax})$，传入0时取$d=6$。

  当$d=6$或$d=7$时：

  $$
  e=\max\left(-127,\begin{cases}
  p-1, & a>d\cdot2^{p-2}\\
  p-2, & a\le d\cdot2^{p-2}
  \end{cases}\right)
  $$

  当$d$为$[6,12]$内的其他值时：

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/d)\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

E8M0以字节值$e+127$编码$2^e$。特殊情况如下：

- 全零组的量化结果为0，scale的编码为0；输出scale中无对应数据分组的补齐位置也填入编码0。该编码表示$2^{-127}$。
- $S=0$时取$e=-127$。算法0，以及算法2中$d=6$或$d=7$的分支，在scale编码为0时将该组量化结果置0。
- 含NaN或Inf的组，scale的编码为255（NaN）。

<!-- end id7 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantMatmulActivationQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulActivationQuant”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  const aclTensor* x1ScaleOptional,
  const aclTensor* x2Scale,
  const aclTensor* biasOptional,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  char*            activationType,
  char*            quantMode,
  char*            roundMode,
  int64_t          scaleAlg,
  double           dstTypeMax,
  aclTensor*       yOut,
  aclTensor*       yScaleOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantMatmulActivationQuant(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulActivationQuantGetWorkspaceSize

- **参数说明**

  所有Tensor的数据均位于Device侧；除`biasOptional`可为nullptr外，Tensor参数均须提供且不能为空Tensor。

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1（const aclTensor*）</td>
        <td>输入</td>
        <td>矩阵乘的左矩阵，对应公式中的X<sub>1</sub>。</td>
        <td><ul> <li>不可为nullptr，不支持空Tensor。</li> <li>shape及转置限制见<a href="#形状">形状</a>和<a href="#数据类型与激活">数据类型与激活</a>。</li> </ul></td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2～6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2（const aclTensor*）</td>
        <td>输入</td>
        <td>矩阵乘的右矩阵，对应公式中的X<sub>2</sub>。</td>
        <td><ul> <li>不可为nullptr，不支持空Tensor。</li> <li>逻辑shape见<a href="#形状">形状</a>，存储要求见<a href="#布局与存储">布局与存储</a>。</li> </ul></td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2～6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1ScaleOptional（const aclTensor*）</td>
        <td>输入</td>
        <td>x1沿K轴的分组scale，对应公式中的S<sub>1</sub>。</td>
        <td><ul> <li>本接口仅支持MX模式，该输入必须提供，不可为nullptr。</li> <li>不支持空Tensor，batch维与x1一致。</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3～7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale（const aclTensor*）</td>
        <td>输入</td>
        <td>x2沿K轴的分组scale，对应公式中的S<sub>2</sub>。</td>
        <td><ul> <li>不可为nullptr，不支持空Tensor。</li> <li>batch维与x2一致。</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3～7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>在激活前相加的偏置，对应公式中的b。</td>
        <td><ul> <li>shape为(N,)；三维输出还支持(B,1,N)。</li> <li>不使用时传nullptr；传入时不支持空Tensor。</li> </ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1或3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>transposeX1（bool）</td>
        <td>输入</td>
        <td>是否转置x1的最后两维。</td>
        <td><ul> <li>false表示不转置，true表示转置。</li> <li>支持的类型与激活组合见<a href="#数据类型与激活">数据类型与激活</a>。</li> <li>false表示不转置，true表示转置；可用组合见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2（bool）</td>
        <td>输入</td>
        <td>是否转置x2的最后两维。</td>
        <td><ul> <li>false表示不转置，true表示转置。</li> <li>false表示不转置，true表示转置；可用组合见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize（int64_t）</td>
        <td>输入</td>
        <td>输入量化在M、N、K方向的分组大小编码。</td>
        <td><ul> <li>仅支持0或4295032864，均表示[1,1,32]。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activationType（char*）</td>
        <td>输入</td>
        <td>激活类型。</td>
        <td><ul> <li>支持"gelu_tanh"、"gelu_erf"、"silu"、"swiglu"。</li> <li>不可为nullptr。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode（char*）</td>
        <td>输入</td>
        <td>量化模式。</td>
        <td><ul> <li>仅支持"mx"，不可为nullptr。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>roundMode（char*）</td>
        <td>输入</td>
        <td>输出量化的舍入模式。</td>
        <td><ul> <li>FP8仅支持"rint"，表示舍入到最近的可表示值，中间值取尾数最低位为0的值。</li> <li>FP4还支持"floor"（向负无穷方向舍入）、"round"（舍入到最近的可表示值，中间值向远离0的方向舍入）。</li> <li>不可为nullptr。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg（int64_t）</td>
        <td>输入</td>
        <td>输出scale的计算方法。</td>
        <td><ul> <li>FP8支持0、1，FP4支持0、2。</li> <li>0为OCP共享指数算法，1按FP8最大有限值计算，2按dstTypeMax指定的范围计算。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax（double）</td>
        <td>输入</td>
        <td>scaleAlg=2时计算输出scale所用的参考上限，对应公式中的d。</td>
        <td><ul> <li>仅scaleAlg=2时生效：支持0.0（等价于6.0）或[6.0,12.0]内的有限值。</li> <li>其他算法忽略该参数。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut（aclTensor*）</td>
        <td>输出</td>
        <td>动态量化结果，对应公式中的yOut。</td>
        <td><ul> <li>不可为nullptr，不支持空Tensor。</li> <li>类型须与x1一致，shape为(batch,M,H)。</li> </ul></td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND（兼容格式见<a href="#布局与存储">布局与存储</a>）</td>
        <td>2～6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScaleOut（aclTensor*）</td>
        <td>输出</td>
        <td>输出的分组scale，对应公式中的s。</td>
        <td><ul> <li>不可为nullptr，不支持空Tensor。</li> <li>shape为(batch,M,ceil(H/64),2)。</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND（兼容格式见<a href="#布局与存储">布局与存储</a>）</td>
        <td>3～7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
        <td><ul> <li>不可为nullptr。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td><ul> <li>不可为nullptr。</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  `groupSize`的非零值按`(groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`编码，高16位为0；参与移位的分组值使用int64_t类型。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码/状态码</th>
        <th>码值</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1、x2、x1ScaleOptional、x2Scale、yOut、yScaleOut、workspaceSize或executor为空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>Tensor的数据类型、格式、形状、广播关系、stride或访问范围不满足约束；或字符串参数为空指针、属性取值及组合不合法。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建算子执行器失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>内部Tensor创建、预处理、算子计算或结果拷贝失败，返回空指针。</td>
      </tr>
    </tbody>
  </table>

## aclnnQuantMatmulActivationQuant

- **参数说明**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace（void*）</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t）</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，单位为字节，由第一段接口aclnnQuantMatmulActivationQuantGetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor*）</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream（aclrtStream）</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  执行器运行失败时返回ACLNN_ERR_INNER（561000）。

## 约束说明

- x1、x2的逻辑维数为2～6，各维度必须为已知正整数。
- x1、x2的各逻辑维度及广播后的batch维乘积不超过$2^{32}-1$，激活前输出元素数不超过$2^{63}-1$。

### 数据类型与激活

输入scale和输出scale均为FLOAT8_E8M0。biasOptional传入时的数据类型须为FLOAT32；不使用偏置时传入nullptr。矩阵及输出支持以下组合：

| x1 | x2 | yOut |
| --- | --- | --- |
| FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E4M3FN |
| FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E5M2 |
| FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E4M3FN |
| FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E5M2 |
| FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT4_E2M1 |

下表中，FP8指FLOAT8_E4M3FN或FLOAT8_E5M2，FP4指FLOAT4_E2M1；同时须满足上方的数据类型组合要求。

| activationType | 数据类型 | 输出宽度H | N的要求 | transposeX1 |
| --- | --- | --- | --- | --- |
| "gelu_tanh"、"gelu_erf" | FP8 | N | N > 0 | false、true |
| "gelu_tanh"、"gelu_erf" | FP4 | N | 正偶数 | false、true |
| "silu" | FP8 | N | N > 0 | false、true |
| "swiglu" | FP8 | N/2 | 64的正整数倍 | false、true |

`transposeX2`均支持false、true。FP4仅支持GELU，且K > 2、N为偶数。FP4输入x1和x2在连续存储布局下的最内轴须为偶数：x1不转置时检查K，转置时检查M；x2不转置时检查N，转置时检查K。

### 形状

`batch1`、`batch2`表示各输入前0～4个batch维，`batch`表示右对齐广播后的batch维；没有batch维时省略该部分。令`K64=ceil(K/64)`，其中ceil表示向上取整。以下均为逻辑形状，FP4的维度按4位元素计数：

| 张量 | 不转置 | 转置 |
| --- | --- | --- |
| x1 | `(batch1, M, K)` | `(batch1, K, M)` |
| x1ScaleOptional | `(batch1, M, K64, 2)` | `(batch1, K64, M, 2)` |
| x2 | `(batch2, K, N)` | `(batch2, N, K)` |
| x2Scale | `(batch2, K64, N, 2)` | `(batch2, N, K64, 2)` |

- 两个输入的K必须相等；输入scale的batch维必须与对应矩阵逐维一致。
- 每32个K元素对应一个输入scale，相邻两组scale按顺序存放在长度为2的末维中。
- `yOut`的形状为`(batch, M, H)`，`yScaleOut`为`(batch, M, ceil(H/64), 2)`，H由激活类型决定。
- `biasOptional`的形状为`(N,)`；仅当输出为三维时，还支持`(B, 1, N)`，B须等于广播后的batch大小。SwiGLU的biasOptional长度仍为激活前的N。

### 布局与存储

- yOut和yScaleOut支持ND及与ND兼容的NCHW、NHWC、HWCN、NCDHW、NDHWC、NCL基础格式，不支持私有格式；输出shape仍须满足上述约定。
- ND输入和输出的stride、offset须非负，shape、stride和offset确定的全部访问位置须位于storage内。支持满足上述条件的非零offset切片；FP4的offset以逻辑元素计数，须为偶数。
- x1、x2、两个输入scale和biasOptional支持[非连续Tensor](../../../docs/zh/context/non_contiguous_tensor.md)。输出支持非连续Tensor。
- 以末两轴的转置视图传入矩阵时，配套scale须交换倒数第3、倒数第2轴，末维2保持不变。若同时使用转置视图并将对应的transposeX1/transposeX2设为true，两次转置相互抵消；上表的转置限制按最终有效转置状态判断。

### 确定性/Batch一致性

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 确定性说明：aclnnQuantMatmulActivationQuant默认确定性实现。
  - Batch一致性说明：默认[Batch一致性](../../../docs/zh/context/batch_consistency.md)实现：在相同运行环境下，保持N、K、数据类型、布局及激活/量化属性不变，且同一Token的x1、x1ScaleOptional及其对应的x2、x2Scale、biasOptional数据不变时，仅改变batch大小、M大小、Token的位置或组合，该Token对应的yOut和yScaleOut保持逐比特一致，无需额外开启开关。
<!-- end id8 -->

## 调用示例

以下示例使用FLOAT8_E4M3FN输入、GELU（tanh）激活和`scaleAlg=0`。

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

其他示例见[GELU样例](../examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp)和[SwiGLU样例](../examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp)。

```cpp
// ND weights: QuantMatmul + GELU(tanh) + MXFP8 quantization.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_activation_quant.h"
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
int Init(int32_t deviceId, aclrtStream* stream)
{
    // 初始化运行时资源。
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return 0;
}
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnQuantMatmulGeluMxQuantTest(aclrtStream stream)
{
    int ret = ACL_SUCCESS;
    // 2. 构造FLOAT8_E4M3FN输入、E8M0 scale和输出。
    int64_t m = 5;
    int64_t k = 64;
    int64_t n = 128;
    bool transposeX1 = false;
    bool transposeX2 = false;
    constexpr int64_t MX_SCALE_BLOCK_SIZE = 64; // scale块对齐大小，用于计算scale shape
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {k, n};
    std::vector<int64_t> x1ScaleShape = {m, (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
    std::vector<int64_t> x2ScaleShape = {(k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, n, 2};
    std::vector<int64_t> outShape = {m, n};
    std::vector<int64_t> outScaleShape = {m, (n + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* outScaleDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* outScale = nullptr;
    std::vector<uint8_t> x1HostData(m * k, 0b00111000); // float8_e4m3的1.0
    std::vector<uint8_t> x2HostData(k * n, 0b00111000); // float8_e4m3的1.0
    std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape),
                                         0b01111111); // float8_e8m0的1.0
    std::vector<uint8_t> x2ScaleHostData(GetShapeSize(x2ScaleShape),
                                         0b01111111); // float8_e8m0的1.0
    std::vector<uint8_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<uint8_t> outScaleHostData(GetShapeSize(outScaleShape), 0);
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x1Scale aclTensor
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2Scale aclTensor
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale aclTensor
    ret = CreateAclTensor(outScaleHostData, outScaleShape, &outScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &outScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outScaleTensorPtr(outScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用算子的两段式接口。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int64_t groupSizeValue = 4295032864;
    char activationType[] = "gelu_tanh";
    char quantMode[] = "mx";
    char roundMode[] = "rint";
    ret = aclnnQuantMatmulActivationQuantGetWorkspaceSize(x1, x2, x1Scale, x2Scale, bias, transposeX1, transposeX2,
                                                          groupSizeValue, activationType, quantMode, roundMode, 0, 0.0,
                                                          out, outScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnQuantMatmulActivationQuant第二段接口
    ret = aclnnQuantMatmulActivationQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuant failed. ERROR: %d\n", ret); return ret);
    // 4. 同步等待任务执行结束。
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 将量化结果及scale的原始字节拷贝回Host。
    auto size = GetShapeSize(outShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%lld] (FP8 raw byte): %u\n", static_cast<long long>(i), static_cast<unsigned>(resultData[i]));
    }
    size = GetShapeSize(outScaleShape);
    std::vector<uint8_t> scaleData(size, 0);
    ret = aclrtMemcpy(scaleData.data(), scaleData.size() * sizeof(scaleData[0]), outScaleDeviceAddr,
                      size * sizeof(scaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("scale[%lld] (E8M0 raw byte): %u\n", static_cast<long long>(i), static_cast<unsigned>(scaleData[i]));
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. 初始化Device和Stream。
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AclnnQuantMatmulGeluMxQuantTest(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("AclnnQuantMatmulGeluMxQuantTest failed. ERROR: %d\n", ret);
    }
    Finalize(deviceId, stream);
    return ret;
}
```
