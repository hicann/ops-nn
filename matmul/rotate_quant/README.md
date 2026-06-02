# RotateQuant

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    ×     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- 算子功能：对张量x进行旋转变换，再执行对称动态量化。
- 计算公式：
    1.  旋转变换

        $$
        Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
        $$

        其中：$\mathbf{x} \in \mathbb{R}^{m \times n}$，$\mathbf{Y} \in \mathbb{R}^{m \times n}$，$\mathbf{rotation} \in \mathbb{R}^{k \times k}$。

    2.  对称动态量化（pertoken 逐行量化）
        - 缩放因子计算（逐行计算）

            $$
            s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
            $$

            其中：$s_i$ 是第 $i$ 行的缩放因子；$C_{\text{MAX}}$ 是量化范围最大值，int8 取 127，quint4x2 取 7。
        - 量化计算

            $$
            y_{i,j} = \frac{Y_{i,j}}{s_i}
            $$
            

## 参数说明
  
  <table style="undefined;table-layout: fixed; width: 962px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待旋转量化的输入张量。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rotation</td>
      <td>输入</td>
      <td>旋转矩阵。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>clamp需要限制的范围的缩放系数。</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>量化后的输出张量。</td>
      <td>INT8、INT32、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>动态量化计算出的缩放系数。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当yOut的输出类型为int8时，y的shape必须和x相同(M, N)。
  - 当yOut的输出类型为int32时，y的shape必须为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(M)。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rotate_quant](examples/test_aclnn_rotate_quant.cpp) | 通过[aclnnRotateQuant](docs/aclnnRotateQuant.md)等方式调用RotateQuant算子。 |
