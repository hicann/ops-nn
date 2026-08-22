# InplaceAdd

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    √     |
| <term>Atlas 推理系列产品</term>                          |    √     |
| <term>Atlas 训练系列产品</term>                          |    √     |

## 功能说明

- 算子功能：根据`indices`指定的第0维位置，将`v`中的切片加到`x`的对应切片，并原地更新`x`。输出`y`与`x`共享存储。
- 使用场景：兼容TensorFlow的`InplaceAdd`图算子，用于按行执行索引更新；本算子不是普通的逐元素或Broadcast Add。
- 计算公式：设`x`的shape为$[N, d_1, \dots, d_{D-1}]$，`indices`的shape为$[K]$，`v`的shape为$[K, d_1, \dots, d_{D-1}]$。先把索引按第0维大小`N`做数学模归一：

  $$
  row_j = ((indices_j \bmod N) + N) \bmod N , \quad j = 0, 1, \dots, K-1
  $$

  再在归一化后的行上累加，未被索引命中的行原样透传：

  $$
  y_{i} =
  \begin{cases}
  x_{i} + v_{j} & \exists j,\ row_j = i \\
  x_{i} & 其他
  \end{cases}
  $$

  其中$i$遍历第0维，$x_i$、$y_i$、$v_j$均表示对应下标上的整个尾部切片。上述数值定义要求归一化后的目标行互不重复，如果多个索引归一化到同一行，输出内容未定义。

- 示例：

  ```text
  x       = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
  indices = [0, 2]
  v       = [[1, 1, 1],
             [2, 2, 2]]

  y       = [[2, 3, 4],
             [4, 5, 6],
             [9, 10, 11]]
  ```

## 参数说明

<table style="table-layout: fixed; width: 1460px"><colgroup>
  <col style="width: 90px">
  <col style="width: 140px">
  <col style="width: 280px">
  <col style="width: 350px">
  <col style="width: 480px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>Shape/支持维度</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>被原地更新的输入张量。支持空Tensor：第0维为0，或任一尾维为0；第0维为0时要求indices与v的第0维同时为0。</td>
      <td>[N, d1, ..., d(D-1)]，支持1～8维。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>指定x第0维上需要更新位置的索引张量。支持空Tensor：长度为0时不执行更新，y与x一致。</td>
      <td>[K]，仅支持1维。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输入</td>
      <td>更新值张量，数据类型必须与x相同。支持空Tensor：第0维随indices长度，尾维随x。</td>
      <td>[K, d1, ..., d(D-1)]，支持1～8维，维度数必须与x相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>更新后的张量，数据类型与x相同，并与x共享存储。支持空Tensor：x为空Tensor时y为空Tensor，不执行计算。</td>
      <td>与x相同，支持1～8维。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR/Ascend 950DT</term>：采用原生AI Core实现，`x`、`v`和`y`支持参数说明中列出的全部13种数据类型。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：未提供原生AI Core实现。GE图模式下，`AInplaceAddFusionPass`将本算子改写为`TensorMove`和`ScatterAdd`执行；`x`、`v`和`y`支持FLOAT16、FLOAT、INT32。
- <term>Atlas 训练系列产品</term>：未提供原生AI Core实现。GE图模式下，`AInplaceAddFusionPass`将本算子改写为`TensorMove`和`ScatterAdd`执行；`x`、`v`和`y`支持FLOAT、INT32。

## 约束说明

- `x`、`v`、`y`支持1～8维，`indices`仅支持一维；`v`的维度数必须与`x`相同。
- `v.shape[0]`必须等于`indices.shape[0]`；`v.shape[1:]`必须等于`x.shape[1:]`。
- `x`、`v`、`y`的数据类型必须一致，`indices`的数据类型必须为INT32。
- 当`x.shape[0] > 0`时，索引先扩展为INT64，再按`x`的第0维大小进行数学模归一，因此支持负索引和超范围正索引。
- 当归一化后的索引存在重复值时，输出内容未定义，不保证累加顺序、确定性或具体数值。
- 当`indices`为空时，不执行更新，输出保持与`x`一致。
- 当`x.shape[0] == 0`时，`indices`和`v`的第0维必须同时为0。
- 当`x`的任一尾维为0时，输出为空，不执行索引取模或数据读写。
- `x.shape[0]`与`indices`长度不超过INT32最大值（2147483647）；尾维大小与总元素数不受该限制，按INT64校验，仅在乘积溢出INT64时报错。
- 整数加法不提供饱和运算保证，调用方不应依赖整数溢出后的结果。
- 支持静态Shape、动态Shape和动态Rank；实际shape必须满足上述约束。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。现有同名`aclnnInplaceAdd`执行的是逐元素或Broadcast计算`self += alpha * other`，与本算子的按索引行更新语义不同，不能混用。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| GE图模式 | [test_geir_inplace_add](examples/test_geir_inplace_add.cpp) | 通过[算子IR](op_graph/inplace_add_proto.h)构建InplaceAdd算子图并执行RunGraph验证；样例使用INT64验证原生Ascend 950 binary。 |
