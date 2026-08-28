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

- 算子功能：根据`indices`指定的第0维位置，将`v`中的切片加到`x`的对应切片，并将更新结果输出到`y`。接口不声明`x`与`y`共享存储。
- 使用场景：兼容TensorFlow的`InplaceAdd`图算子，用于按行执行索引更新；本算子不是普通的逐元素或Broadcast Add。
- 计算公式：设`x`的shape为$[N, d_1, \dots, d_{D-1}]$，`indices`的shape为$[K]$，`v`的shape为$[K, d_1, \dots, d_{D-1}]$。记$row_j$为第$j$个更新对应的实际目标行，其确定规则见“产品差异说明”。在目标行互不重复的前提下，计算公式如下：

  $$
  y_{i} =
  \begin{cases}
  x_{i} + v_{j} & \exists j,\ row_j = i \\
  x_{i} & 其他
  \end{cases}
  $$

  其中$i$遍历第0维，$x_i$、$y_i$、$v_j$均表示对应下标上的整个尾部切片。如果多个索引对应同一目标行，输出内容未定义。

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
      <th>shape/支持维度</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入<code>x</code>，作为更新基值。</td>
      <td>[N, d1, ..., d(D-1)]，支持1～8维。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>公式中的输入<code>indices</code>，指定<code>x</code>第0维上需要更新的位置。</td>
      <td>[K]，仅支持1维。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输入</td>
      <td>公式中的输入<code>v</code>，表示加到目标行上的更新值，数据类型必须与<code>x</code>相同。</td>
      <td>[K, d1, ..., d(D-1)]，支持1～8维，维度数必须与x相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出<code>y</code>，shape和数据类型与<code>x</code>相同；接口不声明与<code>x</code>共享存储。</td>
      <td>与x相同，支持1～8维。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

### 产品差异说明

各支持产品的输入和输出均使用ND格式，`indices`的数据类型均为INT32。不同产品的能力子集如下：

| 产品 | 参数或场景 | `x`、`v`、`y`数据类型 | 静态shape能力 | 动态shape能力 | 索引、空Tensor及规模限制 |
| :--- | :----------- | :-------------------- | :------------ | :------------ | :----------------------- |
| <term>Ascend 950PR/Ascend 950DT</term> | `indices`为运行期输入或编译期常量 | FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64 | 输入ND->输出ND；`x`、`v`、`y`支持1～8维，`indices`仅支持1维。 | 输入ND->输出ND；支持动态shape和动态Rank。 | 当$N>0$时，按$row_j=((indices_j \bmod N)+N)\bmod N$确定目标行。支持$K=0$；当$N=0$时要求$K=0$，当任一尾维为0时输出为空。$N$和$K$不超过INT32最大值；尾维大小、张量元素数、合计处理规模及字节大小必须在INT64可表示范围内。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br><term>Atlas 200I/500 A2 推理产品</term><br><term>Atlas 推理系列产品</term> | `indices`为运行期输入或编译期常量 | FLOAT16、FLOAT、INT32 | 输入ND->输出ND；`x`、`v`、`y`支持1～8维，`indices`仅支持1维。 | 输入ND->输出ND；支持动态shape和动态Rank。 | 要求$0\leq indices_j<N$；不支持空Tensor。 |
| <term>Atlas 训练系列产品</term> | `x`、`v`、`y`为FLOAT或INT32，`indices`为运行期输入或编译期常量 | FLOAT、INT32 | 输入ND->输出ND；`x`、`v`、`y`支持1～8维，`indices`仅支持1维。 | 输入ND->输出ND；支持动态shape和动态Rank。 | 要求$0\leq indices_j<N$；不支持空Tensor。 |
| <term>Atlas 训练系列产品</term> | `x`、`v`、`y`为FLOAT16，且`indices`在编译期为常量 | FLOAT16 | 仅支持静态shape，输入ND->输出ND；`x`、`v`、`y`支持1～8维，`indices`仅支持1维。 | 不支持。 | 要求$0\leq indices_j<N$；不支持空Tensor。 |

## 约束说明

- `v.shape[0]`必须等于`indices.shape[0]`；`v.shape[1:]`必须等于`x.shape[1:]`。
- `x`、`v`、`y`的数据类型必须一致，`indices`的数据类型必须为INT32。
- 当多个索引对应同一目标行时，输出内容未定义，不保证累加顺序、确定性或具体数值。
- 整数加法不提供饱和运算保证，调用方不应依赖整数溢出后的结果。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。现有同名`aclnnInplaceAdd`执行的是逐元素或Broadcast计算`self += alpha * other`，与本算子的按索引行更新语义不同，不能混用。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| GE图模式 | [test_geir_inplace_add](examples/test_geir_inplace_add.cpp) | 通过[算子IR](op_graph/inplace_add_proto.h)构建InplaceAdd算子图并执行RunGraph与数值校验；样例覆盖FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32。 |
