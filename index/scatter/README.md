# Scatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：
  将tensor `updates`中的值按指定的轴`axis`和索引`indices`逐个更新tensor `data`中的值。

- 示例：
  该算子有3个输入和一个属性：`data`、`updates`、`indices`和`axis`，其中`data`是待更新的tensor，`updates`是存储更新数据的tensor，`indices`表示更新位置，
  `axis`是指定的更新维度。按`indices`为1维或2维划分，存在以下两种场景（`indices`为0维时等价于shape为（1）的1维场景，此时`updates`的0轴必须为1）：

  **场景一：** `indices`为1维。`indices`的第i个元素对应`updates`的第i个batch（`updates`的0轴），表示该batch写入`data`的第i个batch时在`axis`维上的起始偏移；
  `updates`在`axis`维的大小s为每次写入的长度，需满足`indices[i] + s`不超过`data`在`axis`维的大小。

  ```text
  样例输入：
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i] + e <= d
  indices:(n,), n <= a
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim = -2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim = -1

  **场景二：** `indices`为2维，shape的1轴为2。`indices`的第i行对应`updates`的第i个batch：`indices[i][0]`指定该batch写入`data`的0轴位置（不同i可以写入`data`的不同batch），
  `indices[i][1]`指定在`axis`维上的起始偏移；`updates`在`axis`维的大小s为每次写入的长度，需满足`indices[i][1] + s`不超过`data`在`axis`维的大小。

  ```text
  样例输入：
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i][1] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i][1] + e <= d
  indices:(n, 2), indices[i][0] < a
  ```

      data[indices[i][0]][j][indices[i][1]+k][l] = updates[i][j][k][l] # if dim = -2
      data[indices[i][0]][j][k][indices[i][1]+l] = updates[i][j][k][l] # if dim = -1

## 约束说明

- `updates` shape的0轴与`indices` shape的0轴一致。
- `indices`为0维时，`updates` shape的0轴为1。
- `updates` shape的0轴小于等于`data` shape的0轴。
- `updates`与`data`的shape，除`axis`轴和0轴以外，其余轴的shape均相同。
- 当`indices` shape为二维时，shape的1轴需要等于2。
- `indices`数据类型为INT32时，DtypeSize=4，为INT64时，DtypeSize=8，IndicesShapeSize为`indices`的shape乘积，需要使用的ub = IndicesShapeSize * DtypeSize + 224，当ub大于对应可以获取到的AI处理器版本总ub大小时，不支持。
- 当`indices`有重复时，重复位置的结果不保证。
- 确定性计算：当`indices`存在重复值时，结果将是不确定的。若开启了确定性计算，可保证结果的确定性（仅Ascend 950PR/Ascend 950DT需要显式开启，其余支持型号默认确定性）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_inplace_scatter_update](example/test_aclnn_inplace_scatter_update.cpp) |通过[aclnnInplaceScatterUpdate](docs/aclnnInplaceScatterUpdate.md)接口方式调用Scatter算子。 |
