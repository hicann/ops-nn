# Scatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term> Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能:
  将tensor updates中的值按指定的轴axis和索引indices逐个更新tensor data中的值。该算子为自定义算子语义，无对应的tensorflow或pytorch接口。

- 示例：
  该算子有3个输入和一个属性：data，updates，indices和axis，其中data是待更新的tensor，updates是存储更新数据的tensor，indices表示更新位置，
  axis是指定的更新维度。当indices为1维，存在以下两种场景：

  **场景一：** indices为1维，axis指定更新的维度shape为1，indices指定的是每个batch维度（最高维）在axis维度的偏移。

  ```
  样例输入：
  data:(a, b, c, d)
  updates:(a, b, 1, d)
  indices:(a,)
  axis = -2
  ```

      data[i][j][indices[i]][k] = updates[i][j][0][k] # if dim=-2
      data[i][j][k][indices[i]] = updates[i][j][k][0] # if dim=-1

  **场景二：** indices为1维，axis指定更新的维度shape大于1，indices指定的是每个batch维度（最高维）在axis维度的偏移。

  ```
  样例输入：
  data:(a, b, c, d)
  updates:(a, b, e, d), indices[i] + e <= c
  indices:(a,)
  axis = -2 or 2
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim=-2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim=-1


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_scatter](example/test_aclnn_scatter.cpp) | 通过[aclnnInplaceScatterUpdate](docs/aclnnInplaceScatterUpdate.md)接口方式调用Scatter算子。 |
