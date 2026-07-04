# Index

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：根据 `indices` 从输入张量 `x` 中取出指定位置的数据，生成输出张量 `y`。
- 接口形态：当前实验目录提供 `aclnnIndexGetWorkspaceSize` / `aclnnIndex` 接口，内部将输入和索引转为连续张量后执行 Index 计算。
- 执行路径：普通 `INT32` / `INT64` 索引走 `IndexAiCore`；BOOL 索引、索引数量超过输入 rank 等特殊路径走 `IndexAiCpu` 或直接返回。

## 参数说明

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 340px">
  <col style="width: 350px">
  <col style="width: 120px">
  <col style="width: 130px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待索引的输入张量，对应 aclnn 接口中的 self。</td>
      <td>FLOAT、DOUBLE、FLOAT16、BFLOAT16、INT16、INT32、INT64、INT8、UINT8、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>indexed_sizes</td>
      <td>输入</td>
      <td>内部元数据，记录被索引维度信息，由 aclnn 接口根据 self 和 indices 生成。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>indexed_strides</td>
      <td>输入</td>
      <td>内部元数据，记录输入张量 stride 或 BOOL 索引输出形状信息，由 aclnn 接口生成。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>动态输入</td>
      <td>索引张量列表。普通 AICore 路径支持 INT32、INT64；aclnn 参数检查接受 BOOL，BOOL 索引走 AICPU 路径。</td>
      <td>INT32、INT64、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>根据索引取出后的输出张量，数据类型需与 x 一致。</td>
      <td>FLOAT、DOUBLE、FLOAT16、BFLOAT16、INT16、INT32、INT64、INT8、UINT8、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 仅支持 ND 格式，不支持私有格式输入、索引或输出。
- `x`、`indices` 中每个索引张量、`y` 的 rank 均不超过 8。
- 普通索引路径要求 `indices` 数量在 `[1, x rank]` 范围内，且各索引张量 shape 可广播；输出 shape 需与接口推导结果一致。
- 空 Tensor 和标量输入的单索引场景由 aclnn 接口做特殊处理。
- AICore 二进制当前配置覆盖 `ascend910b` 和 `ascend950`。

## 调用说明

| 调用方式 | 接口/文件 | 说明 |
| :------- | :-------- | :--- |
| aclnn 调用 | `op_api/aclnn_index_v2.h`（`aclnnIndexGetWorkspaceSize` / `aclnnIndex`） | 当前目录未提供 `examples/test_aclnn_index.cpp`，可通过 aclnn 接口头文件和 ATK 用例进行验证。 |
| Level0 内部调用 | `op_api/experimental_index_l0.h`（`IndexAiCore` / `IndexAiCpu`） | aclnn 接口根据索引类型和形状选择 AICore 或 AICPU 路径。 |
