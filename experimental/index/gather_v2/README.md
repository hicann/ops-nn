# GatherV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：从输入张量 `x` 的指定维度 `axis` 上，根据 `indices` 中的下标取出数据，生成输出张量 `y`。
- 接口形态：当前实验目录提供 `aclnnGatherV2GetWorkspaceSize` / `aclnnGatherV2` 接口，接口中 `axis` 以 `dim` 参数传入，并在 Level0 内部转换为 axis Tensor。
- 执行路径：aclnn 接口将输入和索引转为连续张量后调用 Level0 `GatherV2`，当前目录按 Ascend C AICore 路径加入 launcher。

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
      <th>输入/输出/属性</th>
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
      <td>待 gather 的输入张量，对应 aclnn 接口中的 self。</td>
      <td>FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE</td>
      <td>ND</td>
      <td>1-8</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引张量，用于指定从 axis 维度取数的位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>gather 维度。aclnn 接口中该值由 int64_t dim 参数传入，Level0 内部转换为标量 Tensor。</td>
      <td>INT64；图模式 axis Tensor 支持 INT32、INT64</td>
      <td>ND</td>
      <td>0</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>根据 indices 从 x 中取出后的输出张量，数据类型需与 x 一致。</td>
      <td>FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>batch_dims</td>
      <td>属性</td>
      <td>图模式和 Level0 属性，默认值为 0；当前 aclnn 接口不暴露该参数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>negative_index_support</td>
      <td>属性</td>
      <td>图模式和 Level0 属性，默认值为 false；当前 aclnn 接口不暴露该参数。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 仅支持 ND 格式，不支持私有格式输入、索引或输出。
- `x`、`indices`、`y` 的 rank 均不超过 8；当前 Ascend C experimental 路径要求 `x` 为非标量张量。
- `indices` 仅支持 INT32、INT64，`y` 的数据类型必须与 `x` 一致。
- `dim` 支持负数，合法范围为 `[-x rank, x rank)`。
- 输出 shape 需与接口推导结果一致：`indices` 为标量时按长度为 1 的维度处理；其他场景下，将 `x` 的 `dim` 维替换为 `indices` shape。
- `x` 或 `indices` 为空 Tensor 且前置参数校验通过时，`aclnnGatherV2GetWorkspaceSize` 返回成功且 workspace 为 0。
- AICore 二进制当前配置覆盖 `ascend950`。当前 AICore kernel 为实验骨架，`Process()` 尚未实现实际 gather 计算。

## 调用说明

| 调用方式 | 接口/文件 | 说明 |
| :------- | :-------- | :--- |
| aclnn 调用 | `op_api/aclnn_gather_v2_experimental.h`（`aclnnGatherV2GetWorkspaceSize` / `aclnnGatherV2`） | 当前目录未提供 `examples/test_aclnn_gather_v2.cpp`，可通过 aclnn 接口头文件和 ATK/UT 用例进行验证。 |
| Level0 内部调用 | `op_api/gather_v2_l0_experimental.h`（`GatherV2`） | aclnn 接口完成参数校验和连续化处理后，调用 Level0 接口加入 AICore launcher。 |
| 图模式调用 | `op_graph/gather_v2_proto_experimental.h` | IR 输入为 `x`、`indices`、`axis`，输出为 `y`，属性为 `batch_dims` 和 `negative_index_support`。 |
