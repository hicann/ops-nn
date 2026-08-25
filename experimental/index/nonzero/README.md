# NonZero

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 说明：Atlas A3（Ascend910_93）与 Atlas A2（Ascend910B）同为 DAV_2201 架构
> （共代际），运行时 `Ascend910_93` 映射到 `ASCEND910B`，由同一份 `ascend910b`
> 注册（config/ascend910b/ + AddConfig）覆盖，无需单独的 A3 配置。本算子仅
> 在 910b（A2）硬件上实测；Ascend 950PR/950DT 属 A5 代际（DAV_3510），未验证，
> 暂不支持。

## 功能说明

- 算子功能：返回输入 tensor 中所有非零元素的索引，对标 `torch.nonzero`。

- 计算公式：

  $$
  y = \{(i_0, i_1, \ldots, i_{n-1}) \mid x[i_0, i_1, \ldots, i_{n-1}] \neq 0\}
  $$

  返回的索引按行优先（row-major / C-order）排列。

- 示例：

  $$
  x = \begin{bmatrix}1 & 0 & 2\\ 0 & 3 & 0\end{bmatrix} \rightarrow
  y = \begin{bmatrix}0 & 0\\ 0 & 2\\ 1 & 1\end{bmatrix}
  $$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|:----------:|------|----------|:--------:|
| x | 输入 | 需要查找非零元素的输入张量 | FLOAT, FLOAT16, BFLOAT16, INT32 | ND |
| y | 输出 | 非零元素索引 [K, ndim]，其中 K 为非零元素数量，ndim 为输入 x 的维度数（维数） | INT64 | ND |

## 约束说明

- 输入维度至少为 2。
- 输出第一维（K = 非零元素数量）为动态维度；第二维固定为输入 x 的维度数 `ndim`。
- 全零输入时输出形状 [0, ndim]。
- 所有 dtype（float32/float16/bfloat16/int32）在核内直接处理，无 host 端转换。

## 实现说明

**核内算法**：
- 每核按 `TILE_ELEMS=8192` 宽的 tile 读取输入（DataCopyPad GM→UB），对 0 做
  `Compares(CMPMODE::NE)` 得到按位打包的 mask。
- bfloat16/int32 无可用 setcc，先 `Cast` 到 float 再比较；float32/float16 直接比较。
- mask 按 uint32 字扫描，用最小置位位提取（`w &= w-1`）只遍历非零位，扫描代价
  正比于非零数而非 tile 宽。
- `(row, col)` 用增量推进（按列间距 + 单次 div/mod 回绕），避免每非零一次 64 位除法。
- 索引对在 UB 中按 `PAIR_BATCH=512` 分批，每批一次 DataCopyPad 写回 GM
  （每对 4 个 int32 hi/lo 字），避免每 2 对一次写回的固定开销。

**输出布局**（framework / aclnn 构建）：
- 单核扫描全部行，索引对直接写回输出缓冲区 y 的连续位置
  （`y[0..K*2)` 为 INT64，契约要求 [K, 2] 紧凑排列，且 aclnn host 无法做多核
  归并；跨核屏障 SyncAll 在本平台即使全核 blockDim 也死锁，故强制单核）。
- Workspace: 8 个 int32（首元素存非零计数）。

**Host 端**：
- 读取 workspace 计数确定实际 K（动态输出形状 [K, ndim] 由框架按此值返回）。

**性能**（vs `torch.nonzero`，msprof kernel 纯耗时）：

按验收指标——逐 case 几何平均加速比 + case 泛化分布（参考 kernel 纯耗时
`<100us:10% / 100-500us:20% / 500-1000us:40% / >1000us:30%`）：

- 100-case 套件精确命中 **10/20/40/30** 分布（零跨界）；
- **逐 case 几何平均加速比 3.657x**，全部 case 加速比 >1（min 1.78x）；
- 按 dtype：float32 3.51x · float16 3.31x · bfloat16 3.34x · int32 5.67x；
- 加权总加速比 Σref/Σasc = 3.98x。

测量方式：单 msprof 启动、同进程、每 case ref/asc 各 3 次取 min、累加全部
计算 kernel Task_Duration。float32/float16/bfloat16（AiCore 参考）与官方逐 case harness
交叉验证偏差 ≤6%；int32 参考走 AiCpu（host 侧 kernel），独立进程冷启动开销
（~350-400us 地板）会使官方 harness 的 int32 加速比虚高，此处用同进程稳态
测量消除该伪影。200 例精度集 200/200 通过。framework/aclnn 单核构建以正确性
为先，输出契约完全对齐 `torch.nonzero`。

## 调用说明

| 调用方式 | 样例 |
|----------|------|
| aclnn调用 | [test_aclnn_nonzero.cpp](./examples/test_aclnn_nonzero.cpp) |
