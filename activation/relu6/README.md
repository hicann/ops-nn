# Relu6 算子

Relu6 激活函数算子的 Ascend C 自定义算子实现，面向 Ascend 950 芯片。

## 功能简介

Relu6 算子按元素对输入张量执行 Relu6 激活函数计算，将输入张量中的每个元素值限制在 `[0, 6]` 区间内。

### 数学公式

```text
y = min(max(x, 0), 6)
```

等价于分段函数：

```text
y = { 0,   if x < 0
    { x,   if 0 <= x <= 6
    { 6,   if x > 6
```

## 支持特性

### 数据类型

| 数据类型 | C 类型 |
|---------|--------|
| float16 | half |
| float | float |
| int32 | int32_t |
| bfloat16 | bfloat16_t |

### Shape 约束

- 输入 x 和输出 y 的 shape 必须相同
- 最高支持 8 维张量
- 支持任意维度（1D ~ 8D），使用 FORMAT_ND（任意维度连续排布）

### 芯片支持

| 芯片 | 状态 |
|-----|------|
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |

## 工程结构

```text
relu6/
├── examples/                       # 调用示例
│   ├── test_geir_relu6.cpp           # GE IR 图模式调用示例
│   └── arch35/
│       └── test_geir_relu6.cpp       # arch35 调用示例
├── op_graph/                       # 图模式适配
│   ├── relu6_proto.h                 # 算子原型定义
│   └── fusion_pass/
├── op_host/                        # 主机端算子注册代码
│   ├── relu6_def.cpp                 # 算子定义
│   └── arch35/
│       ├── relu6_tiling_arch35.cpp   # Tiling 策略
│       └── relu6_tiling_arch35.h     # Tiling 头文件
├── op_kernel/                      # Kernel 实现代码
│   ├── relu6.cpp                     # Kernel 入口
│   └── arch35/
│       ├── relu6.h                   # Kernel 类实现
│       ├── relu6_tiling_data.h       # TilingData 结构体
│       └── relu6_tiling_key.h        # TilingKey 模板参数
├── tests/                          # 测试代码
├── CMakeLists.txt                  # 顶层 CMake 配置
└── README.md                       # 本文件
```

## 调用说明

本算子通过图模式（GE IR）进行调用，示例参考 `examples/test_geir_relu6.cpp`。

## 实现原理

Relu6 算子在 AI Core 上通过两次矢量计算实现：

1. **Maxs 计算**：`tmp = max(x, 0)` -- 将小于 0 的值截断为 0
2. **Mins 计算**：`y = min(tmp, 6)` -- 将大于 6 的值截断为 6

内部采用多核并行 + UB 缓冲优化策略：

- 根据数据量和 AI Core 数量自动切分数据，实现多核并行处理
- UB 中使用 3 个 LocalTensor（input、tmp、output），根据 UB 大小动态计算单次循环处理量
- LocalTensor 起始地址 32 字节对齐，尾部数据通过 DataCopyPad 自动处理
