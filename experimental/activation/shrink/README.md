# Shrink

## 算子说明

对输入张量进行非线性收缩变换。根据输入值与阈值 `lambd` 的大小关系，通过偏移量 `bias` 进行缩放和偏移处理。

### 计算公式

```
out_i = self_i - bias,  if self_i > lambd
out_i = self_i + bias,  if self_i < -lambd
out_i = 0,              if -lambd <= self_i <= lambd
```

## 支持平台

| 平台 | 支持 |
|------|------|
| Ascend950 | Yes |

## 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| FLOAT16 | FLOAT16 |
| FLOAT | FLOAT |

## 目录结构

```
shrink/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── test_aclnn_shrink.cpp
├── op_host/
│   ├── CMakeLists.txt
│   ├── shrink_def.cpp
│   ├── shrink_infershape.cpp
│   └── shrink_tiling.cpp
├── op_kernel/
│   ├── shrink.cpp
│   ├── shrink.h
│   ├── shrink_tiling_data.h
│   └── shrink_tiling_key.h
└── tests/
    └── .gitkeep
```
