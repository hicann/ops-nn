# 图模式开发指南
自定义算子如需运行图模式，不需要aclnn适配，做如下交付件适配：
```
${op_name}                              # 替换为实际算子名的小写下划线形式
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # op_graph侧cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导，在运行时推导输出dataType
└── └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
```

### Shape与DataType推导

在深度学习中，当一个算子被加入计算图时，为确保图的正确性和后续的编译、优化、执行流程顺利进行，通常需要为该算子实现两个关键的推导函数：
  - InferShape：用于推导输出张量的形状（shape）。
  - InferDataType：用于推导输出张量的数据类型（dataType）。

操作步骤如下：

**1. 注册InferShape与InferDataType。**

   实现两个目标函数之前，需要先进行注册，框架判断算子的shape和dataType推导逻辑由哪两个函数来处理。

**2. InferShape推导实现。**

   Infershape函数的作用是根据输入的shape推导输出的shape。

**3. InferDataType推导实现。**

   InferDataType函数的作用是根据输入的dataType推导输出的dataType。

根据上述步骤，编写`AddExample`算子的推导实现，示例代码如下：

```C++
// AddExample算子逻辑是两个数相加，因此输出shape与输入shape一致
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // 获取输出shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // 获取输入DimNum
    auto xShapeSize = xShape->GetDimNum();
    // 设置输出的DimNum
    yShape->SetDimNum(xShapeSize);
    // 依次将输入Dim值设置给输出
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}

// AddExample算子逻辑是两个数相加，因此输出dataType与输入dataType一致
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // 获取输入的dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // 将输出dataType设置到输出
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// 注册InferShape与InferDataType
IMPL_OP_INFERSHAPE(AddExample).
    InferShape(InferShapeAddExample).
    InferDataType(InferDataTypeAddExample);
```

完整代码请参考`examples/add_example/op_host`目录下[add_example_infershape.cpp](../../examples/add_example/op_host/add_example_infershape.cpp)。   

### 算子原型配置
图模式调用需要将算子原型注册到[Graph Engine](https://www.hiascend.com/cann/graph-engine)（简称GE）中，以便GE能够识别该类型算子的输入、输出及属性信息。注册通过`REG_OP`接口完成，开发者需要定义算子的输入、输出张量类型及数量等基本信息。

示例代码如下，展示了如何注册`AddExample`算子：

```CPP
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(AddExample)
```

完整代码请参考`examples/add_example/op_graph`目录下[add_example_proto.h](../../examples/add_example/op_graph/add_example_proto.h)。