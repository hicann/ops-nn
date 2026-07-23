# Graph Mode Adaptation Guide

## Overview

If a custom operator needs to run in graph mode, the overall process is consistent with the operator development guide ([AI Core Operator Development Guide](aicore_develop_guide.md)/[AI CPU Operator Development Guide](aicpu_develop_guide.md)). Note that **aclnn adaptation is not required**; only the following deliverable adaptations are needed.

```text
${op_name}                              # Replace with the lowercase underscore form of the actual operator name
├── op_host                             # Host-side implementation
│   └── ${op_name}_infershape.cpp       # InferShape implementation, implementing operator shape inference, inferring the output shape at runtime
├── op_graph                            # Graph fusion-related implementation
│   ├── CMakeLists.txt                  # op_graph side cmakelist file
│   ├── ${op_name}_graph_infer.cpp      # InferDataType file, implementing operator type inference, inferring the output dataType at runtime
└── └── ${op_name}_proto.h              # Operator prototype definition, used for graph optimization and fusion phase operator identification
```

This document uses the `AddExample` operator (assuming it is an AI Core operator) as an example to introduce the implementation of graph mode deliverables. The implementation for AI CPU operators entering the graph is basically similar. For complete code, refer to the `add_example` and `add_example_aicpu` directories under `examples`.

## Shape and DataType Inference

Graph mode requires two deliverables: `${op_name}_graph_infer.cpp` and `${op_name}_infershape.cpp`

**Deliverable 1: ${op_name}_infershape.cpp**

The InferShape function infers the output shape based on the input shape.

The example is as follows. For the complete code of the `AddExample` operator, refer to [add_example_infershape.cpp](../../../examples/add_example/op_host/add_example_infershape.cpp) under `examples/add_example/op_host`.

```C++
// The AddExample operator logic is adding two numbers, so the output shape is consistent with the input shape
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // Obtain the input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // Obtain the output shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // Obtain the input DimNum
    auto xShapeSize = xShape->GetDimNum();
    // Set the output DimNum
    yShape->SetDimNum(xShapeSize);
    // Set the input Dim values to the output one by one
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
// InferShape registration
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**Deliverable 2: ${op_name}_graph_infer.cpp**

The InferDataType function infers the output DataType based on the input DataType. The example is as follows.

```C++
// The AddExample operator logic is adding two numbers, so the output dataType is consistent with the input dataType
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // Obtain the input dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // Set the input dataType to the output
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// Register InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## Operator Prototype Configuration

Graph mode invocation requires registering the operator prototype into [Graph Engine](https://www.hiascend.com/eng/cann/graph-engine) (abbreviated as GE) so that GE can identify the input, output, and attribute information of this type of operator. Registration is completed through the `REG_OP` interface. Developers need to define basic information such as the operator input, output tensor types, and quantities.

Common tensor/attribute data type examples are as follows:

|Tensor Type|Attribute Type|Example|
|-----|------|-----|
|int64|/|DT_INT64|
|int32|/|DT_INT32|
|int16|/|DT_INT16|
|int8|/|DT_INT8|
|double|/|DT_DOUBLE|
|float32|/|DT_FLOAT|
|float16|/|DT_FLOAT16|
|bfloat16|/|DT_BF16|
|complex128|/|DT_COMPLEX128|
|complex64|/|DT_COMPLEX64|
|complex32|/|DT_COMPLEX32|
|/|int|Int|
|/|bool|Bool|
|/|string|String|
|/|float|Float|
|/|list|ListInt|

Basic information is as follows:

|Input/Output|Keyword|Example|
|-----|------|-----|
|Required input|INPUT|.INPUT(${name}, TensorType({input_dtype}))|
|Optional input|OPTIONAL_INPUT|.OPTIONAL_INPUT(${name}, TensorType({optional_input_dtype}))|
|Required attribute|REQUIRED_ATTR|.REQUIRED_ATTR(${name}, ${dtype})|
|Optional attribute|ATTR|.ATTR(${name}, ${dtype}, ${default_value})|
|Output|OUTPUT|.OUTPUT(${name}, TensorType({output_dtype}))|

The sample code below shows how to register the `AddExample` operator:

```CPP
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

For complete code, refer to [add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h) under the `examples/add_example/op_graph` directory.
