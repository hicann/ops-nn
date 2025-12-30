/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
enum OnnxDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT_8 = 2,
  INT_8 = 3,
  UINT_16 = 4,
  INT_16 = 5,
  INT_32 = 6,
  INT_64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT_16 = 10,
  DOUBLE = 11,
  UINT_32 = 12,
  UINT_64 = 13,
  BFLOAT_16 = 16
};

static DataType GetGeDataType(int32_t data_type)
{
  static DataType onnxToGeDataType[UINT_64 + 1] = {
    DT_UNDEFINED,
    DT_FLOAT,
    DT_UINT8,
    DT_INT8,
    DT_UINT16,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_STRING,
    DT_BOOL,
    DT_FLOAT16,
    DT_DOUBLE,
    DT_UINT32,
    DT_UINT64
  };
  if (data_type == BFLOAT_16) {
    return DT_BF16;
  } else if (data_type > UINT_64 || data_type < 0) {
    return DT_UNDEFINED;
  }
  return onnxToGeDataType[data_type];
}

static uint8_t* ParseTensorValue(const ge::onnx::TensorProto &tp)
{
  const uint8_t *data = nullptr;
  auto data_type = tp.data_type();
  OP_LOGD("ConstantOfShape", "Datatype[%d.]", data_type);
  switch (data_type) {
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_INT64:
      data = reinterpret_cast<const uint8_t *>(tp.int64_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_INT32:
      data = reinterpret_cast<const uint8_t *>(tp.int32_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
      data = reinterpret_cast<const uint8_t *>(tp.float_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
      data = reinterpret_cast<const uint8_t *>(tp.double_data().data());
      break;
    default:
      OP_LOGE("ConstantOfShape", "Datatype[%d] don't support.", data_type);
  }
  return const_cast<uint8_t *>(data);
}

static Status  ParseParamsConstantOfShape(const Message *op_src, ge::Operator &op_dest)
{
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("ConstantOfShape", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  // 1.add dynamic input and out
  op_dest.DynamicInputRegister("args", 1);
  op_dest.DynamicOutputRegister("output", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::11::ConstantOfShape");
  // 3.set attr if needed
  ge::TensorDesc tensorDesc;
  vector<int64_t> dims = {};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_FLOAT);
  tensorDesc.SetFormat(ge::FORMAT_NCHW);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  size_t size = sizeof(float);
  uint8_t *data = nullptr;
  vector<float> data_dim = {0};
  data = reinterpret_cast<uint8_t *>(data_dim.data());
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "value" && attr.type() == ge::onnx::AttributeProto::TENSOR) {
      if (attr.t().raw_data() != "") {
        auto value = const_cast<char *>(attr.t().raw_data().data());
        data = reinterpret_cast<uint8_t *>(value);
      } else {
        data = ParseTensorValue(attr.t());
      }
      DataType datatype0 = GetGeDataType(attr.t().data_type());
      size = GetSizeByDataType(datatype0);
      tensorDesc.SetDataType(datatype0);
    }
  }
  const ge::Tensor valueTensor(tensorDesc, data, size);
  op_dest.SetAttr("value", valueTensor);
  op_dest.SetAttr("name", node->name());
  return SUCCESS;
}

static Status ParseOpToGraphConstantOfShape(const Operator &op, Graph &graph)
{
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  ge::Tensor value;
  if (op.GetAttr("value", value) != SUCCESS) {
    OP_LOGE("ConstantOfShape", "get value from op failed");
    return FAILED;
  }
  auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(value);
  auto fill = op::Fill((ori_name + "_Fill").c_str()).set_input_dims(data0).set_input_value(data1);

  std::vector<Operator> inputs { data0 };
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(fill, vector<std::size_t> { 0 });
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::ConstantOfShape"),
                 ge::AscendString("ai.onnx::9::ConstantOfShape"),
                 ge::AscendString("ai.onnx::10::ConstantOfShape"),
                 ge::AscendString("ai.onnx::11::ConstantOfShape"),
                 ge::AscendString("ai.onnx::12::ConstantOfShape"),
                 ge::AscendString("ai.onnx::13::ConstantOfShape"),
                 ge::AscendString("ai.onnx::14::ConstantOfShape"),
                 ge::AscendString("ai.onnx::15::ConstantOfShape"),
                 ge::AscendString("ai.onnx::16::ConstantOfShape"),
                 ge::AscendString("ai.onnx::17::ConstantOfShape"),
                 ge::AscendString("ai.onnx::18::ConstantOfShape")})
  .ParseParamsFn(ParseParamsConstantOfShape)
  .ParseOpToGraphFn(ParseOpToGraphConstantOfShape)
  .ImplyType(ImplyType::TVM);
}
