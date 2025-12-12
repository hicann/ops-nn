/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ut_op_common.cpp
 */

#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "ut_op_util.h"
#include "graph/attr_value.h"
#include "runtime/tensor_data.h"
#include "graph.h"
#include "gnode.h"
#include "any_value.h"
#include "op_impl_registry.h"

static const std::map<std::string, Ops::NN::AnyValue::ValueType> kAttrTypesMap = {
  {"VT_INT", Ops::NN::AnyValue::ValueType::VT_INT},
  {"VT_BOOL", Ops::NN::AnyValue::ValueType::VT_BOOL},
  {"VT_FLOAT", Ops::NN::AnyValue::ValueType::VT_FLOAT},
  {"VT_STRING", Ops::NN::AnyValue::ValueType::VT_STRING},
  {"VT_LIST_INT", Ops::NN::AnyValue::ValueType::VT_LIST_INT},
  {"VT_LIST_BOOL", Ops::NN::AnyValue::ValueType::VT_LIST_BOOL},
  {"VT_LIST_FLOAT", Ops::NN::AnyValue::ValueType::VT_LIST_FLOAT},
  {"VT_LIST_LIST_INT", Ops::NN::AnyValue::ValueType::VT_LIST_LIST_INT},
};

static uint8_t* GetConstTensor(ge::Operator& op, const size_t index, int64_t shape_size) {
  ge::Tensor const_tensor;
  uint8_t* data = nullptr;
  size_t size = 0;
  Graph graph;
  GNode node = graph.AddNodeByOp(op);
  if (node.GetInputConstData(index, const_tensor) == ge::GRAPH_SUCCESS) {
    size = const_tensor.GetSize();
    data = const_tensor.GetData();
  }
  ge::DataType const_dtype = op.GetInputDesc(index).GetDataType();
  uint8_t* input_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder);
  int64_t value_size = shape_size;
  if (size > 0) {
    std::memcpy(input_tensor + 1, data, size);
    value_size =
        (const_dtype == ge::DT_INT64 || const_dtype == ge::DT_UINT64) ? size / sizeof(int64_t) : size / sizeof(int32_t);
  }
  gert::Tensor tensor({{value_size}, {value_size}},        // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                      (size > 0) ? gert::kFollowing : gert::kOnHost,   // placement
                      const_dtype,                         // dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  return input_tensor_holder;
}

static void GetConstTensorData(ge::Operator& op, const size_t index, gert::Tensor &tensor) {
  Graph graph;
  GNode node = graph.AddNodeByOp(op);
  ge::Tensor const_tensor;
  (void)node.GetInputConstData(index, const_tensor);
  tensor.SetData(gert::TensorData(const_tensor.GetData()));
  tensor.SetSize(const_tensor.GetSize());
  tensor.SetDataType(const_tensor.GetDataType());
  tensor.SetOriginFormat(const_tensor.GetOriginFormat());
  tensor.SetStorageFormat(const_tensor.GetFormat());
  for (size_t idx = 0; idx < const_tensor.GetOriginShapeDimNum(); ++idx) {
    tensor.MutableOriginShape().AppendDim(const_tensor.GetOriginShapeDim(idx));
  }
  for (size_t idx = 0; idx < const_tensor.GetShapeDimNum(); ++idx) {
    tensor.MutableStorageShape().AppendDim(const_tensor.GetShapeDim(idx));
  }
}

ge::graphStatus InferShapeTest(ge::Operator& op, const Runtime2TestParam& param) {
  ge::graphStatus ret;
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  std::string optype = op.GetOpType();
  size_t output_size = op.GetOutputsSize();
  auto faker = gert::InferShapeContextFaker();
  faker.SetOpType(optype).NodeIoNum(input_size, output_size).IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<void *> input_shapes_ref(input_size);
  std::vector<gert::Tensor> input_tensors(input_size);
  std::vector<gert::Tensor *> input_tensors_ref(input_size);
  if (input_size > 0) {
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        auto input_desc = op.GetInputDesc(i);
        gert::Tensor tensor;
        GetConstTensorData(op, i, tensor);
        for (int64_t dim : input_desc.GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc.GetShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }

        input_shapes_ref[count] = &input_shapes[count];

        ge::Format input_format = input_desc.GetFormat();
        ge::Format origin_format = input_desc.GetOriginFormat();
        ge::DataType dtype = input_desc.GetDataType();
        faker.NodeInputTd(count, dtype, origin_format, input_format);

        input_tensors.push_back(std::move(tensor));
        input_tensors_ref[count] = (gert::Tensor *)&input_shapes[count];
        count++;
      } else for (int idx = 0; idx < irnum[i]; idx++) {
        size_t idx_off = i + idx;
        if (i > 0) {
          auto irnum_i = irnum[i-1] == 0 ? 1 : irnum[i-1];
          idx_off = irnum_i * i + idx;
        }
        auto input_desc = op.GetInputDesc(idx_off);
        ge::Format input_format = input_desc.GetFormat();
        ge::Format origin_format = input_desc.GetOriginFormat();
        ge::DataType dtype = input_desc.GetDataType();
        faker.NodeInputTd(count, dtype, origin_format, input_format);
        for (int64_t dim : input_desc.GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc.GetShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }
        input_shapes_ref[count] = &input_shapes[count];

        gert::Tensor tensor;
        tensor.SetDataType(dtype);
        tensor.SetOriginFormat(origin_format);
        tensor.SetStorageFormat(input_format);
        tensor.MutableOriginShape() = input_shapes[count].GetOriginShape();
        tensor.MutableStorageShape() = input_shapes[count].GetStorageShape();
        input_tensors.push_back(std::move(tensor));
        input_tensors_ref[count] = (gert::Tensor *)&input_shapes[count];

        count++;
      }
    }
    faker.InputShapes(input_shapes_ref);
    faker.InputTensors(input_tensors_ref);
  }

  std::vector<gert::StorageShape> output_shapes(output_size);
  std::vector<gert::StorageShape *> output_shapes_ref(output_size);
  if (output_size > 0) {
    ge::TensorDesc tensor_desc = create_desc({-2});
    for (size_t i = 0; i < output_size; ++i) {
      output_shapes_ref[i] = &output_shapes[i];
      op.UpdateOutputDesc(i, tensor_desc);
    }
    faker.OutputShapes(output_shapes_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case Ops::NN::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              faker.Attr(item, AscendString(value.c_str()));
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_FLOAT: {
              std::vector<float32_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            default:
              std::cout << "[ERROR]"<<__FILE__<<":"<<__LINE__<<"The ValueType is not supported!" << std::endl;
          }
        }
      }
    }
  }

  auto holder = faker.Build();

  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->infer_shape;
  if (infer_shape_func == nullptr) return GRAPH_FAILED;
  gert::InferShapeContext *context = holder.GetContext<gert::InferShapeContext>();
  if (context == nullptr) return GRAPH_FAILED;
  ret = infer_shape_func(context);
  for (uint8_t* tensor : const_tensors) { delete []tensor; }
  for (size_t i = 0; i < output_size; i++) {
    auto out_shape = context->GetOutputShape(i);
    if (out_shape == nullptr) return GRAPH_FAILED;
    auto output_desc = op.GetOutputDesc(i);
    std::vector<int64_t> shape;
    for (size_t idx = 0; idx < out_shape->GetDimNum(); ++idx) {
      shape.push_back(out_shape->GetDim(idx));
    }
    output_desc.SetShape(ge::Shape(shape));
    op.UpdateOutputDesc(i, output_desc);
  }
  return ret;
}

ge::graphStatus InferDataTypeTest(ge::Operator& op, const Runtime2TestParam& param) {
  ge::graphStatus ret;
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  std::string optype = op.GetOpType();
  size_t output_size = op.GetOutputsSize();
  auto faker = gert::InferDataTypeContextFaker();
  faker.NodeIoNum(input_size, output_size).IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<ge::DataType> input_datatype(input_size);
  std::vector<void *> input_datatype_ref(input_size);
  std::vector<gert::Tensor *> input_tensors_ref(input_size);
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<void *> input_shapes_ref(input_size);
  std::vector<gert::Tensor> input_tensors(input_size);
  if (input_size > 0) {
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        auto input_desc = op.GetInputDesc(i);
        gert::Tensor tensor;
        GetConstTensorData(op, i, tensor);
        for (int64_t dim : input_desc.GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc.GetShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }

        input_shapes_ref[count] = &input_shapes[count];

        ge::Format input_format = input_desc.GetFormat();
        ge::Format origin_format = input_desc.GetOriginFormat();
        ge::DataType dtype = input_desc.GetDataType();
        faker.NodeInputTd(count, dtype, origin_format, input_format);

        input_tensors.push_back(std::move(tensor));
        input_tensors_ref[count] = (gert::Tensor *)&input_shapes[count];
        count++;
      } else for (int idx = 0; idx < irnum[i]; idx++) {
        size_t idx_off = i + idx;
        if (i > 0) {
          auto irnum_i = irnum[i-1] == 0 ? 1 : irnum[i-1];
          idx_off = irnum_i * i + idx;
        }
        auto input_desc = op.GetInputDesc(idx_off);
        ge::Format input_format = input_desc.GetFormat();
        ge::Format origin_format = input_desc.GetOriginFormat();
        ge::DataType dtype = input_desc.GetDataType();
        faker.NodeInputTd(count, dtype, origin_format, input_format);
        for (int64_t dim : input_desc.GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc.GetShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }
        input_shapes_ref[count] = &input_shapes[count];

        gert::Tensor tensor;
        tensor.SetDataType(dtype);
        tensor.SetOriginFormat(origin_format);
        tensor.SetStorageFormat(input_format);
        tensor.MutableOriginShape() = input_shapes[count].GetOriginShape();
        tensor.MutableStorageShape() = input_shapes[count].GetStorageShape();
        input_tensors.push_back(std::move(tensor));
        input_tensors_ref[count] = (gert::Tensor *)&input_shapes[count];

        count++;
      }
    }
    faker.InputDataTypes(input_datatype_ref);
  }

  std::vector<ge::DataType> output_datatype(output_size);
  std::vector<void *> output_datatype_ref(output_size);
  if (output_size > 0) {
    ge::TensorDesc tensor_desc = create_desc({-2});
    for (size_t i = 0; i < output_size; ++i) {
      output_datatype_ref[i] = &output_datatype[i];
      op.UpdateOutputDesc(i, tensor_desc);
    }
    faker.OutputDataTypes(output_datatype_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case Ops::NN::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, AscendString(value.c_str()));
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_FLOAT: {
              std::vector<float32_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            case Ops::NN::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
                faker.Attr(item, value);
            }
            break;
            default:
              std::cout << "[ERROR]"<<__FILE__<<":"<<__LINE__<<"The ValueType is not supported!" << std::endl;
          }
        }
      }
    }
  }

  auto holder = faker.Build();

  auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->infer_datatype;
  if (infer_datatype_func == nullptr) return GRAPH_FAILED;

  gert::InferDataTypeContext *context = holder.GetContext<gert::InferDataTypeContext>();
  if (context == nullptr) return GRAPH_FAILED;
  ret = infer_datatype_func(context);
  for (uint8_t* tensor : const_tensors) { delete []tensor; }
  for (size_t i = 0; i < output_size; i++) {
    auto output_datatype = context->GetOutputDataType(i);
    auto output_desc = op.GetOutputDesc(i);
    output_desc.SetDataType(output_datatype);
    op.UpdateOutputDesc(i, output_desc);
  }
  return ret;
}

ge::graphStatus InferShapeTest(ge::Operator& op) {
  Runtime2TestParam param;
  return InferShapeTest(op, param);
}

ge::graphStatus InferDataTypeTest(ge::Operator& op) {
  Runtime2TestParam param;
  return InferDataTypeTest(op, param);
}