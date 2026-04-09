/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ctime>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <string.h>
#include <vector>

#include "assert.h"

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "tensor.h"
#include "types.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/scatter_elements_proto.h"

#define FAILED -1
#define SUCCESS 0

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}  // namespace ge

using namespace ge;
using std::string;
using std::vector;

string GetTime() {
  time_t timep;
  time(&timep);
  char tmp[64];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
  return tmp;
}

uint32_t GetDataTypeSize(DataType dt) {
  if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
    return 2;
  }
  if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
    return 4;
  }
  if (dt == ge::DT_DOUBLE || dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_COMPLEX64) {
    return 8;
  }
  if (dt == ge::DT_COMPLEX128) {
    return 16;
  }
  return 1;
}

void PrintComplexTensorData(const std::complex<float> *result_data, int64_t output_shape) {
  for (int64_t j = 0; j < output_shape; ++j) {
    printf("result[%ld] is: (%f, %f)\n", j, result_data[j].real(), result_data[j].imag());
  }
}

template <typename T>
int32_t GenTensorData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
                      const vector<T> &values) {
  input_tensor_desc.SetRealDimCnt(shapes.size());
  size_t size = 1;
  for (auto dim : shapes) {
    size *= dim;
  }
  if (size != values.size()) {
    return FAILED;
  }
  auto *data = new (std::nothrow) T[size];
  if (data == nullptr) {
    delete[] data;
    return FAILED;
  }
  for (size_t i = 0; i < size; ++i) {
    data[i] = values[i];
  }
  input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(data), size * sizeof(T));
  return SUCCESS;
}

int32_t WriteDataToFile(const string &bin_file, uint64_t data_size, uint8_t *input_data) {
  FILE *fp = fopen(bin_file.c_str(), "w");
  if (fp == nullptr) {
    return FAILED;
  }
  fwrite(input_data, sizeof(uint8_t), data_size, fp);
  fclose(fp);
  return SUCCESS;
}

void SaveInputOutput(vector<ge::Tensor> &input, vector<ge::Tensor> &output) {
  for (size_t i = 0; i < input.size(); ++i) {
    string input_file = "./tc_ge_irrun_test_scatter_elements_input_" + std::to_string(i) + ".bin";
    auto input_size = input[i].GetTensorDesc().GetShape().GetShapeSize() * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
    WriteDataToFile(input_file, input_size, input[i].GetData());
  }
  for (size_t i = 0; i < output.size(); ++i) {
    string output_file = "./tc_ge_irrun_test_scatter_elements_output_" + std::to_string(i) + ".bin";
    auto output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
    auto output_dtype = output[i].GetTensorDesc().GetDataType();
    auto output_size = output_shape * GetDataTypeSize(output_dtype);
    auto *output_data = output[i].GetData();
    WriteDataToFile(output_file, output_size, output_data);
    PrintComplexTensorData(reinterpret_cast<std::complex<float> *>(output_data), output_shape);
  }
}

#define ADD_INPUT(input_index, input_name, input_dtype, input_shape, input_data)                              \
  do {                                                                                                        \
    auto placeholder = op::Data("placeholder" + std::to_string(input_index)).set_attr_index((input_index) - 1); \
    TensorDesc desc(ge::Shape(input_shape), FORMAT_ND, input_dtype);                                          \
    desc.SetPlacement(ge::kPlacementHost);                                                                    \
    desc.SetFormat(FORMAT_ND);                                                                                \
    Tensor tensor;                                                                                            \
    ret = GenTensorData(input_shape, tensor, desc, input_data);                                               \
    if (ret != SUCCESS) {                                                                                     \
      printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                         \
      return FAILED;                                                                                          \
    }                                                                                                         \
    placeholder.update_input_desc_x(desc);                                                                    \
    placeholder.update_output_desc_y(desc);                                                                   \
    input.push_back(tensor);                                                                                  \
    graph.AddOp(placeholder);                                                                                 \
    scatter_elements.set_input_##input_name(placeholder);                                                     \
    inputs.push_back(placeholder);                                                                            \
  } while (0)

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs, std::vector<Operator> &outputs,
                     Graph &graph) {
  Status ret = SUCCESS;
  auto scatter_elements = op::ScatterElements("scatter_elements");

  vector<int64_t> data_shape = {1, 5};
  vector<int64_t> indices_shape = {1, 2};
  vector<int64_t> updates_shape = {1, 2};

  vector<std::complex<float>> data_values = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
  vector<int64_t> indices_values = {1, 3};
  vector<std::complex<float>> updates_values = {{10, -1}, {20, -2}};

  ADD_INPUT(1, data, DT_COMPLEX64, data_shape, data_values);
  ADD_INPUT(2, indices, DT_INT64, indices_shape, indices_values);
  ADD_INPUT(3, updates, DT_COMPLEX64, updates_shape, updates_values);
  scatter_elements.set_attr_axis(1);
  scatter_elements.set_attr_reduction("none");
  scatter_elements.update_output_desc_y(TensorDesc(ge::Shape(data_shape), FORMAT_ND, DT_COMPLEX64));
  outputs.push_back(scatter_elements);
  return SUCCESS;
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  Graph graph("tc_ge_irrun_test_scatter_elements");
  vector<ge::Tensor> input;
  vector<Operator> inputs;
  vector<Operator> outputs;

  std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
  Status ret = ge::GEInitialize(global_options);
  if (ret != SUCCESS) {
    return FAILED;
  }

  ret = CreateOppInGraph(input, inputs, outputs, graph);
  if (ret != SUCCESS) {
    GEFinalize();
    return FAILED;
  }
  graph.SetInputs(inputs).SetOutputs(outputs);

  std::map<AscendString, AscendString> session_options = {};
  Session session(session_options);
  std::map<AscendString, AscendString> graph_options = {};
  ret = session.AddGraph(0, graph, graph_options);
  if (ret != SUCCESS) {
    GEFinalize();
    return FAILED;
  }

  vector<ge::Tensor> output;
  ret = session.RunGraph(0, input, output);
  if (ret != SUCCESS) {
    GEFinalize();
    return FAILED;
  }
  SaveInputOutput(input, output);
  return GEFinalize() == SUCCESS ? SUCCESS : FAILED;
}
