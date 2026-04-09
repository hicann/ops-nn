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
#include <numeric>
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
#include "../op_graph/gather_v2_proto.h"

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

void PrintComplexTensorData(const std::complex<double> *result_data, int64_t output_shape) {
  for (int64_t j = 0; j < output_shape; ++j) {
    printf("result[%ld] is: (%lf, %lf)\n", j, result_data[j].real(), result_data[j].imag());
  }
}

template <typename T>
int32_t GenTensorData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
                      const vector<T> &values) {
  input_tensor_desc.SetRealDimCnt(shapes.size());
  size_t size = shapes.empty() ? 1 : 1;
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
    string input_file = "./tc_ge_irrun_test_gather_v2_input_" + std::to_string(i) + ".bin";
    auto input_size = input[i].GetTensorDesc().GetShape().GetShapeSize() * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
    WriteDataToFile(input_file, input_size, input[i].GetData());
  }
  for (size_t i = 0; i < output.size(); ++i) {
    string output_file = "./tc_ge_irrun_test_gather_v2_output_" + std::to_string(i) + ".bin";
    auto output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
    auto output_dtype = output[i].GetTensorDesc().GetDataType();
    auto output_size = output_shape * GetDataTypeSize(output_dtype);
    auto *output_data = output[i].GetData();
    WriteDataToFile(output_file, output_size, output_data);
    PrintComplexTensorData(reinterpret_cast<std::complex<double> *>(output_data), output_shape);
  }
}

#define ADD_INPUT(input_index, input_name, input_dtype, input_shape, input_data)                             \
  do {                                                                                                       \
    auto placeholder = op::Data("placeholder" + std::to_string(input_index)).set_attr_index((input_index) - 1); \
    TensorDesc desc(ge::Shape(input_shape), FORMAT_ND, input_dtype);                                         \
    desc.SetPlacement(ge::kPlacementHost);                                                                   \
    desc.SetFormat(FORMAT_ND);                                                                               \
    Tensor tensor;                                                                                           \
    ret = GenTensorData(input_shape, tensor, desc, input_data);                                              \
    if (ret != SUCCESS) {                                                                                    \
      printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                        \
      return FAILED;                                                                                         \
    }                                                                                                        \
    placeholder.update_input_desc_x(desc);                                                                   \
    placeholder.update_output_desc_y(desc);                                                                  \
    input.push_back(tensor);                                                                                 \
    graph.AddOp(placeholder);                                                                                \
    gather_v2.set_input_##input_name(placeholder);                                                           \
    inputs.push_back(placeholder);                                                                           \
  } while (0)

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs, std::vector<Operator> &outputs,
                     Graph &graph) {
  Status ret = SUCCESS;
  auto gather_v2 = op::GatherV2("gather_v2");

  vector<int64_t> x_shape = {3, 5};
  vector<int64_t> indices_shape = {2};
  vector<int64_t> axis_shape = {1};
  vector<int64_t> y_shape = {3, 2};

  vector<std::complex<double>> x_data = {
      {0.5, 10.5},  {1.5, 11.5},  {2.5, 12.5},  {3.5, 13.5},  {4.5, 14.5},
      {5.5, 15.5},  {6.5, 16.5},  {7.5, 17.5},  {8.5, 18.5},  {9.5, 19.5},
      {10.5, 20.5}, {11.5, 21.5}, {12.5, 22.5}, {13.5, 23.5}, {14.5, 24.5}};
  vector<int64_t> indices_data = {1, 3};
  vector<int64_t> axis_data = {1};

  ADD_INPUT(1, x, DT_COMPLEX128, x_shape, x_data);
  ADD_INPUT(2, indices, DT_INT64, indices_shape, indices_data);
  ADD_INPUT(3, axis, DT_INT64, axis_shape, axis_data);
  gather_v2.set_attr_batch_dims(0);
  gather_v2.set_attr_negative_index_support(false);
  gather_v2.set_attr_is_preprocessed(false);
  gather_v2.update_output_desc_y(TensorDesc(ge::Shape(y_shape), FORMAT_ND, DT_COMPLEX128));
  outputs.push_back(gather_v2);
  return SUCCESS;
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  Graph graph("tc_ge_irrun_test_gather_v2");
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
