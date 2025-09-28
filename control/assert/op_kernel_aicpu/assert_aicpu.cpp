/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assert_aicpu.h"

#include <stdint.h>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char *const kAssert = "Assert";

template <typename T>
string PrintOneElement(const T &elt) {
  return to_string(elt);
}

string PrintOneElement(const Eigen::half &elt) {
  return to_string(static_cast<float>(elt));
}

string PrintOneElement(const string &elt) {
  return elt;
}
}

namespace aicpu {
uint32_t AssertCpuKernel::Compute(aicpu::CpuKernelContext &ctx) {
  Tensor *cond = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(cond, KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 failed.")
  KERNEL_CHECK_FALSE((cond->GetTensorShape()->GetDims() == 0),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input 0 should be a scalar.")
  bool *cond_val = reinterpret_cast<bool *>(cond->GetData());
  if ((cond_val != nullptr) && (*cond_val)) {
    KERNEL_LOG_INFO("AssertCpuKernel compute end, op_name=Assert");
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
  }

  auto sum_attr = ctx.GetAttr("summarize");
  KERNEL_CHECK_NULLPTR(sum_attr, KERNEL_STATUS_PARAM_INVALID,
                       "Get summarize attr failed.")
  int64_t sum = sum_attr->GetInt();
  // assert fail info, the value printed depends on the attribute summarize
  string msg = "assertion failed: ";
  for (uint32_t i = 1; i < ctx.GetInputsSize(); ++i) {
    if ((*(ctx.Input(i))).GetDataType() == DT_STRING) {
      string warn_msg = "assert(False) and its dynamic inputs contain DT_STRING type.";
      KERNEL_LOG_WARN("op_name=Assert, [%s]", warn_msg.c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_OK);
    }

    msg.append("[");
    msg.append(SummarizeValue(*(ctx.Input(i)), sum));
    msg.append("]");
    if (i < ctx.GetInputsSize() - 1) {
      msg.append(" ");
    }
  }
  KERNEL_LOG_ERROR("op_name=Assert, [%s]", msg.c_str());
  return KERNEL_STATUS_PARAM_INVALID;
}

string AssertCpuKernel::SummarizeValue(const Tensor &t, int64_t max_entries) {
  const int64_t num_elts = t.NumElements();
  if (max_entries < 0) {
    max_entries = num_elts;
  }
  int64_t limit = std::min(max_entries, num_elts);
  if ((limit > 0) && (t.GetData() == nullptr)) {
    string ret = "uninitialized Tensor of ";
    ret.append(to_string(num_elts));
    ret.append(" elements of type ");
    return ret.append(to_string(static_cast<int>(t.GetDataType())));
  }
  switch (t.GetDataType()) {
    case DT_FLOAT16:
      return SummarizeArray<Eigen::half>(limit, num_elts, t);
    case DT_FLOAT:
      return SummarizeArray<float>(limit, num_elts, t);
    case DT_DOUBLE:
      return SummarizeArray<double>(limit, num_elts, t);
    case DT_UINT32:
      return SummarizeArray<uint32_t>(limit, num_elts, t);
    case DT_INT32:
      return SummarizeArray<int32_t>(limit, num_elts, t);
    case DT_UINT8:
    case DT_QUINT8:
      return SummarizeArray<uint8_t>(limit, num_elts, t);
    case DT_UINT16:
    case DT_QUINT16:
      return SummarizeArray<uint16_t>(limit, num_elts, t);
    case DT_INT16:
    case DT_QINT16:
      return SummarizeArray<int16_t>(limit, num_elts, t);
    case DT_INT8:
    case DT_QINT8:
      return SummarizeArray<int8_t>(limit, num_elts, t);
    case DT_UINT64:
      return SummarizeArray<uint64_t>(limit, num_elts, t);
    case DT_INT64:
      return SummarizeArray<int64_t>(limit, num_elts, t);
    case DT_BOOL:
      return SummarizeArray<bool>(limit, num_elts, t);
    case DT_STRING:
      return SummarizeArray<string>(limit, num_elts, t);
    default: {
      // All irregular cases
      string ret;
      for (int64_t i = 0; i < limit; ++i) {
        ret.append(" ?");
      }
      if (max_entries < num_elts) {
        ret.append("...");
      }
      return ret;
    }
  }
}

// Print from left dim to right dim recursively.
template <typename T>
void AssertCpuKernel::PrintOneDim(int dim_index, const std::shared_ptr<TensorShape> shape,
                                  int64_t limit, int shape_size, const T *data,
                                  int64_t *data_index, string &result) {
  if (*data_index >= limit) {
    return;
  }
  int64_t element_count = shape->GetDimSize(dim_index);
  // We have reached the right-most dimension of the tensor.
  if (dim_index == shape_size - 1) {
    for (int64_t i = 0; i < element_count; i++) {
      if (*data_index >= limit) {
        // If not enough elements has been printed, append "...".
        if (dim_index != 0 && i < element_count) {
          result.append("...");
        }
        return;
      }
      if (i > 0) {
        result.append(" ");
      }
      result.append(PrintOneElement(data[(*data_index)++]));
    }
    return;
  }
  // Loop every element of one dim.
  for (int64_t i = 0; i < element_count; i++) {
    bool flag = false;
    if (*data_index < limit) {
      result.append("[");
      flag = true;
    }
    // As for each element, print the sub-dim.
    PrintOneDim(dim_index + 1, shape, limit, shape_size, data, data_index,
                result);
    if (*data_index < limit || flag) {
      result.append("]");
      flag = false;
    }
  }
}

template <typename T>
string AssertCpuKernel::SummarizeArray(const int64_t limit, const int64_t num_elts,
                                       const Tensor &t) {
  string ret;
  const T *array = reinterpret_cast<const T *>(t.GetData());
  std::shared_ptr<TensorShape> shape = t.GetTensorShape();
  if (shape->GetDimSizes().empty()) {
    for (int64_t i = 0; i < limit; ++i) {
      if (i > 0) ret.append(" ");
      ret.append(PrintOneElement(array[i]));
    }
    if (num_elts > limit) {
      ret.append("...");
    }
    return ret;
  }
  int64_t data_index = 0;
  PrintOneDim(0, shape, limit, shape->GetDims(), array, &data_index, ret);
  if (num_elts > limit) {
    ret.append("...");
  }
  return ret;
}
REGISTER_CPU_KERNEL(kAssert, AssertCpuKernel);
}