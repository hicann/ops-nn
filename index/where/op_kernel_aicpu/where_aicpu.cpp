/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "where_aicpu.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kWhere = "Where";
constexpr int kInitStep = 2;
constexpr int kRankNum = 2;

#define WHERE_COMPUTE_CASE(DTYPE, TYPE)                              \
  case (DTYPE): {                                                    \
    if (WhereCompute<TYPE>(ctx) != KERNEL_STATUS_OK) {               \
        KERNEL_LOG_ERROR("Where kernel compute failed.");            \
        return KERNEL_STATUS_INNER_ERROR;                            \
    }                                                                \
    break;                                                           \
  }

#define WHERE_CALCULATE_CASE(RANK)                                             \
  case (RANK): {                                                               \
    auto cal_status = Where<T, RANK, int64_t>::DoCompute(flat_input,           \
                              eigen_output, input_dims);                       \
    if (cal_status != KERNEL_STATUS_OK) {                                      \
        return KERNEL_STATUS_INNER_ERROR;                                      \
    }                                                                          \
    break;                                                                     \
  }
} // namespaces

namespace aicpu {
template <typename T, int DIMS, typename TIndex>
class Where {
public:
static void CalIndexRowMajor(typename TTypes<int64_t>::Matrix output,
    const typename Eigen::DSizes<TIndex, DIMS> &dim_strides,
    TIndex true_n, TIndex index) {
  for (int i = 0; i < DIMS; ++i) {
    output(true_n, i) = index / dim_strides[i];
    index %= dim_strides[i];
  }
}

static uint32_t DoCompute(typename TTypes<T>::ConstTensor flat_input,
                      typename TTypes<int64_t>::Matrix output,
                      std::vector<int64_t> input_dims) {
  Eigen::DenseIndex true_nums = 0;
  // every dim's stride in memory
  Eigen::DSizes<Eigen::DenseIndex, DIMS> dim_strides;
  static_assert((static_cast<int>(decltype(flat_input)::Layout) == static_cast<int>(Eigen::RowMajor)),
                 "Where kernel input should be RowMajor");
  if (input_dims.size() != DIMS)
    return KERNEL_STATUS_INNER_ERROR;
  // count error dim's number
  int error_dim_count = std::count_if(input_dims.begin(), input_dims.end(),
                                      [](int i){return i <= 0;});
  if (error_dim_count > 0) {
    KERNEL_LOG_ERROR("Input dim can not less than or equal to 0");
    return KERNEL_STATUS_INNER_ERROR;
  }

  // strides only for row major order
  dim_strides[DIMS - 1] = 1;
  for (int i = DIMS - kInitStep; i >= 0; --i) {
    dim_strides[i] = dim_strides[i + 1] * input_dims[i + 1];
  }
  int64_t data_size = flat_input.size();
  auto input_data = flat_input.data();
  for (Eigen::DenseIndex n = 0; n < data_size; ++n) {
    if (!IsValueEqual<T>(input_data[n], T(0))) {
      CalIndexRowMajor(output, dim_strides, true_nums, n);
      ++true_nums;
    }
  }
  return KERNEL_STATUS_OK;
}
};

template <typename T>
uint32_t WhereCpuKernel::WhereCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  typename TTypes<T>::ConstFlat flat_input(reinterpret_cast<const T*>(
          input->GetData()), input->NumElements());
  int64_t num_true =
      std::accumulate(flat_input.data(), flat_input.data() + flat_input.size(),
                      0UL, [](int64_t accum, const T &val) {
                        if (!IsValueEqual<T>(val, T(0))) {
                          return (++accum);
                        }
                        return accum;
                      });
  auto input_shape = input->GetTensorShape();
  int64_t rank = input_shape->GetDims();
  auto input_dims = input_shape->GetDimSizes();
  std::shared_ptr<TensorShape> output_shape = output->GetTensorShape();
  output_shape->SetDimSizes({num_true, rank});
  if (!output->SetTensorShape(output_shape.get())) {
    KERNEL_LOG_ERROR("Set output shape [%ld] [%ld] failed", num_true, rank);
    return KERNEL_STATUS_INNER_ERROR;
  }
  Eigen::DSizes<Eigen::DenseIndex, kRankNum> output_size{num_true, rank};
  typename TTypes<int64_t>::Matrix eigen_output(
          reinterpret_cast<int64_t*>(output->GetData()),
          output_size);
  // In order to maintain compatibility with TF API,
  // only 8 dimensions are supported
  switch (rank) {
    WHERE_CALCULATE_CASE(1)
    WHERE_CALCULATE_CASE(2)
    WHERE_CALCULATE_CASE(3)
    WHERE_CALCULATE_CASE(4)
    WHERE_CALCULATE_CASE(5)
    WHERE_CALCULATE_CASE(6)
    WHERE_CALCULATE_CASE(7)
    WHERE_CALCULATE_CASE(8)
    default:
      KERNEL_LOG_ERROR("Unsupport input dimensions: [%ld]-dim.", rank);
      return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t WhereCpuKernel::Compute(CpuKernelContext &ctx) {
  // check input and output number
  if (NormalCheck(ctx, 1, 1) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // if input empty tensor, return a empty output
  auto dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  for (uint32_t i = 0; i < dims.size(); i++) {
    if (dims[i] == 0) {
      return KERNEL_STATUS_OK;
    }
  }
  auto data_type = static_cast<DataType>(ctx.Input(0)->GetDataType());
  switch (data_type) {
    WHERE_COMPUTE_CASE(DT_BOOL, bool)
    WHERE_COMPUTE_CASE(DT_INT8, int8_t)
    WHERE_COMPUTE_CASE(DT_INT16, int16_t)
    WHERE_COMPUTE_CASE(DT_INT32, int32_t)
    WHERE_COMPUTE_CASE(DT_INT64, int64_t)
    WHERE_COMPUTE_CASE(DT_UINT8, uint8_t)
    WHERE_COMPUTE_CASE(DT_UINT16, uint16_t)
    WHERE_COMPUTE_CASE(DT_UINT32, uint32_t)
    WHERE_COMPUTE_CASE(DT_UINT64, uint64_t)
    WHERE_COMPUTE_CASE(DT_QINT8, int8_t)
    WHERE_COMPUTE_CASE(DT_QUINT8, uint8_t)
    WHERE_COMPUTE_CASE(DT_QINT32, int32_t)
    WHERE_COMPUTE_CASE(DT_FLOAT16, Eigen::half)
    WHERE_COMPUTE_CASE(DT_FLOAT, float)
    WHERE_COMPUTE_CASE(DT_DOUBLE, double)
    WHERE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>)
    WHERE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>)
    default:
      KERNEL_LOG_ERROR("Where kernel data type [%u] not support.", data_type);
      return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kWhere, WhereCpuKernel);
} // nemespace aicpu
