/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_v2_aicpu.h"

#include <complex>
#include <functional>
#include <map>
#include <vector>

#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "utils/kernel_util.h"

namespace {
using namespace aicpu;

const char *const kGatherV2 = "GatherV2";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const uint32_t kAxisIndex = 2;
const std::vector<std::string> kBatchDimsAttr = {"batch_dims"};

struct GatherV2ComputeInfo {
  int64_t axis = 0;
  int64_t batch_dims = 0;
  int64_t gather_dim_size = 0;
  int64_t batch_size = 1;
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t batch_indices = 0;
  int64_t slice_size = 0;
};

int64_t GetAxisValue(const CpuKernelContext &ctx) {
  if (ctx.Input(kAxisIndex)->GetDataType() == DT_INT32) {
    auto *axis_addr = static_cast<int32_t *>(ctx.Input(kAxisIndex)->GetData());
    return axis_addr == nullptr ? 0 : static_cast<int64_t>(*axis_addr);
  }

  auto *axis_addr = static_cast<int64_t *>(ctx.Input(kAxisIndex)->GetData());
  return axis_addr == nullptr ? 0 : *axis_addr;
}

template <typename T>
uint32_t InitGatherV2Info(const CpuKernelContext &ctx, GatherV2ComputeInfo &info) {
  auto *params = ctx.Input(0);
  auto *indices = ctx.Input(1);
  if (params->NumElements() == 0 || indices->NumElements() == 0) {
    return KERNEL_STATUS_OK;
  }

  info.axis = GetAxisValue(ctx);
  if (info.axis < 0) {
    info.axis += params->GetTensorShape()->GetDims();
  }
  info.batch_dims = ctx.GetAttr("batch_dims")->GetInt();
  info.gather_dim_size = params->GetTensorShape()->GetDimSize(static_cast<int32_t>(info.axis));
  for (int64_t i = 0; i < info.batch_dims; ++i) {
    info.batch_size *= params->GetTensorShape()->GetDimSize(static_cast<int32_t>(i));
  }
  for (int64_t i = info.batch_dims; i < info.axis; ++i) {
    info.outer_size *= params->GetTensorShape()->GetDimSize(static_cast<int32_t>(i));
  }
  for (int64_t i = info.axis + 1; i < params->GetTensorShape()->GetDims(); ++i) {
    info.inner_size *= params->GetTensorShape()->GetDimSize(static_cast<int32_t>(i));
  }

  info.batch_indices = indices->NumElements() / info.batch_size;
  KERNEL_CHECK_FALSE(MulWithoutOverflow(info.inner_size, static_cast<int64_t>(sizeof(T)), info.slice_size),
                     KERNEL_STATUS_INNER_ERROR, "slice size over flow UINT64_MAX");
  return KERNEL_STATUS_OK;
}

template <typename T, typename Index>
uint32_t CopyGatherV2Data(const CpuKernelContext &ctx, const GatherV2ComputeInfo &info) {
  auto *params_base = static_cast<T *>(ctx.Input(0)->GetData());
  auto *indices_data = static_cast<Index *>(ctx.Input(1)->GetData());
  auto *output_base = static_cast<T *>(ctx.Output(0)->GetData());
  for (int64_t batch = 0; batch < info.batch_size; ++batch) {
    for (int64_t i = 0; i < info.outer_size; ++i) {
      for (int64_t j = 0; j < info.batch_indices; ++j) {
        int64_t indices_idx = batch * info.batch_indices + j;
        int64_t indices_value = indices_data[indices_idx] < 0 ? indices_data[indices_idx] + info.gather_dim_size
                                                               : indices_data[indices_idx];
        KERNEL_CHECK_FALSE(indices_value < info.gather_dim_size, KERNEL_STATUS_PARAM_INVALID,
                           "Indices[%ld] = %ld is not in [0, %ld)", indices_idx,
                           static_cast<int64_t>(indices_data[indices_idx]), info.gather_dim_size);
        auto params_idx = (i * info.gather_dim_size + indices_value + batch * info.outer_size * info.gather_dim_size) *
                          info.inner_size;
        auto output_idx =
            (i * info.batch_indices + j + batch * info.outer_size * info.batch_indices) * info.inner_size;
        auto copy_ret = BiggerMemCpy(output_base + output_idx, info.slice_size, params_base + params_idx, info.slice_size);
        KERNEL_CHECK_FALSE(copy_ret == true, KERNEL_STATUS_INNER_ERROR,
                           "memcpy_s to output failed. ret[%d], out_idx[%ld], slice_size[%ld], params_idx[%ld]",
                           copy_ret, output_idx, info.slice_size, params_idx);
      }
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace

namespace aicpu {
template <typename T, typename Index>
uint32_t DoGatherV2Compute(const CpuKernelContext &ctx) {
  GatherV2ComputeInfo info;
  auto ret = InitGatherV2Info<T>(ctx, info);
  if (ret != KERNEL_STATUS_OK || ctx.Input(0)->NumElements() == 0 || ctx.Input(1)->NumElements() == 0) {
    return ret;
  }
  return CopyGatherV2Data<T, Index>(ctx, info);
}

template <typename Index>
uint32_t IndicesCompute(CpuKernelContext &ctx) {
  static const std::map<DataType, std::function<uint32_t(CpuKernelContext &)>> kCalls = {
      {DT_FLOAT16, DoGatherV2Compute<Eigen::half, Index>},
      {DT_FLOAT, DoGatherV2Compute<float, Index>},
      {DT_DOUBLE, DoGatherV2Compute<double, Index>},
      {DT_BOOL, DoGatherV2Compute<bool, Index>},
      {DT_INT8, DoGatherV2Compute<int8_t, Index>},
      {DT_INT16, DoGatherV2Compute<int16_t, Index>},
      {DT_INT32, DoGatherV2Compute<int32_t, Index>},
      {DT_INT64, DoGatherV2Compute<int64_t, Index>},
      {DT_UINT8, DoGatherV2Compute<uint8_t, Index>},
      {DT_UINT16, DoGatherV2Compute<uint16_t, Index>},
      {DT_UINT32, DoGatherV2Compute<uint32_t, Index>},
      {DT_UINT64, DoGatherV2Compute<uint64_t, Index>},
      {DT_COMPLEX64, DoGatherV2Compute<std::complex<float>, Index>},
      {DT_COMPLEX128, DoGatherV2Compute<std::complex<double>, Index>}};
  auto data_type = ctx.Input(0)->GetDataType();
  auto it = kCalls.find(data_type);
  if (it == kCalls.end()) {
    KERNEL_LOG_ERROR("GatherV2 doesn't support data type [%s]", DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return it->second(ctx);
}

uint32_t GatherV2CpuKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  auto axis = GetAxisValue(ctx);
  auto params_dims = ctx.Input(0)->GetTensorShape()->GetDims();
  int64_t min_params_dim = axis < 0 ? -axis : axis + 1;
  KERNEL_CHECK_FALSE(params_dims >= min_params_dim, KERNEL_STATUS_PARAM_INVALID,
                     "Shape must be at least rank [%ld] but is rank [%d]", min_params_dim, params_dims);

  auto *batch_dims_ptr = ctx.GetAttr("batch_dims");
  KERNEL_CHECK_NULLPTR(batch_dims_ptr, KERNEL_STATUS_PARAM_INVALID, "Get batch_dims failed");
  KERNEL_CHECK_FALSE(batch_dims_ptr->GetInt() >= 0, KERNEL_STATUS_PARAM_INVALID,
                     "batch_dims must be at least 0 but is [%ld]", batch_dims_ptr->GetInt());
  return KERNEL_STATUS_OK;
}

uint32_t GatherV2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, kBatchDimsAttr), "Check GatherV2 params failed.");
  auto ret = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "GetInputAndCheck failed");
  return ctx.Input(1)->GetDataType() == DT_INT32 ? IndicesCompute<int32_t>(ctx) : IndicesCompute<int64_t>(ctx);
}

REGISTER_CPU_KERNEL(kGatherV2, GatherV2CpuKernel);
}  // namespace aicpu
