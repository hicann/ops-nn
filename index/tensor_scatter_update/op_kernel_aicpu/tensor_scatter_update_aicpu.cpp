/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_scatter_update_aicpu.h"

#include <atomic>
#include <complex>
#include <functional>
#include <map>

#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "utils/kernel_util.h"

namespace {
const char *const kTensorScatterUpdate = "TensorScatterUpdate";
constexpr uint32_t kInputNum = 3;
constexpr uint32_t kOutputNum = 1;

constexpr uint32_t kXIndex = 0;
constexpr uint32_t kIndicesIndex = 1;
constexpr uint32_t kUpdatesIndex = 2;
constexpr uint32_t kOutputIndex = 0;

constexpr uint32_t kMinXDims = 1;
constexpr uint32_t kMinUpdatesDims = 1;
constexpr uint32_t kMinIndicesDims = 2;
constexpr uint32_t kSplitSize = 64 * 1024;

bool ValidEmptyOutputShape(uint64_t x_nums, uint64_t indices_nums, uint64_t updates_nums) {
  if (indices_nums == 0 && updates_nums == 0) {
    return true;
  }
  return (x_nums != 0 && indices_nums != 0 && updates_nums != 0);
}
} // namespace

namespace aicpu {
inline uint64_t GetInnerShapeNums(const CpuKernelContextInfo &info) {
  uint64_t inner_shape_nums = 1;
  if (info.index_depth < info.x_dims) {
    inner_shape_nums = info.updates_nums / info.num_updates;
  }
  return inner_shape_nums;
}

inline void GetOuterShape(const CpuKernelContextInfo &info, uint64_t *x_outer_shape) {
  for (uint64_t i = 0; i < info.index_depth; ++i) {
    x_outer_shape[i] = info.x_shape->GetDimSize(i);
  }
}

inline void GetBatchStrides(uint64_t *outer_shape, uint64_t index_depth, uint64_t *batch_strides) {
  batch_strides[index_depth - 1] = 1;
  for (int64_t i = static_cast<int64_t>(index_depth) - 2; i >= 0; --i) {
    batch_strides[i] = batch_strides[i + 1] * outer_shape[i + 1];
  }
}

template <typename T, typename Index>
struct UpdateOutputArgs {
  const Index *indices_data;
  const uint64_t *batch_strides;
  T *output_data;
  const T *updates_data;
  uint64_t inner_shape_nums;
};

template <typename T, typename Index>
uint32_t UpdateOneOutput(const CpuKernelContextInfo &info, const UpdateOutputArgs<T, Index> &args, uint64_t update_idx) {
  uint64_t row_pos = 0;
  for (uint64_t j = 0; j < info.index_depth; ++j) {
    const Index index_d = static_cast<Index>(update_idx * info.index_depth + j);
    const bool out_of_bounds =
        !(args.indices_data[index_d] >= 0 && args.indices_data[index_d] < info.x_shape->GetDimSize(j));
    if (out_of_bounds) {
      KERNEL_LOG_ERROR("the index is out of bounds.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    row_pos += static_cast<uint64_t>(args.indices_data[index_d]) * args.batch_strides[j];
  }

  const auto cpret = memcpy_s(args.output_data + row_pos * args.inner_shape_nums, args.inner_shape_nums * sizeof(T),
                              args.updates_data + update_idx * args.inner_shape_nums,
                              args.inner_shape_nums * sizeof(T));
  if (cpret != EOK) {
    KERNEL_LOG_ERROR("from updates memcpy_s to output failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t DataTypeCheck(const CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((ctx.Input(kXIndex)->GetDataType() == ctx.Input(kUpdatesIndex)->GetDataType()),
                     KERNEL_STATUS_PARAM_INVALID, "The types of x and updates are not same.");
  KERNEL_CHECK_FALSE((ctx.Input(kIndicesIndex)->GetDataType() == DT_INT32 ||
                      ctx.Input(kIndicesIndex)->GetDataType() == DT_INT64),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The indices type only supports int32 or int64, please check!");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InitializeOutput(const CpuKernelContextInfo &info) {
  auto *x_data = static_cast<T *>(info.x->GetData());
  KERNEL_CHECK_NULLPTR(x_data, KERNEL_STATUS_PARAM_INVALID, "Get x_data failed.");
  auto *output_data = static_cast<T *>(info.y->GetData());
  KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID, "Get output failed.");

  const uint64_t total_nums = info.x_nums;
  const uint64_t initial_size = total_nums * static_cast<uint64_t>(sizeof(T));
  uint64_t max_thread_num = initial_size / kSplitSize;
  if (max_thread_num > total_nums || max_thread_num == 0) {
    max_thread_num = total_nums;
  }

  std::atomic<uint32_t> work_ret(KERNEL_STATUS_OK);
  const size_t per_core_size = static_cast<size_t>(total_nums / max_thread_num) * sizeof(T);
  const size_t last_core_size = static_cast<size_t>(total_nums % max_thread_num) * sizeof(T) + per_core_size;

  auto shard_copy = [&](uint64_t start, uint64_t end) {
    for (uint64_t i = start; i < end; ++i) {
      const size_t core_size = (i == (max_thread_num - 1)) ? last_core_size : per_core_size;
      const uint64_t ptr_offset = i * (total_nums / max_thread_num);
      if (ptr_offset >= total_nums) {
        work_ret = KERNEL_STATUS_PARAM_INVALID;
        KERNEL_LOG_ERROR("Pointer offset %lu more than %lu which is overflow", ptr_offset, total_nums);
        return;
      }

      auto *out = output_data + ptr_offset;
      auto *in = x_data + ptr_offset;
      const auto mem_ret = memcpy_s(out, core_size, in, core_size);
      if (mem_ret != EOK) {
        KERNEL_LOG_ERROR("Initial memory copy failed[%d]. Offset is %lu, copy size is %lu",
                         mem_ret, ptr_offset, core_size);
        work_ret = KERNEL_STATUS_INNER_ERROR;
        return;
      }
    }
  };

  if (total_nums > kSplitSize) {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(*info.ctx, max_thread_num, 1, shard_copy),
                        "Initial output data failed!");
  } else {
    const auto cpret = memcpy_s(output_data, total_nums * sizeof(T), x_data, total_nums * sizeof(T));
    KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR, "memcpy_s to output failed.");
  }
  return work_ret;
}

template <typename T, typename Index>
uint32_t UpdateOutput(const CpuKernelContextInfo &info) {
  auto *indices_data = static_cast<Index *>(info.indices->GetData());
  KERNEL_CHECK_NULLPTR(indices_data, KERNEL_STATUS_PARAM_INVALID, "Get indices_data failed.");
  auto *updates_data = static_cast<T *>(info.updates->GetData());
  KERNEL_CHECK_NULLPTR(updates_data, KERNEL_STATUS_PARAM_INVALID, "Get updates_data failed.");
  auto *output_data = static_cast<T *>(info.y->GetData());
  KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID, "Get output failed.");

  const uint64_t inner_shape_nums = GetInnerShapeNums(info);
  auto *outer_shape = new (std::nothrow) uint64_t[info.index_depth];
  if (outer_shape == nullptr) {
    KERNEL_LOG_ERROR("New memory fail.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  GetOuterShape(info, outer_shape);

  auto *batch_strides = new (std::nothrow) uint64_t[info.index_depth];
  if (batch_strides == nullptr) {
    KERNEL_LOG_ERROR("New memory fail.");
    delete[] outer_shape;
    return KERNEL_STATUS_PARAM_INVALID;
  }
  GetBatchStrides(outer_shape, info.index_depth, batch_strides);

  const UpdateOutputArgs<T, Index> args = {indices_data, batch_strides, output_data, updates_data, inner_shape_nums};
  uint32_t ret = KERNEL_STATUS_OK;
  auto shard_copy = [&](uint64_t start, uint64_t end) {
    if (ret != KERNEL_STATUS_OK) {
      return;
    }
    for (uint64_t i = start; i < end; ++i) {
      ret = UpdateOneOutput(info, args, i);
      if (ret != KERNEL_STATUS_OK) {
        return;
      }
    }
  };

  if (info.updates_nums > kSplitSize) {
    ret = CpuKernelUtils::ParallelFor(*info.ctx, info.num_updates, 1, shard_copy);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("Update output data failed!");
    }
  } else {
    shard_copy(0, info.num_updates);
  }

  delete[] outer_shape;
  delete[] batch_strides;
  return ret;
}

template <typename T, typename Index>
uint32_t DoTensorScatterUpdateCompute(const CpuKernelContextInfo &info) {
  uint32_t ret = InitializeOutput<T>(info);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return UpdateOutput<T, Index>(info);
}

const std::map<int, std::map<int, std::function<uint32_t(const CpuKernelContextInfo &)>>> &GetComputeHooks() {
  static const std::map<int, std::map<int, std::function<uint32_t(const CpuKernelContextInfo &)>>> computehooks = {
      {DT_INT32,
       {{DT_FLOAT16, DoTensorScatterUpdateCompute<Eigen::half, int32_t>},
        {DT_FLOAT, DoTensorScatterUpdateCompute<float, int32_t>},
        {DT_DOUBLE, DoTensorScatterUpdateCompute<double, int32_t>},
        {DT_INT8, DoTensorScatterUpdateCompute<int8_t, int32_t>},
        {DT_INT16, DoTensorScatterUpdateCompute<int16_t, int32_t>},
        {DT_INT32, DoTensorScatterUpdateCompute<int32_t, int32_t>},
        {DT_INT64, DoTensorScatterUpdateCompute<int64_t, int32_t>},
        {DT_UINT8, DoTensorScatterUpdateCompute<uint8_t, int32_t>},
        {DT_UINT16, DoTensorScatterUpdateCompute<uint16_t, int32_t>},
        {DT_UINT32, DoTensorScatterUpdateCompute<uint32_t, int32_t>},
        {DT_UINT64, DoTensorScatterUpdateCompute<uint64_t, int32_t>},
        {DT_COMPLEX64, DoTensorScatterUpdateCompute<std::complex<float>, int32_t>},
        {DT_COMPLEX128, DoTensorScatterUpdateCompute<std::complex<double>, int32_t>},
        {DT_QINT8, DoTensorScatterUpdateCompute<int8_t, int32_t>},
        {DT_QINT16, DoTensorScatterUpdateCompute<int16_t, int32_t>},
        {DT_QINT32, DoTensorScatterUpdateCompute<int32_t, int32_t>},
        {DT_QUINT8, DoTensorScatterUpdateCompute<uint8_t, int32_t>},
        {DT_QUINT16, DoTensorScatterUpdateCompute<uint16_t, int32_t>},
        {DT_BOOL, DoTensorScatterUpdateCompute<bool, int32_t>}}},
      {DT_INT64,
       {{DT_FLOAT16, DoTensorScatterUpdateCompute<Eigen::half, int64_t>},
        {DT_FLOAT, DoTensorScatterUpdateCompute<float, int64_t>},
        {DT_DOUBLE, DoTensorScatterUpdateCompute<double, int64_t>},
        {DT_INT8, DoTensorScatterUpdateCompute<int8_t, int64_t>},
        {DT_INT16, DoTensorScatterUpdateCompute<int16_t, int64_t>},
        {DT_INT32, DoTensorScatterUpdateCompute<int32_t, int64_t>},
        {DT_INT64, DoTensorScatterUpdateCompute<int64_t, int64_t>},
        {DT_UINT8, DoTensorScatterUpdateCompute<uint8_t, int64_t>},
        {DT_UINT16, DoTensorScatterUpdateCompute<uint16_t, int64_t>},
        {DT_UINT32, DoTensorScatterUpdateCompute<uint32_t, int64_t>},
        {DT_UINT64, DoTensorScatterUpdateCompute<uint64_t, int64_t>},
        {DT_COMPLEX64, DoTensorScatterUpdateCompute<std::complex<float>, int64_t>},
        {DT_COMPLEX128, DoTensorScatterUpdateCompute<std::complex<double>, int64_t>},
        {DT_QINT8, DoTensorScatterUpdateCompute<int8_t, int64_t>},
        {DT_QINT16, DoTensorScatterUpdateCompute<int16_t, int64_t>},
        {DT_QINT32, DoTensorScatterUpdateCompute<int32_t, int64_t>},
        {DT_QUINT8, DoTensorScatterUpdateCompute<uint8_t, int64_t>},
        {DT_QUINT16, DoTensorScatterUpdateCompute<uint16_t, int64_t>},
        {DT_BOOL, DoTensorScatterUpdateCompute<bool, int64_t>}}}};
  return computehooks;
}

uint32_t IndicesCompute(const CpuKernelContextInfo &info) {
  const DataType params_type = info.x_type;
  const DataType indices_type = info.indices_type;
  try {
    return GetComputeHooks().at(indices_type).at(params_type)(info);
  } catch (const std::out_of_range &e) {
    KERNEL_LOG_ERROR("TensorScatterUpdate op params indices and x doesn't support tensor types: [%s][%s], e.what:[%s]",
                     DTypeStr(indices_type).c_str(), DTypeStr(params_type).c_str(), e.what());
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t TensorScatterUpdateCpukernel::GetInputAndCheck(const CpuKernelContextInfo &info) {
  KERNEL_CHECK_FALSE((info.indices_dims >= kMinIndicesDims), KERNEL_STATUS_PARAM_INVALID,
                     "indices must be at least rank %u but is rank %lu", kMinIndicesDims, info.indices_dims);
  KERNEL_CHECK_FALSE((info.updates_dims >= kMinUpdatesDims), KERNEL_STATUS_PARAM_INVALID,
                     "updates must be at least rank %u but is rank %lu", kMinUpdatesDims, info.updates_dims);
  KERNEL_CHECK_FALSE((info.index_depth <= info.x_dims), KERNEL_STATUS_PARAM_INVALID,
                     "indices.shape[-1] %lu must be less than rank of x %lu", info.index_depth, info.x_dims);
  KERNEL_CHECK_FALSE((info.x_dims >= kMinXDims), KERNEL_STATUS_PARAM_INVALID, "x must be at least 1-D");
  KERNEL_CHECK_FALSE(ValidEmptyOutputShape(info.x_nums, info.indices_nums, info.updates_nums),
                     KERNEL_STATUS_PARAM_INVALID, "Indices and updates specified for empty output shape");

  const uint64_t indices_outer_dims = info.indices_dims - 1;
  for (uint64_t i = 0; i < indices_outer_dims; ++i) {
    KERNEL_CHECK_FALSE((info.indices_shape->GetDimSize(i) == info.updates_shape->GetDimSize(i)),
                       KERNEL_STATUS_PARAM_INVALID, "Outer dimensions of indices and update must match.");
  }

  KERNEL_CHECK_FALSE((info.updates_dims - indices_outer_dims == info.x_dims - info.index_depth),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Inner dimensions of output shape must match inner dimmensions of updates shape");

  for (uint64_t i = 0; i + indices_outer_dims < info.updates_dims; ++i) {
    KERNEL_CHECK_FALSE((info.updates_shape->GetDimSize(i + indices_outer_dims) ==
                        info.x_shape->GetDimSize(info.index_depth + i)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "The inner %lu dimensions of output.shape must match the inner %lu dimmensions of updates.shape",
                       info.x_dims - info.index_depth, info.updates_dims - indices_outer_dims);
  }

  return KERNEL_STATUS_OK;
}

uint32_t TensorScatterUpdateCpukernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Checking TensorScatterUpdate input or output is failed.");
  KERNEL_HANDLE_ERROR(DataTypeCheck(ctx), "The type of input is wrong!");

  CpuKernelContextInfo info;
  info.ctx = &ctx;
  info.x = ctx.Input(kXIndex);
  info.indices = ctx.Input(kIndicesIndex);
  info.updates = ctx.Input(kUpdatesIndex);
  info.y = ctx.Output(kOutputIndex);

  info.x_shape = info.x->GetTensorShape();
  info.indices_shape = info.indices->GetTensorShape();
  info.updates_shape = info.updates->GetTensorShape();

  info.x_dims = info.x_shape->GetDims();
  info.indices_dims = info.indices_shape->GetDims();
  KERNEL_CHECK_FALSE((info.indices_dims >= kMinIndicesDims), KERNEL_STATUS_PARAM_INVALID,
                     "indices must be at least rank %u but is rank %lu", kMinIndicesDims, info.indices_dims);
  info.updates_dims = info.updates_shape->GetDims();

  info.x_nums = info.x_shape->NumElements();
  info.indices_nums = info.indices_shape->NumElements();
  info.updates_nums = info.updates_shape->NumElements();

  info.index_depth = info.indices_shape->GetDimSize(info.indices_dims - 1);
  info.num_updates = (info.index_depth < 1) ? info.indices_nums : (info.indices_nums / info.index_depth);

  info.x_type = info.x->GetDataType();
  info.indices_type = info.indices->GetDataType();

  auto ret = GetInputAndCheck(info);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "GetInputAndCheck failed");

  if (info.indices_nums == 0 && info.updates_nums == 0) {
    return KERNEL_STATUS_OK;
  }

  ret = IndicesCompute(info);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "Compute failed");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTensorScatterUpdate, TensorScatterUpdateCpukernel);
} // namespace aicpu
