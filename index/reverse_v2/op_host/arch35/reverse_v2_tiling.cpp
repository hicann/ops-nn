/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reverse.cc
 * \brief
 */
#include "reverse_v2_tiling.h"
#include "reverse_v2_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "op_api/op_util.h"

namespace optiling {
static constexpr int64_t THRESHOLD_64 = 64;
static constexpr int64_t THRESHOLD_128 = 128;
static constexpr int64_t THRESHOLD_2048 = 2048;
static const int32_t AXIS = 1;
static const int64_t NUM_2 = 2;
static const int64_t NUM_6 = 6;
static const int64_t NUM_4 = 4;
static const int64_t NUM_5 = 5;

template <typename T>
bool GetAxisToReverseRunInfo(const gert::TilingContext* context, const gert::Shape& x_shape,
                             const gert::Tensor* axes_tensor, ReverseRunInfo& run_info) {
  OP_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  const T* axes_value = axes_tensor->GetData<T>();
  const size_t axes_num = axes_tensor->GetShapeSize();
  const size_t input_x_rank = x_shape.GetDimNum();
  if (axes_num > input_x_rank) {
    OP_LOGE(context->GetNodeName(), "the axes num(%zu) cannot greater then input rank(%zu).",
                                    axes_num, input_x_rank);
    return false;
  }
  run_info.real_shape_size = input_x_rank;
  run_info.shape_size = x_shape.GetShapeSize();
  for (size_t i = 0; i < input_x_rank; ++i) {
    run_info.reverse_shape[i] = x_shape.GetDim(i);
  }

  for (size_t i = 0; i < axes_num; ++i) {
    OP_CHECK_IF(
        !Ops::Nn::IsDimValid(input_x_rank, axes_value[i]),
        OP_LOGE(context->GetNodeName(), "%s",
                                        Ops::Nn::GenInvalidDimMsg("axis", i, input_x_rank, axes_value[i]).c_str()),
        return false);

    const int64_t real_axes = axes_value[i] < 0 ? input_x_rank + axes_value[i] : axes_value[i];
    OP_LOGD(context->GetNodeName(), "axes[%zu] = %ld", i, real_axes);
    OP_CHECK_IF(run_info.reverse_status[real_axes],
                    OP_LOGE(
                        context->GetNodeName(), "axes is invalid, a specified dim can not be reversed more than once!"),
                    return false);

    run_info.reverse_status[real_axes] = true;
  }

  return true;
}

inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  return parsed_object_cinfo;
}

bool GetTilingCoreNum(const gert::TilingParseContext* context, uint32_t& core_num) {
  auto platform_info = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);

  core_num = platform_info->GetCoreNum();
  OP_LOGD(context->GetNodeName(), "get tiling core num is %u", core_num);
  return true;
}

bool GetAxisToReverseRunInfo(const gert::TilingContext* context, const gert::Shape& x_shape, const size_t axes_idx,
                             ReverseRunInfo& run_info) {
  const gert::Tensor* axes_tensor = context->GetInputTensor(axes_idx);
  OP_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  OP_CHECK_IF(!Ops::Nn::IsConstTensor(axes_tensor),
                  OP_LOGE(context->GetNodeName(), "get axes const value failed."),
                  return false);
  ge::DataType axes_dtype = axes_tensor->GetDataType();
  switch (axes_dtype) {
    case ge::DT_INT32: {
      return GetAxisToReverseRunInfo<int32_t>(context, x_shape, axes_tensor, run_info);
    }
    case ge::DT_INT64: {
      return GetAxisToReverseRunInfo<int64_t>(context, x_shape, axes_tensor, run_info);
    }
    default:
      OP_LOGE(context->GetNodeName(), "axis only support [int32, int64]. but is %s",
                                      Ops::Base::ToString(axes_dtype).c_str());
      return false;
  }

  return true;
}

static void MergedAxes(ReverseRunInfo& run_info) {
  // filter the last dim than dim value is one
  for (size_t i = run_info.real_shape_size - static_cast<size_t>(1); i >= static_cast<size_t>(1); --i) {
    if (run_info.reverse_shape[i] == 1) {
      run_info.real_shape_size -= 1;
      continue;
    }
    break;
  }

  run_info.reverse_status[run_info.merged_wr_idx] = run_info.reverse_status[run_info.real_shape_size - 1];
  run_info.reverse_shape[run_info.merged_wr_idx] = run_info.reverse_shape[run_info.real_shape_size - 1];
  for (int64_t i = run_info.real_shape_size - 2; i >= 0; --i) {
    const int64_t reverse_dim = run_info.reverse_shape[i];
    if (reverse_dim == 1) {
      // shape dim = 1, will ignore the dim
      continue;
    }
    run_info.first_dim_num = reverse_dim;
    const bool reverse_status = run_info.reverse_status[i];
    if (reverse_status == run_info.reverse_status[run_info.merged_wr_idx]) {
      // merge the new dim to wr dim when reverse_status is same
      run_info.reverse_shape[run_info.merged_wr_idx] = run_info.reverse_shape[run_info.merged_wr_idx] * reverse_dim;
      continue;
    }
    // update the wr idx and value
    run_info.merged_wr_idx -= 1;
    run_info.reverse_status[run_info.merged_wr_idx] = reverse_status;
    run_info.reverse_shape[run_info.merged_wr_idx] = reverse_dim;
  }
}

static void InsertAxes(ReverseRunInfo& run_info, const int64_t split_factor) {
  OP_CHECK_IF(split_factor == 0,
                  OP_LOGE("InsertAxes", "split_factor = 0 is not support"),
                  return );
  run_info.reverse_shape[run_info.merged_wr_idx] = run_info.reverse_shape[run_info.merged_wr_idx] / split_factor;
  run_info.merged_wr_idx -= 1;
  run_info.reverse_shape[run_info.merged_wr_idx] = split_factor;
  run_info.reverse_status[run_info.merged_wr_idx] = run_info.reverse_status[run_info.merged_wr_idx + 1];
}

static void SplitAxes(const ReverseV2CompileInfo* compile_info, ReverseRunInfo& run_info) {
  size_t real_merged_len = run_info.reverse_shape.size() - run_info.merged_wr_idx;
  if (real_merged_len < MIN_MERGED_SIZE && real_merged_len > 0) {
    int64_t split_dim = 1;
    for (int64_t i = 0; i < compile_info->core_num / NUM_2; i++) {
      int64_t cu_split_core_dim = compile_info->core_num - i;
      if (run_info.reverse_shape[run_info.merged_wr_idx] % cu_split_core_dim == 0) {
        split_dim = cu_split_core_dim;
        break;
      }
    }
    if (split_dim != 1 && run_info.reverse_shape[run_info.merged_wr_idx] / split_dim != 1) {
      InsertAxes(run_info, split_dim);
      real_merged_len += static_cast<size_t>(1);
    }
  }

  // the ori_first will not be modified after merged
  const int64_t input_ori_first_dim = run_info.first_dim_num;
  OP_LOGD("SplitAxes", "input_ori_first_dim = %ld", input_ori_first_dim);
  // after split the core dim, when first is big, will split again
  if (real_merged_len < MIN_MERGED_SIZE &&
      run_info.shape_size / run_info.reverse_shape[run_info.merged_wr_idx] < SIZE_SPLIT_THRESH &&
      run_info.reverse_shape[run_info.merged_wr_idx] > input_ori_first_dim &&
      input_ori_first_dim > compile_info->core_num) {
    InsertAxes(run_info, input_ori_first_dim);
  }
}

static int64_t CalcTilingKey(ReverseRunInfo& run_info, const ReverseV2CompileInfo* compile_info,
                             ResizeV2TilingData* tiling_data_ptr) {
  // charge tiling_key base on last dim size
  const size_t last_dim_idx = run_info.reverse_status.size() - 1;
  const bool is_last_dim_reverse = run_info.reverse_status[last_dim_idx];
  const int64_t last_dim_num = run_info.reverse_shape[last_dim_idx];
  int64_t max_cut_len = compile_info->max_elements;
  if (is_last_dim_reverse) {
    if (last_dim_num > compile_info->max_elements_last_large_size) {
      tiling_data_ptr->tiling_key = KEY_6;
      max_cut_len = compile_info->max_elements_last_large_size;
    } else if (last_dim_num <= compile_info->max_elements_last_large_size && last_dim_num > THRESHOLD_128) {
      tiling_data_ptr->tiling_key = KEY_5;
    } else {
      tiling_data_ptr->tiling_key = KEY_4;
    }
  } else {
    if (last_dim_num > compile_info->max_elements_last_large_size) {
      tiling_data_ptr->tiling_key = KEY_3;
      max_cut_len = compile_info->max_elements_last_large_size;
    } else if (last_dim_num <= compile_info->max_elements_last_large_size && last_dim_num > F16_NUM_ONE_BLOCK &&
               last_dim_num % F16_NUM_ONE_BLOCK != 0) {
      tiling_data_ptr->tiling_key = KEY_2;
    } else if (last_dim_num <= compile_info->max_elements_last_large_size && last_dim_num > F16_NUM_ONE_BLOCK &&
               last_dim_num % F16_NUM_ONE_BLOCK == 0) {
      tiling_data_ptr->tiling_key = KEY_1;
    } else {
      tiling_data_ptr->tiling_key = KEY_0;
    }
  }

  // topk tiling charge
  if (last_dim_num < compile_info->topk_threshold && run_info.shape_size > THRESHOLD_128 &&
      compile_info->is_vconcat == 1) {
    tiling_data_ptr->tiling_key = KEY_11;
  }

  return max_cut_len;
}

struct ReverseInnerCutInfo {
  int64_t inner_max_count;
  int64_t inner_first_dim;
  int64_t inner_align_count;
};

static void DoInnerCut(const ReverseRunInfo& run_info, const int64_t tiling_key, 
                        ReverseInnerCutInfo& inner_cut_info) {
  // get inner / outer cut dim
  int64_t inner_real_count = 0;
  int64_t mid_inner_loop = 1;
  int64_t last_align_size = 1;

  const bool is_reverse_with_last_dim = run_info.reverse_status[run_info.reverse_shape.size() - 1];
  const int64_t reverse_last_dim_num = run_info.reverse_shape[run_info.reverse_shape.size() - 1];
  inner_cut_info.inner_first_dim = run_info.merged_wr_idx;
  OP_LOGD("DoInnerCut", "before cut inner: inner_real_dims = %ld.", inner_cut_info.inner_first_dim);
  for (int64_t i = run_info.reverse_shape.size() - 1; i >= run_info.merged_wr_idx; --i) {
    if (inner_cut_info.inner_align_count == 0) {
      inner_real_count = run_info.reverse_shape[i];
      inner_cut_info.inner_align_count =
          (run_info.reverse_shape[i] + F16_NUM_ONE_BLOCK - 1) / F16_NUM_ONE_BLOCK * F16_NUM_ONE_BLOCK;
      if (is_reverse_with_last_dim && reverse_last_dim_num > THRESHOLD_128) {
        inner_cut_info.inner_align_count =
            (run_info.reverse_shape[i] + F16_NUM_FOR_VNHWC - 1) / F16_NUM_FOR_VNHWC * F16_NUM_FOR_VNHWC;
      }
      last_align_size = inner_cut_info.inner_align_count;
      continue;
    }
    if ((mid_inner_loop + F16_NUM_FOR_VNHWC - 1) / F16_NUM_FOR_VNHWC * F16_NUM_FOR_VNHWC * last_align_size >
            inner_cut_info.inner_max_count &&
        tiling_key == KEY_4) {
      inner_cut_info.inner_first_dim = i + 1;
      break;
    }
    // when tiling_key = 11, use topk to reverse, need tatol num of inner_shape[1:] < 2048
    if (tiling_key == 11) {
      if (inner_real_count > THRESHOLD_2048) {
        inner_cut_info.inner_first_dim = i + 1;
        break;
      }
    }
    if (inner_cut_info.inner_align_count > inner_cut_info.inner_max_count) {
      inner_cut_info.inner_first_dim = i + 1;
      break;
    }
    if (i == run_info.merged_wr_idx && inner_real_count > THRESHOLD_64) {
      inner_cut_info.inner_first_dim = run_info.merged_wr_idx + 1;
      break;
    }
    mid_inner_loop = mid_inner_loop * run_info.reverse_shape[i];
    inner_cut_info.inner_align_count = inner_cut_info.inner_align_count * run_info.reverse_shape[i];
    inner_real_count = inner_real_count * run_info.reverse_shape[i];
  }
  OP_LOGD("DoInnerCut", "after cut inner: inner_real_dims = %ld.", inner_cut_info.inner_first_dim);
}

static void SetTilingData(const ReverseRunInfo& run_info, const ReverseV2CompileInfo* compile_info,
                          const ReverseInnerCutInfo& inner_cut_info, ResizeV2TilingData* tiling_data_ptr) {
  const int64_t merged_shape_len = run_info.reverse_shape.size() - run_info.merged_wr_idx;
  tiling_data_ptr->inner_real_dims = run_info.reverse_shape.size() - inner_cut_info.inner_first_dim;
  tiling_data_ptr->outer_real_dims = merged_shape_len - tiling_data_ptr->inner_real_dims;
  const size_t tiling_array_size = tiling_data_ptr->inner_shape_array.size();
  for (int64_t i = 0; i < tiling_data_ptr->inner_real_dims; ++i) {
    tiling_data_ptr->inner_shape_array[tiling_array_size - static_cast<size_t>(1) - static_cast<size_t>(i)] =
        static_cast<int64_t>(run_info.reverse_shape[run_info.reverse_shape.size() - 1 - i]);
    tiling_data_ptr->inner_axis_array[tiling_array_size - static_cast<size_t>(1) - static_cast<size_t>(i)] =
        static_cast<int64_t>(run_info.reverse_status[run_info.reverse_status.size() - 1 - i]);
  }
  // modify the inner_shape
  if (tiling_data_ptr->inner_axis_array[NUM_6] == tiling_data_ptr->inner_axis_array[NUM_5] &&
      (tiling_data_ptr->tiling_key == KEY_4 || tiling_data_ptr->tiling_key == KEY_5)) {
    tiling_data_ptr->inner_axis_array[NUM_4] = tiling_data_ptr->inner_axis_array[NUM_5];
    tiling_data_ptr->inner_axis_array[NUM_5] = 0;
    tiling_data_ptr->inner_shape_array[NUM_4] = tiling_data_ptr->inner_shape_array[NUM_5];
    tiling_data_ptr->inner_shape_array[NUM_5] = 1;
    tiling_data_ptr->inner_real_dims += 1;
  }
  for (int64_t i = 0; i < tiling_data_ptr->outer_real_dims; ++i) {
    tiling_data_ptr->outer_shape_array[tiling_array_size - static_cast<size_t>(1) - static_cast<size_t>(i)] =
        static_cast<int64_t>(run_info.reverse_shape[inner_cut_info.inner_first_dim - 1 - i]);
    tiling_data_ptr->outer_axis_array[tiling_array_size - static_cast<size_t>(1) - static_cast<size_t>(i)] =
        static_cast<int64_t>(run_info.reverse_status[inner_cut_info.inner_first_dim - 1 - i]);
  }
  tiling_data_ptr->is_split_axis_reverse =
      tiling_data_ptr->inner_axis_array[tiling_array_size - tiling_data_ptr->inner_real_dims];
  tiling_data_ptr->split_part_num =
      inner_cut_info.inner_max_count /
      (inner_cut_info.inner_align_count /
       tiling_data_ptr->inner_shape_array[tiling_array_size - tiling_data_ptr->inner_real_dims]);
  tiling_data_ptr->split_dim = tiling_data_ptr->inner_shape_array[tiling_array_size - tiling_data_ptr->inner_real_dims];
  if (tiling_data_ptr->split_part_num > tiling_data_ptr->split_dim) {
    tiling_data_ptr->split_part_num = tiling_data_ptr->split_dim;
  }

  if (tiling_data_ptr->inner_shape_array[tiling_array_size - static_cast<size_t>(1)] == static_cast<size_t>(1)) {
    tiling_data_ptr->tiling_key = 0;
  }
  tiling_data_ptr->core_num = compile_info->core_num;
}

static ge::graphStatus Tiling4ReverseV2(gert::TilingContext* context) {
  // get compile info
  const ReverseV2CompileInfo* compile_info = static_cast<const ReverseV2CompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
    return ReverseV2TilingForAscendC(context);
  }
  OP_LOGD(context->GetNodeName(), "begin to do Tiling4ReverseV2.");
  auto runtime_x_shape_ptr = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, runtime_x_shape_ptr);
  const gert::Shape& x_shape = Ops::NN::OpTiling::EnsureNotScalar(runtime_x_shape_ptr->GetStorageShape());

  OP_LOGD(context->GetNodeName(), "will do GetAxisToReverseRunInfo.");
  ReverseRunInfo run_info;
  run_info.Init();
  constexpr size_t input_axis = 1;
  OP_CHECK_IF(!GetAxisToReverseRunInfo(context, x_shape, input_axis, run_info),
                  OP_LOGE(context->GetNodeName(), "do GetAxisToReverseRunInfo failed."),
                  return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "after GetAxisToReverseRunInfo, run info: %s.",
          run_info.OriginInfoToString().c_str());
  OP_CHECK_IF(
      run_info.shape_size == 0,
      OP_LOGE(context->GetNodeName(), "reverse donot support empty tensor, shape is %s.",
                                      Ops::Base::ToString(x_shape).c_str()),
      return ge::GRAPH_FAILED);

  run_info.reverse_shape[run_info.real_shape_size] = compile_info->dtype_rate;
  run_info.shape_size = run_info.shape_size * compile_info->dtype_rate;
  run_info.real_shape_size += 1;
  OP_LOGD(context->GetNodeName(), "after add dtype_rate, run info: %s.", run_info.OriginInfoToString().c_str());

  // do merge to reverse
  OP_LOGD(context->GetNodeName(), "will do MergedAxes.");
  MergedAxes(run_info);
  OP_LOGD(context->GetNodeName(), "after MergedAxes, run info: %s.", run_info.MergeInfoToString().c_str());

  OP_LOGD(context->GetNodeName(), "will to do SplitAxes.");
  SplitAxes(compile_info, run_info);
  OP_LOGD(context->GetNodeName(), "after SplitAxes, run info: %s.", run_info.MergeInfoToString().c_str());

  ResizeV2TilingData* tiling_data_ptr = context->GetTilingData<ResizeV2TilingData>();
  OP_CHECK_NULL_WITH_CONTEXT(context, tiling_data_ptr);
  tiling_data_ptr->Init();

  ReverseInnerCutInfo inner_cut_info{0, 0, 0};
  inner_cut_info.inner_max_count = CalcTilingKey(run_info, compile_info, tiling_data_ptr);

  DoInnerCut(run_info, tiling_data_ptr->tiling_key, inner_cut_info);

  SetTilingData(run_info, compile_info, inner_cut_info, tiling_data_ptr);

  OP_LOGD(context->GetNodeName(), "get tiling info: %s.", tiling_data_ptr->ToString().c_str());
  // set block dim
  context->SetBlockDim(compile_info->core_num);
  OP_LOGD(context->GetNodeName(), "end to do Tiling4ReverseV2.");

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareReverseV2ForAscendC(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareReverseV2ForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<ReverseV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReverseV2(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4ReverseV2");
  auto compile_info = context->GetCompiledInfo<ReverseV2CompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
      return TilingPrepareReverseV2ForAscendC(context);
  }
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OP_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);

  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_CHECK_IF(vars.empty(), OP_LOGE(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  uint32_t core_num = 0;
  OP_CHECK_IF(!GetTilingCoreNum(context, core_num),
                  OP_LOGE(context->GetNodeName(), "get core_num from GE faided."),
                  return ge::GRAPH_FAILED);
  compile_info->core_num = static_cast<int64_t> (core_num);
  OP_CHECK_IF(!ReadCompileItem(vars, "max_elements", compile_info->max_elements),
                  OP_LOGE(context->GetNodeName(), "get max_elements from compile info faided."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(!ReadCompileItem(vars, "max_elements_last_large_size", compile_info->max_elements_last_large_size),
                  OP_LOGE(context->GetNodeName(),
                                                  "get max_elements_last_large_size from compile info faided."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(!ReadCompileItem(vars, "dtype_rate", compile_info->dtype_rate),
                  OP_LOGE(context->GetNodeName(), "get dtype_rate from compile info faided."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(
      !ReadCompileItem(vars, "topk_threshold", compile_info->topk_threshold),
      OP_LOGE(context->GetNodeName(), "get topk_threshold from compile info faided."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF(!ReadCompileItem(vars, "is_vconcat", compile_info->is_vconcat),
                  OP_LOGE(context->GetNodeName(), "get is_vconcat from compile info faided."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ReverseV2 op.
IMPL_OP_OPTILING(ReverseV2)
    .Tiling(Tiling4ReverseV2).TilingParse<ReverseV2CompileInfo>(TilingPrepare4ReverseV2);
}  // namespace optiling
