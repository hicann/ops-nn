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
 * \file gather_v2_tiling_arch35.cpp
 * \brief
 */

#include "gather_v2_tiling_arch35.h"
#include "gather_v2_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/inc/tiling_base/tiling_util.h"

namespace optiling {
static const size_t INPUT_IDX_AXIS = 2;
static const int64_t HALF_UB = 2;
static const int64_t DATA_VALUE = 1024;
static const int64_t NUM_32 = 32;
static const int64_t ACTUAL_NUM = 56.5;
static const int64_t GATE_VALUE = 0.012;
static const int64_t BLOCK_SIZE = 32;
static const int64_t NANO_BLOCK_SIZE = 16;
static const int64_t UB_BUF_CNT = 2;
static const int64_t DOUBLE = 2;
static const int64_t VECTOR_BLOCK_SIZE = 256;
static const int64_t UB_BUF_CNT_LARGE_SHAPE = 3;
static const int64_t MAX_X_SPLIT_NUM = 5;
static const int64_t PARAMS_CACHED_UB = static_cast<int64_t>(100 * 1024);
static const int64_t RESERVED_UB_SIZE = static_cast<int64_t>(8 * 1024);
static const int64_t RESERVED_UB_BLOCK = 8;
static const int64_t INT_PER_BLOCK_NUM = 64;
static const int64_t INDICES_MIN_NUM_FOR_CACHE = static_cast<int64_t>(10 * 1024);
static const int64_t CACHE_MODE_UB_SLICE = 6;
static const int64_t RESERVED_UB_SIZE_2K = static_cast<int64_t>(2 * 1024);
static const int64_t TRANS_POSE_LINE_SIZE = 16;
static const int64_t SUPPORT_PARAM_SIZE = 2;
static const int64_t THREE_PART_UB_SIZE = 3;
static const int64_t FOUR_PART_UB_SIZE = 4;
static const int64_t ALIGN_FOR_ONCE_UB_SIZE = 64;
static const int64_t ONE_CORE = 1;
static const int64_t EIGHT_CORES = 8;
static const int64_t SIZE_FOR_ONE_CORE = 12800;
static const int64_t SIZE_FOR_EIGHT_CORE = static_cast<int64_t>(12800 * 24);
static const int64_t IO_RATIO_THRE = 20;
// A. block tiling: indices tiling
// 1. one params row size is smaller than 32B
// params is not cache
static const int64_t TILING_MODE_1 = 1;
// params is cache in UB
static const int64_t TILING_MODE_4 = 4;
// params is cache in L1
static const int64_t TILING_MODE_13 = 13;

// 2. one params row size is greater than or equal to 32B
// params_row is not 32B aligned
static const int64_t TILING_MODE_2 = 2;
// the data of one params row can not store in half UB, need tiling
static const int64_t TILING_MODE_5 = 5;

// 3. params_row is 32B aligned
// params is not cache in UB or L1
static const int64_t TILING_MODE_3 = 3;
// params is cache in UB
static const int64_t TILING_MODE_6 = 6;
// params is cache in L1
static const int64_t TILING_MODE_7 = 7;

// B. block tiling: params_pre tiling
// 1. one params row size is smaller than 32B
// params is not cache
static const int64_t TILING_MODE_8 = 8;
// params is cache in UB
static const int64_t TILING_MODE_9 = 9;

// 2. params_row is 32B aligned
// params is not cache in UB or L1
static const int64_t TILING_MODE_10 = 10;
// params is cache in UB
static const int64_t TILING_MODE_11 = 11;
// params is cache in L1
static const int64_t TILING_MODE_12 = 12;

// tiling_mode with impl_mode
static const int64_t TILING_MODE_14 = 14;

// sampling indices and sorting topN for cache indices
static const int64_t TILING_MODE_15 = 15;
static const int64_t TILING_MODE_15_MAX_PARAM_NUM_SIZE = 1024;
// sampling indices for not align
static const int64_t TILING_MODE_16 = 16;
static const int64_t TILING_MODE_17 = 17;
static const int64_t TILING_MODE_18 = 18;
static const int64_t TILING_MODE_19 = 19;

// tiling_mode with batch_dims
// 1.one params row size is smaller than 32B
// 1.1 params is cached in UB
static const int64_t TILING_MODE_20 = 20;
static const int64_t TILING_MODE_21 = 21;
static const int64_t TILING_MODE_22 = 22;
// 1.2 params is not cached in UB
static const int64_t TILING_MODE_23 = 23;
static const int64_t TILING_MODE_24 = 24;
static const int64_t TILING_MODE_25 = 25;

// 2.one params row size is larger than 32B and not align
static const int64_t TILING_MODE_26 = 26;
static const int64_t TILING_MODE_27 = 27;
static const int64_t TILING_MODE_28 = 28;

// 3.one params row size is align
// 3.1 params is cached in UB
static const int64_t TILING_MODE_29 = 29;
static const int64_t TILING_MODE_30 = 30;
static const int64_t TILING_MODE_31 = 31;
// 3.2 params is not cached in UB
static const int64_t TILING_MODE_32 = 32;
static const int64_t TILING_MODE_33 = 33;
static const int64_t TILING_MODE_34 = 34;

// 4. large params row size
static const int64_t TILING_MODE_35 = 35;
static const int64_t TILING_MODE_36 = 36;
static const int64_t TILING_MODE_37 = 37;

// 5. small indices row size
static const int64_t TILING_MODE_38 = 38;
static const int64_t TILING_MODE_39 = 39;

// 6. small params and indices row size
static const int64_t TILING_MODE_40 = 40;
static const int64_t TILING_MODE_41 = 41;

static const std::vector<std::vector<int64_t>> AD_Trustlist_RT = {{200, 50}, {200, 29}, {200, 1468}, {200, 7},
                                                                  {200, 1}, {200, 10}, {200, 20}, {200, 544},
                                                                  {200, 23}, {200, 5}, {200, 11}, {200, 19},
                                                                  {200, 8}, {32, 39}};

// define impl_mode of gather_v2 attr
static const int64_t IMPL_MODE_HIGH_PERFORMANCE_VALUE = 1;
static const int64_t TILING_MODE_15_MAX_PARAM_NUM = 256;

const gert::Shape g_vec_1_shape = {1};

template <typename T1, typename T2>
static inline auto CeilDiv(T1 a, T2 b) -> T1 {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
static inline auto CeilAlign(T1 a, T2 b) -> T1 {
    return CeilDiv(a, b) * b;
}

template <typename T1, typename T2>
static inline auto min(T1 a, T2 b) -> T1 {
    return a < b ? a : b;
}

template <typename T1, typename T2>
static inline auto max(T1 a, T2 b) -> T1 {
    return a > b ? a : b;
}

static inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

bool CheckAndUpdateAxisAndBatchdims(gert::TilingContext* context, int64_t& axis, int64_t& batch_dims,
                                    int64_t params_dims, int64_t indices_dims) {
  const gert::Shape& xShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
  const gert::Shape& indiesShape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
  OP_CHECK_IF(params_dims <= 0 || indices_dims <= 0,
                  OP_LOGE("GatherV2:", "GatherV2Tiling: params_dims or indices_dims is 0."),
                  return false);

  OP_CHECK_IF(axis < -params_dims || axis >= params_dims,
                  OP_LOGE("GatherV2:", "op GatherV2Tiling: axis is invalid"), return false);

  if (axis < 0) {
    axis += params_dims;
  }

  if (batch_dims != 0) {
    OP_CHECK_IF(batch_dims < -indices_dims || batch_dims > indices_dims,
                    OP_LOGE("GatherV2:", "op GatherV2Tiling: batch_dims is invalid."),
                    return false);
    if (batch_dims < 0) {
      batch_dims += indices_dims;
    }
    OP_CHECK_IF(
        batch_dims >= params_dims,
        OP_LOGE("GatherV2:", "op GatherV2Tiling: batch_dims must be less than rank(params)."),
        return false);
    OP_CHECK_IF(batch_dims > axis,
                    OP_LOGE(
                        "GatherV2:", "op GatherV2Tiling: batch_dims must be less than or equal to axis."),
                    return false);
    for (int64_t i = 0; i < batch_dims; i++) {
      if (xShape.GetDim(i) != indiesShape.GetDim(i)) {
        OP_LOGE("GatherV2",
                "op GatherV2Tiling: Params.shape[:batch_dims] "
                "should be equal to indices.shape[:batch_dims].");
        return false;
      }
    }
  }
  return true;
}

bool ProcessDirectGatherTmpl(GatherV2TilingParams* params,
  const GatherV2CompileInfo* compile_info, int64_t available_ub_size) {
  int64_t x_buffers_size_byte = CeilAlign(available_ub_size / UB_BUF_CNT, BLOCK_SIZE);
  int64_t line_num_once_ub = x_buffers_size_byte / compile_info->params_dsize;

  params->x_split_num = CeilDiv(params->params_axis , line_num_once_ub);
  int64_t params_per_block_num = VECTOR_BLOCK_SIZE / params->params_axis;
  if (params->x_split_num != 1) {
    return false;
  } else if (params->x_split_num == 1) {
    params->x_split_num = 1;
    params->x_buffer_size = params->params_axis;
    params->x_buffer_size = CeilAlign(params->x_buffer_size, params_per_block_num);
    int64_t dszie = compile_info->params_dsize + compile_info->indices_dsize;
    params->indices_buffer_size = (available_ub_size - params->x_buffer_size * compile_info->params_dsize) / dszie;
    params->indices_buffer_size = CeilAlign( params->indices_buffer_size, INT_PER_BLOCK_NUM);
    params->indices_split_num = CeilDiv(params->indices_num , params->indices_buffer_size);
    params->total_count_num = params->indices_split_num * params->params_pre;
  } 
  
  params->count_time = CeilDiv(params->total_count_num, compile_info->core_num);

  int64_t last_time_reserved_size = params->indices_num % (compile_info->core_num * params->indices_buffer_size);
  int64_t last_time_count_num = CeilDiv(last_time_reserved_size, INT_PER_BLOCK_NUM);
  
  int64_t compute_size_of_post_core = last_time_count_num / compile_info->core_num;
  int64_t post_core = last_time_count_num % compile_info->core_num;
  
  params->median_core_id = post_core - 1;
  params->pre_core_compute_size = (compute_size_of_post_core + 1) * INT_PER_BLOCK_NUM ;
  params->median_core_compute_size = compute_size_of_post_core * INT_PER_BLOCK_NUM + last_time_reserved_size - (last_time_count_num - 1) * INT_PER_BLOCK_NUM;
  params->post_core_compute_size = compute_size_of_post_core * INT_PER_BLOCK_NUM;
  
  if (params->indices_split_num >= compile_info->core_num || last_time_count_num >= compile_info->core_num) {
    params->need_core_num = compile_info->core_num;
  } else {
    params->need_core_num = last_time_count_num;
  }
  OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for vgather mode");
  params->tiling_mode = TILING_MODE_19;
  return true;
}

bool DoCacheModeAlignCheck(int64_t axis, const GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  OP_CHECK_IF(
      compile_info->impl_mode != IMPL_MODE_HIGH_PERFORMANCE_VALUE,
      OP_LOGD("GatherV2",
              "[DoCacheModeAlignCheck] no need simpiling for topn cache, becase current is not high_performance"),
      return false);
  OP_CHECK_IF(
      params->indices_num <= INDICES_MIN_NUM_FOR_CACHE,
      OP_LOGD("GatherV2", "[DoCacheModeAlignCheck] no need simpiling for topn cache, but indices_num is %ld",
              params->indices_num),
      return false);
  OP_CHECK_IF(
      axis != 0, OP_LOGD("GatherV2", "[DoCacheModeAlignCheck] no need simpiling for topn cache, but axis is %ld", axis),
      return false);
  OP_CHECK_IF(
      compile_info->core_num == 0,
      OP_LOGD("GatherV2", "[DoCacheModeAlignCheck] no need simpiling for topn cache, but need_core_num is %ld",
              compile_info->core_num),
      return false);

  // if input param size less than cache n number buffer, no need cache mode
  int64_t cache_n_num_max_size = (compile_info->ub_size - RESERVED_UB_SIZE) / CACHE_MODE_UB_SLICE;
  cache_n_num_max_size = cache_n_num_max_size / BLOCK_SIZE * BLOCK_SIZE;
  OP_CHECK_IF(
      params->params_total * compile_info->params_dsize < cache_n_num_max_size,
      OP_LOGD("GatherV2", "[DoCacheModeAlignCheck] no need simpiling for topn cache, but cache_n_num_max_size is %ld",
              cache_n_num_max_size),
      return false);

  int64_t one_param_row_size = params->params_row * compile_info->params_dsize;
  OP_CHECK_IF(
      ((one_param_row_size > TILING_MODE_15_MAX_PARAM_NUM_SIZE) || (one_param_row_size < BLOCK_SIZE) ||
       (one_param_row_size % BLOCK_SIZE != 0) || (one_param_row_size * BLOCK_SIZE > cache_n_num_max_size)),
      OP_LOGD("GatherV2",
              "[DoCacheModeTilingAlign] no need cache mode, but params_row_size is %ld, cache_n_num_max_size is %ld",
              one_param_row_size, cache_n_num_max_size),
      return false);

  return true;
}

bool DoCacheModeTilingAlign(int64_t axis, GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  if (!DoCacheModeAlignCheck(axis, params, compile_info)) {
    return false;
  }

  params->tiling_mode = TILING_MODE_15;
  params->need_core_num = compile_info->core_num;
  params->indices_num_each_core = (params->indices_num + params->need_core_num - 1) / params->need_core_num;
  params->indices_num_remaining = params->indices_num / params->need_core_num;

  params->tail_process_core = params->indices_num % params->need_core_num;
  if (params->tail_process_core == 0) {
    params->tail_process_core = params->need_core_num;
  }
  OP_LOGD("GatherV2", "[DoCacheModeTilingAlign] For the core which blockId < %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_each_core);
  OP_LOGD("GatherV2", "[DoCacheModeTilingAlign] For the core which blockId >= %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_remaining);

  return true;
}

bool DoCacheModeNotAlignCheck(int64_t axis, int64_t one_param_row_size, const GatherV2TilingParams* params,
                              const GatherV2CompileInfo* compile_info) {
  OP_CHECK_IF(
      compile_info->impl_mode != IMPL_MODE_HIGH_PERFORMANCE_VALUE,
      OP_LOGD("GatherV2",
              "[DoCacheModeNotAlignCheck] no need simpiling for topn cache, becase current is not high_performance"),
      return false);
  OP_CHECK_IF(compile_info->params_dsize < SUPPORT_PARAM_SIZE,
                  OP_LOGD("GatherV2", "[DoCacheModeNotAlignCheck] params not support uint8/int8, but type size is :%ld",
                          compile_info->params_dsize),
                  return false);
  OP_CHECK_IF(
      (one_param_row_size % BLOCK_SIZE == 0),
      OP_LOGD("GatherV2", "[DoCacheModeNotAlignCheck] no need cache mode, but params_row is %ld", params->params_row),
      return false);
  OP_CHECK_IF(
      axis != 0,
      OP_LOGD("GatherV2", "[DoCacheModeNotAlignCheck] no need simpiling for topn cache, but axis is %ld", axis),
      return false);
  OP_CHECK_IF(
      compile_info->core_num == 0,
      OP_LOGD("GatherV2", "[DoCacheModeNotAlignCheck] no need simpiling for topn cache, but need_core_num is %ld",
              compile_info->core_num),
      return false);
  int64_t six_part_ub_size = (compile_info->ub_size - RESERVED_UB_SIZE_2K) / CACHE_MODE_UB_SLICE;
  six_part_ub_size = six_part_ub_size / BLOCK_SIZE * BLOCK_SIZE;
  int64_t align_param_row = (one_param_row_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  OP_CHECK_IF(
      (align_param_row * TRANS_POSE_LINE_SIZE > six_part_ub_size),
      OP_LOGD("GatherV2",
              "[DoCacheModeNotAlignCheck] no need cache mode, but align_param_row is %ld, params_row is %ld",
              align_param_row, params->params_row),
      return false);

  return true;
}

bool DoCacheModeTilingNotAlian(int64_t axis, GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  int64_t one_param_row_size = params->params_row * compile_info->params_dsize;
  if (!DoCacheModeNotAlignCheck(axis, one_param_row_size, params, compile_info)) {
    return false;
  }

  params->need_core_num = compile_info->core_num;
  params->indices_num_each_core = (params->indices_num + params->need_core_num - 1) / params->need_core_num;
  params->indices_num_remaining = params->indices_num / params->need_core_num;
  OP_CHECK_IF(params->indices_num_remaining * one_param_row_size < BLOCK_SIZE,
                  OP_LOGD("GatherV2",
                          "[DoCacheModeTilingNotAlian] no need simpiling for topn cache, but indices_num_each_core is "
                          "%ld, params_d_size is %ld, need_core_num is %ld",
                          params->indices_num_remaining, compile_info->params_dsize, params->need_core_num),
                  return false);

  params->tail_process_core = params->indices_num % params->need_core_num;
  if (params->tail_process_core == 0) {
    params->tail_process_core = params->need_core_num;
  }
  params->tiling_mode = TILING_MODE_16;

  OP_LOGD("GatherV2", "[DoCacheModeTilingNotAlian] For the core which blockId < %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_each_core);
  OP_LOGD("GatherV2", "[DoCacheModeTilingNotAlian] For the core which blockId >= %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_remaining);

  return true;
}

bool DoCpuPreprocessCheck(int64_t axis, int64_t one_param_row_size, int64_t part_ub_size,
                          const GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  OP_CHECK_IF(
      part_ub_size == 0,
      OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] no need simpiling for topn cache, but part_ub_size is %ld",
              part_ub_size),
      return false);
  OP_CHECK_IF(
      compile_info->is_preprocessed,
      OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] no need simpiling for topn cache, but is_preprocessed is %d",
              compile_info->is_preprocessed),
      return false);

  OP_CHECK_IF((compile_info->params_dsize < SUPPORT_PARAM_SIZE) && (one_param_row_size % BLOCK_SIZE != 0),
                  OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] params not support uint8/int8, but type size is :%ld",
                          one_param_row_size),
                  return false);
  OP_CHECK_IF(axis != 0,
                  OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] no need simpiling for topn cache, but axis is %ld", axis),
                  return false);
  OP_CHECK_IF(
      compile_info->core_num == 0,
      OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] no need simpiling for topn cache, but need_core_num is %ld",
              compile_info->core_num),
      return false);
  int64_t one_part_ub_size = (compile_info->ub_size - RESERVED_UB_SIZE_2K) / part_ub_size;
  one_part_ub_size = one_part_ub_size / BLOCK_SIZE * BLOCK_SIZE;
  int64_t align_param_row = (one_param_row_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  int64_t row_num_once_ub = 0;
  if (one_param_row_size % BLOCK_SIZE != 0) {
    OP_CHECK_IF(
        (align_param_row * TRANS_POSE_LINE_SIZE > one_part_ub_size),
        OP_LOGD("GatherV2", "[DoCpuPreprocessCheck] no need cache mode, but align_param_row is %ld, params_row is %ld",
                align_param_row, params->params_row),
        return false);
    row_num_once_ub = one_part_ub_size / (align_param_row * TRANS_POSE_LINE_SIZE) / ALIGN_FOR_ONCE_UB_SIZE;

    OP_CHECK_IF(
        (row_num_once_ub == 0),
        OP_LOGD("GatherV2",
                "[DoCpuPreprocessCheck] row_num_once_ub is 0, but one_part_ub_size is %ld, align_param_row is %ld",
                one_part_ub_size, align_param_row),
        return false);
  } else {
    OP_CHECK_IF(
        (align_param_row > one_part_ub_size),
        OP_LOGD(
            "GatherV2",
            "[DoCacheModeTilingwithCpuPreprocess] no need cache mode, but align_param_row is %ld, params_row is %ld",
            align_param_row, params->params_row),
        return false);
    row_num_once_ub = one_part_ub_size / align_param_row / ALIGN_FOR_ONCE_UB_SIZE;
    OP_CHECK_IF(
        (row_num_once_ub == 0),
        OP_LOGD("GatherV2",
                "[DoCpuPreprocessCheck] row_num_once_ub is 0, but one_part_ub_size is %ld, align_param_row is %ld",
                one_part_ub_size, align_param_row),
        return false);
  }

  return true;
}

static void CalNeedCoreForCacheModeTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compileParams) {
  int64_t oneParamRowSize = params->params_row * compileParams->params_dsize;

  // this core_num for tiling_mode_18 is gained from actual tests
  if (compileParams->socVersion != 0 && oneParamRowSize % BLOCK_SIZE == 0) {
    if (params->indices_num * oneParamRowSize <= SIZE_FOR_ONE_CORE) {
      params->need_core_num = ONE_CORE;
    } else if (params->indices_num * oneParamRowSize <= SIZE_FOR_EIGHT_CORE) {
      params->need_core_num = EIGHT_CORES;
    } else {
      params->need_core_num = compileParams->core_num;
    }
  } else {
    params->need_core_num = compileParams->core_num;
  }

  OP_LOGD("GatherV2", "[CalNeedCoreForCacheModeTiling] the number of cores used is %ld", params->need_core_num);
}

bool DoCacheModeTilingwithCpuPreprocess(int64_t axis, GatherV2TilingParams* params,
                                        const GatherV2CompileInfo* compile_info) {
  OP_CHECK_IF(compile_info->impl_mode != IMPL_MODE_HIGH_PERFORMANCE_VALUE,
                  OP_LOGD("GatherV2", "[DoCacheModeTilingwithCpuPreprocess] no need cpu cache, not high_performance"),
                  return false);

  int64_t one_param_row_size = params->params_row * compile_info->params_dsize;

  if (one_param_row_size % BLOCK_SIZE != 0) {
    if (!DoCpuPreprocessCheck(axis, one_param_row_size, FOUR_PART_UB_SIZE, params, compile_info)) {
      return false;
    }
  } else {
    if (!DoCpuPreprocessCheck(axis, one_param_row_size, THREE_PART_UB_SIZE, params, compile_info)) {
      return false;
    }
  }

  CalNeedCoreForCacheModeTiling(params, compile_info);
  params->indices_num_each_core = (params->indices_num + params->need_core_num - 1) / params->need_core_num;
  params->indices_num_remaining = params->indices_num / params->need_core_num;

  OP_CHECK_IF(
      params->indices_num_remaining * one_param_row_size < BLOCK_SIZE,
      OP_LOGD("GatherV2",
              "[DoCacheModeTilingwithCpuPreprocess] no need simpiling for topn cache, but indices_num_remaining is "
              "%ld, params_dsize is %ld, need_core_num is %ld",
              params->indices_num_remaining, compile_info->params_dsize, params->need_core_num),
      return false);
  params->tail_process_core = params->indices_num % params->need_core_num;
  if (params->tail_process_core == 0) {
    params->tail_process_core = params->need_core_num;
  }

  if (one_param_row_size % BLOCK_SIZE != 0) {
    params->tiling_mode = TILING_MODE_17;
  } else {
    params->tiling_mode = TILING_MODE_18;
  }

  OP_LOGD("GatherV2",
          "[DoCacheModeTilingwithCpuPreprocess] For the core which blockId < %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_each_core);
  OP_LOGD("GatherV2",
          "[DoCacheModeTilingwithCpuPreprocess] For the core which blockId >= %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_remaining);

  return true;
}

static bool DoImplModeTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  OP_CHECK_IF(
      compile_info->impl_mode != IMPL_MODE_HIGH_PERFORMANCE_VALUE,
      OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for impl_mode is not high_performance"),
      return false);
  OP_CHECK_IF(
      params->params_total * compile_info->params_dsize <= PARAMS_CACHED_UB,
      OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for all params can be cached in UB"),
      return false);
  OP_CHECK_IF(params->indices_num < compile_info->core_num * BLOCK_SIZE / compile_info->params_dsize,
                  OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for the num of indices is small"),
                  return false);

  params->tiling_mode = TILING_MODE_14;
  params->need_core_num = compile_info->core_num;
  params->indices_num_each_core = (params->indices_num + params->need_core_num - 1) / params->need_core_num;
  params->indices_num_remaining = params->indices_num / params->need_core_num;

  params->tail_process_core = params->indices_num % params->need_core_num;
  if (params->tail_process_core == 0) {
    params->tail_process_core = params->need_core_num;
  }
  OP_LOGD("GatherV2", "[DoImplModeTiling] For the core which blockId < %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_each_core);
  OP_LOGD("GatherV2", "[DoImplModeTiling] For the core which blockId >= %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_remaining);

  return true;
}

// compute tiling params for tiling_mode 10&11&12
bool BlockAlignForParamsTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_dsize) {
  OP_CHECK_IF(indices_num_per_loop == 0,
                  OP_LOGE("GatherV2:", "indices_num_per_loop = 0 is not support"),
                  return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_dsize);
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("GatherV2:", "Devide by row_num_once_ub[%ld] exception.",
                            params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_dsize);
  OP_CHECK_IF((params->row_num_last_ub == 0),
                  OP_LOGE("GatherV2:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  return true;
}

// compute tiling params for tiling_mode 1&4&13
bool BlockLessForIndicesTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_d_size, int64_t block_num) {
  OP_CHECK_IF(indices_num_per_loop == 0 || block_num == 0,
                  OP_LOGE("GatherV2:", "indices_num_per_loop or block_num = 0 is not support"),
                  return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_d_size);
  if (int(params->row_num_once_ub % block_num) != 0) {
    params->row_num_once_ub = int(params->row_num_once_ub / block_num) * block_num;
  }
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }
  if (params->inner_loop_num > 0 && params->row_num_once_tail_ub > 0 &&
      params->row_num_once_tail_ub * params->params_row < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_d_size);
  if (int(params->row_num_last_ub % block_num) != 0) {
    params->row_num_last_ub = int(params->row_num_last_ub / block_num) * block_num;
  }
  OP_CHECK_IF((params->row_num_last_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  if (params->inner_loop_num_last > 0 && params->row_num_last_tail_ub > 0 &&
      params->row_num_last_tail_ub * params->params_row < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  OP_LOGD("gatherv2", "BlockLessForIndicesTiling END");
  return true;
}

// compute tiling params for tiling_mode 8&9
bool BlockLessForParamsTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                              int64_t params_dsize, int64_t block_num) {
  OP_CHECK_IF(
      indices_num_per_loop == 0 || block_num == 0,
      OP_LOGE("Gather Tiling:", "indices_num_per_loop or block_num = 0 is not support"),
      return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_dsize);
  if (int(params->row_num_once_ub % block_num) != 0) {
    params->row_num_once_ub = int(params->row_num_once_ub / block_num) * block_num;
  }
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }
  if (params->inner_loop_num > 0 && params->row_num_once_tail_ub > 0 &&
      params->row_num_once_tail_ub * params->params_row < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_dsize);
  if (int(params->row_num_last_ub % block_num) != 0) {
    params->row_num_last_ub = int(params->row_num_last_ub / block_num) * block_num;
  }
  OP_CHECK_IF((params->row_num_last_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  if (params->inner_loop_num_last > 0 && params->row_num_last_tail_ub > 0 &&
      params->row_num_last_tail_ub * params->params_row < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  return true;
}

static void CalNeedCore(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  while (params->need_core_num > 1) {
    params->need_core_num = params->need_core_num / DOUBLE;
    params->indices_num_each_core = params->indices_num / params->need_core_num;
    params->indices_num_remaining = params->indices_num % params->need_core_num;
    if (params->indices_num_each_core * params->params_row * compile_info->params_dsize > BLOCK_SIZE) {
      break;
    }
  }
}

// compute tiling params for tiling_mode 3&6&7
bool BlockAlignForIndicesTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                                int64_t params_d_size) {
  if (indices_num_per_loop == 0) {
    OP_LOGE("gather_v2", "indices_num_per_loop = 0 is not support");
    return false;
  }
  params->indices_loop_num = (params->indices_num_each_core) / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
    params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
  }

  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }

  params->row_num_last_ub = res_ub_size / ((params->params_row) * params_d_size);
  OP_CHECK_IF((params->row_num_last_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = (params->indices_row_num_last) / (params->row_num_last_ub);
  if ((params->indices_row_num_last) % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = (params->indices_row_num_last) % (params->row_num_last_ub);
  }
  return true;
}

bool ParamsPreTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_size,
                     int64_t half_remain_ub_size, int64_t params_total_ceil, int64_t params_row_ceil) {
  params->need_core_num = compile_info->core_num;
  params->tail_process_core = 0;
  params->params_pre_each_core = (params->params_pre) / (params->need_core_num);
  params->params_pre_remaining = (params->params_pre) % (params->need_core_num);
  params->indices_num_each_core = params->indices_num;
  int64_t half_remain_params_elem = half_remain_ub_size / (compile_info->params_dsize);
  int64_t res_ub_size = half_ub_size;
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);

  if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    params->need_core_num = 1;
    params->tail_process_core = 0;
    params->params_pre_each_core = params->params_pre;
    params->params_pre_remaining = 0;
  }

  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize)) {
      params->tiling_mode = TILING_MODE_8;
    } else {
      params->tiling_mode = TILING_MODE_9;
    }

    if (params->tiling_mode == TILING_MODE_8) {
      indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
      res_ub_size = half_remain_ub_size;
    }

    if (!BlockLessForParamsTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize, block_num)) {
      return false;
    }
  } else {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
        params_row_ceil <= half_remain_params_elem) {
      params->tiling_mode = TILING_MODE_10;
    } else if (params_total_ceil <= (compile_info->l1_size) / (compile_info->params_dsize)) {
      params->tiling_mode = TILING_MODE_11;
    } else {
      params->tiling_mode = TILING_MODE_12;
    }

    if (params->tiling_mode == TILING_MODE_10) {
      indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
      res_ub_size = half_remain_ub_size;
    }

    if (!BlockAlignForParamsTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize)) {
      return false;
    }
  }
  return true;
}

bool ParamsSmall32B(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t params_total_ceil,
                    int64_t indices_num_per_loop, int64_t half_remain_ub_size, int64_t res_ub_size, int64_t block_num) {
  if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize)) {
    params->tiling_mode = TILING_MODE_4;
  } else if (params_total_ceil <= ((compile_info->l1_size) / (compile_info->params_dsize))) {
    params->tiling_mode = TILING_MODE_13;
  } else {
    params->tiling_mode = TILING_MODE_1;
  }
  if (((params->params_row) < BLOCK_SIZE) &&
      ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE)) {
    CalNeedCore(params, compile_info);
  }

  if (params->tiling_mode == TILING_MODE_4) {
    indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
    res_ub_size = half_remain_ub_size;
  }

  if (!BlockLessForIndicesTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize, block_num)) {
    OP_LOGE("GatherV2", "BlockLessForIndicesTiling is false");
    return false;
  }
  return true;
}

bool ParamsGreater32B(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info,
                      int64_t half_ub_params_elem, int64_t half_remain_ub_size, int64_t half_ub_size,
                      int64_t params_total_ceil, int64_t params_row_ceil) {
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t half_remain_params_elem = half_remain_ub_size / (compile_info->params_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t res_ub_size = half_ub_size;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  int64_t mode_7_gate_value = ACTUAL_NUM - GATE_VALUE * params->params_total / DATA_VALUE;
  if (params_row_ceil <= half_ub_params_elem) {
    if ((params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE != 0) {  // not 32B aligned
      params->tiling_mode = TILING_MODE_2;

      params->indices_loop_num = (params->indices_num_each_core) / half_ub_indices_elem;
      params->indices_row_num_once = half_ub_indices_elem;
      if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
        params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
      }
    } else {  // 32B aligned
      if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
          params_row_ceil <= half_remain_params_elem) {
        params->tiling_mode = TILING_MODE_6;
      } else if (params_total_ceil <= (compile_info->l1_size) / (compile_info->params_dsize) &&
                 (params->indices_num) > mode_7_gate_value) {
        params->tiling_mode = TILING_MODE_7;
      } else {
        params->tiling_mode = TILING_MODE_3;
      }
      if (params->tiling_mode == TILING_MODE_6) {
        indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
        res_ub_size = half_remain_ub_size;
      }

      if (!BlockAlignForIndicesTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize)) {
        return false;
      }
    }
  } else {
    params->tiling_mode = TILING_MODE_5;  // one params row need tiling

    params->indices_loop_num = params->indices_num_each_core / half_ub_indices_elem;
    params->indices_row_num_once = indices_num_per_loop;
    if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
      params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
    }

    params->one_row_loop = (params->params_row) / half_ub_params_elem;
    params->one_row_tail = (params->params_row) % half_ub_params_elem;
    if (params->one_row_loop > 0 && (params->one_row_tail) > 0 && (params->one_row_tail) < block_num) {
      params->one_row_loop = (params->one_row_loop) - 1;
      params->one_row_tail = half_ub_params_elem + (params->one_row_tail);
    }
  }
  return true;
}

bool ParamsIndicesTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_size,
                         int64_t half_remain_ub_size, int64_t half_ub_params_elem, int64_t params_total_ceil,
                         int64_t params_row_ceil) {
  params->need_core_num = compile_info->core_num;
  params->tail_process_core = 0;
  params->indices_num_each_core = (params->indices_num) / (params->need_core_num);
  params->indices_num_remaining = (params->indices_num) % (params->need_core_num);
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t res_ub_size = half_ub_size;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  if (params->indices_num <= params->need_core_num) {
    params->need_core_num = params->indices_num;
    params->tail_process_core = 0;
    params->indices_num_each_core = 1;
    params->indices_num_remaining = 0;
  }

  // one params row size is smaller than 32B
  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (!ParamsSmall32B(params, compile_info, params_total_ceil, indices_num_per_loop, half_remain_ub_size, res_ub_size,
                        block_num)) {
      OP_LOGE("GatherV2", "ParamsSmall32B is false");
      return false;
    }
  } else {  // one params row size is greater than or equal to 32B
    if (!ParamsGreater32B(params, compile_info, half_ub_params_elem, half_remain_ub_size, half_ub_size,
                          params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsGreater32B is false");
      return false;
    }
  }
  return true;
}

static bool CheckADTrustListForRt2(const gert::Shape& indices_shape) {
  std::vector<int64_t> v_indices_shape;
  int64_t indices_dim = indices_shape.GetDimNum();

  for (int64_t index = 0; index < indices_dim; index++) {
    v_indices_shape.push_back(indices_shape.GetDim(index));
  }

  for (auto it = AD_Trustlist_RT.begin(); it != AD_Trustlist_RT.end(); it++) {
    if (*it == v_indices_shape) {
      OP_LOGD("GatherV2", "[CheckADTrustListForRt2] match the AD_Trustlist, no need EMB");
      return true;
    }
  }

  return false;
}

int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end) {
  int64_t size = 1;
  for (size_t i = begin; i < end; i++) {
    size *= shape[i];
  }
  return size;
}

void CalcTilingWithoutBatchDims(const gert::TilingContext* context, GatherV2TilingParams* params, int64_t axis,
                                int64_t paramsDims) {
  auto& x_shape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
  auto& indies_shape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
  int64_t indices_dims = indies_shape.GetDimNum();
  if (axis == 0) {
    params->params_pre = 1;
  } else {
    for (int64_t i = 0; i < axis; i++) {
      params->params_pre *= x_shape.GetDim(i);
    }
  }
  params->params_axis = x_shape.GetDim(axis);

  if (axis + 1 < paramsDims) {
    for (int64_t i = axis + 1; i < paramsDims; i++) {
      params->params_row *= x_shape.GetDim(i);
    }
  } else {
    params->params_row = 1;
  }
  params->params_total = GetPartShapeSize(x_shape, 0, paramsDims);
  for (int i = 0; i < indices_dims; i++) {
    params->indices_num = (params->indices_num) * indies_shape.GetDim(i);
  }

  return;
}

bool TilingForNano(const gert::TilingContext* context, const GatherV2CompileInfo* compile_info,
                            GatherV2TilingParams* params, int64_t axis, int64_t paramsDims) {
  int64_t available_ub_size = (compile_info->ub_size) - RESERVED_UB_SIZE_2K;  // reserved 2K
  int64_t half_ub_size = available_ub_size / 2;
  // params shape convert to 3D:[params_pre, params_axis, params_row]  indies_shape.GetDimNum();
  // indices shape convert to 1D:[indices_num]
  // output tensor, y shape convert to:[params_pre, indices_num, params_row]
  CalcTilingWithoutBatchDims(context, params, axis, paramsDims);
  int64_t block_num = (compile_info->block_size) / (compile_info->params_dsize);
  int64_t params_row_ceil = ((params->params_row) + block_num - 1) / block_num * block_num;
  int64_t half_ub_params_elem = half_ub_size / (compile_info->params_dsize);

  if (half_ub_params_elem == 0) {
    OP_LOGE("GatherV2", "half_ub_params_elem is 0");
    return false;
  }

  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  params->indices_loop_num = (params->indices_num) / half_ub_indices_elem;
  params->indices_row_num_once = half_ub_indices_elem;
  if ((params->indices_num) % (params->indices_row_num_once) != 0) {
    params->indices_row_num_last = (params->indices_num) % (params->indices_row_num_once);
   }

  if (params_row_ceil <= half_ub_params_elem) {
    params->tiling_mode = TILING_MODE_2;
  } else {
    params->one_row_loop = (params->params_row) / half_ub_params_elem;
    params->one_row_tail = (params->params_row) % half_ub_params_elem;
    if (params->one_row_loop > 0 && (params->one_row_tail) > 0 && (params->one_row_tail) < block_num) {
      params->one_row_loop = (params->one_row_loop) - 1;
      params->one_row_tail = half_ub_params_elem + (params->one_row_tail);
    }
    params->tiling_mode = TILING_MODE_5;
  }
  return true;
}

bool TilingWithoutBatchDims(const gert::TilingContext* context, const GatherV2CompileInfo* compile_info,
                            GatherV2TilingParams* params, int64_t axis, int64_t paramsDims) {
  int64_t available_ub_size = (compile_info->ub_size) - RESERVED_UB_SIZE_2K;  // reserved 2K
  int64_t half_ub_size = available_ub_size / 2;
  // params shape convert to 3D:[params_pre, params_axis, params_row]  indies_shape.GetDimNum();
  // indices shape convert to 1D:[indices_num]
  // output tensor, y shape convert to:[params_pre, indices_num, params_row]
  auto& indies_shape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());

  CalcTilingWithoutBatchDims(context, params, axis, paramsDims);
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  int64_t params_total_ceil = ((params->params_total) + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = ((params->params_row) + block_num - 1) / block_num * block_num;

  int64_t half_remain_ub_size = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_ub_params_elem = half_ub_size / (compile_info->params_dsize);
  if (half_ub_params_elem == 0) {
    OP_LOGE("GatherV2", "half_ub_params_elem is 0");
    return false;
  }

  if (params->params_row == 1 && (compile_info->params_dsize == sizeof(int16_t) 
      || compile_info->params_dsize == sizeof(int32_t)) && compile_info->socVersion != 0
      && compile_info->indices_dsize == sizeof(int32_t)  && ProcessDirectGatherTmpl(params, compile_info, available_ub_size)) {
    return true;
  } 

  if (!CheckADTrustListForRt2(indies_shape) && DoCacheModeTilingwithCpuPreprocess(axis, params, compile_info)) {
    OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for topn cache is DoCacheMode with CpuPreprocess");
    return true;
  }

  if (DoCacheModeTilingAlign(axis, params, compile_info)) {
    OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for DoCacheModeTilingAlign");
    return true;
  }

  if (DoCacheModeTilingNotAlian(axis, params, compile_info)) {
    OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for DoCacheModeTilingNotAlian");
    return true;
  }

  // the data of the formula gained from actual tests
  // set a gate value for tiling_mode_7 to optimized some data_move processes

  if (DoImplModeTiling(params, compile_info)) {
    OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for impl_mode is high_performance");
    return true;
  }

  if (params->params_pre >= (compile_info->core_num) && params_row_ceil <= half_ub_params_elem &&
      ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE ||
       (params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE == 0)) {
    if (!ParamsPreTiling(params, compile_info, half_ub_size, half_remain_ub_size, params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsPreTiling is false");
      return false;
    }
  } else {
    if (!ParamsIndicesTiling(params, compile_info, half_ub_size, half_remain_ub_size, half_ub_params_elem,
                             params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsIndicesTiling is false");
      return false;
    }
  }
  return true;
}

bool LargeRowProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_params_elem,
                     int64_t half_size_ub) {
  OP_CHECK_IF(half_ub_params_elem == 0,
                  OP_LOGE("GatherV2:", "half_ub_params_elem = 0 is not support"), return false);
  params->one_row_loop = (params->params_row) / half_ub_params_elem;
  params->one_row_tail = (params->params_row) % half_ub_params_elem;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  if ((params->one_row_loop) > 0 && (params->one_row_tail) > 0 && (params->one_row_tail) < block_num) {
    params->one_row_loop = (params->one_row_loop) - 1;
    params->one_row_tail = half_ub_params_elem + (params->one_row_tail);
  }

  if ((params->params_batch_each_core) * (params->indices_row) * (compile_info->indices_dsize) <= half_size_ub) {
    params->indices_row_num_once = params->indices_row;
    params->tiling_mode = TILING_MODE_35;
  } else if ((params->indices_row) * (compile_info->indices_dsize) <= half_size_ub) {
    params->indices_row_num_once = params->indices_row;
    params->tiling_mode = TILING_MODE_36;
  } else {
    int64_t indices_num_per_loop = half_size_ub / (compile_info->indices_dsize);
    params->indices_loop_num = (params->indices_row) / indices_num_per_loop;
    params->indices_row_num_once = indices_num_per_loop;
    if ((params->indices_row) % (params->indices_row_num_once) != 0) {
      params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
    }
    params->tiling_mode = TILING_MODE_37;
  }
  return true;
}

bool CalcCacheIndices(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                      int64_t params_d_size, int64_t tiling_mode) {
  OP_CHECK_IF(params_d_size == 0, OP_LOGE("GatherV2:", "params_d_size= 0 is not support"),
                  return false);
  params->indices_row_num_once = indices_num_per_loop;
  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  int64_t block_num = BLOCK_SIZE / params_d_size;
  int64_t align_unit;
  if (tiling_mode == TILING_MODE_38 || tiling_mode == TILING_MODE_39) {
    align_unit = params->indices_row * block_num;
  } else if (tiling_mode == TILING_MODE_40 || tiling_mode == TILING_MODE_41) {
    align_unit = (params->params_pre) * (params->indices_row) * block_num;
  } else if ((params->params_row) * params_d_size >= BLOCK_SIZE) {
    align_unit = 1;
  } else {
    align_unit = block_num;
  }

  if (int((params->row_num_once_ub) % align_unit) != 0) {
    params->row_num_once_ub = int((params->row_num_once_ub) / align_unit) * align_unit;
  }
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num) > 0 && (params->row_num_once_tail_ub) > 0 &&
      (params->row_num_once_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num = (params->inner_loop_num) - 1;
    params->row_num_once_tail_ub = (params->row_num_once_tail_ub) + (params->row_num_once_ub);
  }
  params->tiling_mode = tiling_mode;
  return true;
}

bool CalcWithBatchDims(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                       int64_t params_d_size) {
  if (indices_num_per_loop == 0 || params_d_size == 0) {
    OP_LOGE("gather_v2", "indices_num_per_loop or params_d_size= 0 is not support");
    return false;
  }
  params->indices_loop_num = (params->indices_row) / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  int64_t block_num = BLOCK_SIZE / params_d_size;
  if ((params->params_row) * params_d_size >= BLOCK_SIZE) {
    block_num = 1;
  }
  if ((params->indices_row) % (params->indices_row_num_once) != 0) {
    params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
  }
  if ((params->indices_loop_num) > 0 &&
      (params->indices_row_num_last) * (params->indices_row) * (params->params_row) < block_num) {
    params->indices_loop_num -= 1;
    params->indices_row_num_last += params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  if (int((params->row_num_once_ub) % block_num) != 0) {
    params->row_num_once_ub = int((params->row_num_once_ub) / block_num) * block_num;
  }
  OP_CHECK_IF((params->row_num_once_ub == 0),
                  OP_LOGE("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num) > 0 && (params->row_num_once_tail_ub) > 0 &&
      (params->row_num_once_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = params->row_num_once_ub;
  params->inner_loop_num_last = (params->indices_row_num_last) / (params->row_num_once_ub);
  if ((params->indices_row_num_last) % (params->row_num_once_ub) != 0) {
    params->row_num_last_tail_ub = (params->indices_row_num_last) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num_last) > 0 && (params->row_num_last_tail_ub) > 0 &&
      (params->row_num_last_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  return true;
}

bool IndicesCachedProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t aval_ub_size,
                          int64_t mode_cache_all, int64_t mode_cache_row, int64_t mode_without_cache) {
  int64_t indices_num_per_loop = 1;
  if (params->params_batch_each_core * params->indices_row * compile_info->indices_dsize <= aval_ub_size) {
    indices_num_per_loop = params->indices_row;
    if (!CalcCacheIndices(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize, mode_cache_all)) {
      return false;
    }
  } else if (params->indices_row * compile_info->indices_dsize <= aval_ub_size) {
    indices_num_per_loop = params->indices_row;
    if (!CalcCacheIndices(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize, mode_cache_row)) {
      return false;
    }
  } else {
    indices_num_per_loop = aval_ub_size / compile_info->indices_dsize;
    params->tiling_mode = mode_without_cache;
    if (!CalcWithBatchDims(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize)) {
      return false;
    }
  }
  return true;
}

bool SmallRowProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t mode_with_cache,
                     int64_t mode_without_cache, int64_t half_remain_size_ub, int64_t half_size_ub) {
  if (mode_with_cache == TILING_MODE_38 || mode_without_cache == TILING_MODE_39) {
    params->params_batch_each_core = params->params_pre / params->need_core_num;
    params->params_batch_remaining = params->params_pre % params->need_core_num;
  }
  params->tail_process_core = params->need_core_num - 1;
  params->indices_num_each_core = params->params_batch_each_core * params->indices_row;
  params->indices_num_remaining = 0;
  int64_t block_num = BLOCK_SIZE / compile_info->params_dsize;
  int64_t indices_num_per_loop = params->indices_num_each_core;
  int64_t params_total_ceil = (params->params_total + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = (params->params_row + block_num - 1) / block_num * block_num;
  int64_t half_remain_params_elem = half_remain_size_ub / compile_info->params_dsize;
  if (params_total_ceil <= PARAMS_CACHED_UB / compile_info->params_dsize &&
      params_row_ceil <= half_remain_params_elem) {
    if (!CalcCacheIndices(params, indices_num_per_loop, half_remain_size_ub, compile_info->params_dsize,
                          mode_with_cache)) {
      return false;
    }
  } else {
    if (!CalcCacheIndices(params, indices_num_per_loop, half_size_ub, compile_info->params_dsize, mode_without_cache)) {
      return false;
    }
  }
  return true;
}

static void CalNeedCoreWithBatchDims(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  while (params->need_core_num > 1) {
    params->need_core_num = params->need_core_num / DOUBLE;
    params->params_batch_each_core = params->params_batch / params->need_core_num;
    params->params_batch_remaining = params->params_batch % params->need_core_num;
    params->indices_num_each_core = params->params_batch_each_core * params->indices_row;
    params->indices_num_remaining = params->params_batch_remaining * params->indices_row;
    if (params->indices_num_each_core * params->params_pre * params->params_row * compile_info->params_dsize >
        BLOCK_SIZE) {
      break;
    }
  }
}

void ParasPreProcess(GatherV2TilingParams* params, const gert::TilingContext* context, int64_t axis, int64_t batch_dims,
                     int64_t& indices_batch) {
  auto& x_shape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
  auto& indices_shape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
  int64_t indices_dims = indices_shape.GetDimNum();

  // params shape convert to 4D:[params_batch, params_pre, params_axis, params_row]
  // indices shape convert to 1D:[indices_batch, indices_row]
  // output tensor, y shape convert to:[params_batch, params_pre, indices_row, params_row]
  for (int64_t i = 0; i < batch_dims; i++) {
    indices_batch = indices_batch * indices_shape.GetDim(i);
  }
  params->params_batch = indices_batch;
  for (int64_t i = batch_dims; i < indices_dims; i++) {
    params->indices_row = (params->indices_row) * indices_shape.GetDim(i);
  }

  if (axis == batch_dims) {
    params->params_pre = 1;
  } else {
    for (int64_t i = batch_dims; i < axis; i++) {
      params->params_pre = (params->params_pre) * x_shape.GetDim(i);
    }
  }
  params->params_axis = x_shape.GetDim(axis);
  int64_t params_dims = x_shape.GetDimNum();
  if (axis + 1 < params_dims) {
    for (int64_t i = axis + 1; i < params_dims; i++) {
      params->params_row = (params->params_row) * x_shape.GetDim(i);
    }
  } else {
    params->params_row = 1;
  }

  for (int64_t i = 0; i < indices_dims; i++) {
    params->indices_num = (params->indices_num) * indices_shape.GetDim(i);
  }
}

bool WithBatchDimsSmall(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info,
                        int64_t half_remain_params_elem, int64_t half_size_ub, int64_t params_total_ceil,
                        int64_t params_row_ceil) {
  int64_t available_ub_size = compile_info->ub_size - RESERVED_UB_SIZE_2K;  // reserved 2K
  int64_t half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  if ((params->indices_row) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    if ((params->params_pre) * (params->indices_row) * (params->params_row) * (compile_info->params_dsize) <=
        NUM_32 * BLOCK_SIZE) {
      if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
        CalNeedCoreWithBatchDims(params, compile_info);
      }
      params->params_total =
          (params->params_batch_each_core) * (params->params_pre) * (params->params_axis) * (params->params_row);
      if (!SmallRowProcess(params, compile_info, TILING_MODE_40, TILING_MODE_41, half_remain_size_ub, half_size_ub)) {
        return false;
      }
    } else {
      params->need_core_num =
          ((params->params_pre) < (compile_info->core_num)) ? (params->params_pre) : (compile_info->core_num);
      params->params_total =
          (params->params_batch) * (params->params_pre) * (params->params_axis) * (params->params_row);
      if (!SmallRowProcess(params, compile_info, TILING_MODE_38, TILING_MODE_39, half_remain_size_ub, half_size_ub)) {
        return false;
      }
    }
    return true;
  }
  if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
      params_row_ceil <= half_remain_params_elem) {
    if (!IndicesCachedProcess(params, compile_info, half_remain_size_ub, TILING_MODE_20, TILING_MODE_21,
                              TILING_MODE_22)) {
      return false;
    }
  } else {
    if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_23, TILING_MODE_24, TILING_MODE_25)) {
      return false;
    }
  }
  return true;
}

bool WithBatchDimsSmallCeil(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_size_ub,
                            int64_t half_remain_size_ub, int64_t half_remain_params_elem, int64_t params_total_ceil,
                            int64_t params_row_ceil) {
  if ((params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE != 0) {
    if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_26, TILING_MODE_27, TILING_MODE_28)) {
      return false;
    }
  } else {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
        params_row_ceil <= half_remain_params_elem) {
      if (!IndicesCachedProcess(params, compile_info, half_remain_size_ub, TILING_MODE_29, TILING_MODE_30,
                                TILING_MODE_31)) {
        return false;
      }
    } else {
      if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_32, TILING_MODE_33, TILING_MODE_34)) {
        return false;
      }
    }
  }
  return true;
}

bool WithBatchDimsBig(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t params_row_ceil,
                      int64_t half_size_ub, int64_t half_ub_params_elem, int64_t params_total_ceil,
                      int64_t half_remain_params_elem) {
  int64_t available_ub_size = compile_info->ub_size - RESERVED_UB_SIZE_2K;  // reserved 2K
  int64_t half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  if (params_row_ceil <= half_ub_params_elem) {
    if (!WithBatchDimsSmallCeil(params, compile_info, half_size_ub, half_remain_size_ub, half_remain_params_elem,
                                params_total_ceil, params_row_ceil)) {
      return false;
    }
  } else {
    if (!LargeRowProcess(params, compile_info, half_ub_params_elem, half_size_ub)) {
      return false;
    }
  }
  return true;
}

bool TilingWithBatchDims(gert::TilingContext* context, GatherV2TilingParams* params,
                         const GatherV2CompileInfo* compile_info, int64_t axis, int64_t batch_dims) {
  int64_t available_ub_size = compile_info->ub_size - RESERVED_UB_SIZE_2K;  // reserved 2K
  int64_t half_size_ub = available_ub_size / 2;
  int64_t block_num = BLOCK_SIZE / compile_info->params_dsize;
  int64_t indices_batch = 1;
  int64_t half_remain_size_ub = 1;
  ParasPreProcess(params, context, axis, batch_dims, indices_batch);

  half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_remain_params_elem = half_remain_size_ub / (compile_info->params_dsize);
  int64_t half_ub_params_elem = half_size_ub / compile_info->params_dsize;
  params->need_core_num = (indices_batch < compile_info->core_num) ? indices_batch : compile_info->core_num;
  params->tail_process_core = 0;
  params->params_batch_each_core = (params->params_batch) / (params->need_core_num);
  params->params_batch_remaining = (params->params_batch) % (params->need_core_num);
  params->indices_num_each_core = (params->params_batch_each_core) * (params->indices_row);
  params->indices_num_remaining = (params->params_batch_remaining) * (params->indices_row);

  if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    params->need_core_num = 1;
    params->tail_process_core = 0;
    params->params_batch_each_core = params->params_batch;
    params->params_batch_remaining = 0;
    params->indices_num_each_core = (params->params_batch_each_core) * (params->indices_row);
    params->indices_num_remaining = (params->params_batch_remaining) * (params->indices_row);
  }
  params->params_total =
      (params->params_batch_each_core) * (params->params_pre) * (params->params_axis) * (params->params_row);
  int64_t params_total_ceil = ((params->params_total) + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = ((params->params_row) + block_num - 1) / block_num * block_num;

  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (!WithBatchDimsSmall(params, compile_info, half_remain_params_elem, half_size_ub, params_total_ceil,
                            params_row_ceil)) {
      OP_LOGE("GatherV2", "WithBatchDimsSmall is false");
      return false;
    }
  } else {
    if (!WithBatchDimsBig(params, compile_info, params_row_ceil, half_size_ub, half_ub_params_elem, params_total_ceil,
                          half_remain_params_elem)) {
      OP_LOGE("GatherV2", "WithBatchDimsBig is false");
      return false;
    }
  }
  return true;
}

static ge::graphStatus GatherTiling(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "GatherTiling running begin");
  auto compile_info = reinterpret_cast<const GatherV2CompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  Gatherv2TilingBase tiling(context);
  return tiling.DoTiling();
  OP_LOGD(context->GetNodeName(), "GatherTiling running end");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForGatherV2(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareForGatherV2 running.");
  auto compile_info = context->GetCompiledInfo<GatherV2CompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "GatherV2Simt no need to parse compile info.");
  fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null."), return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compile_info->ub_size = static_cast<int64_t>(ubSize);
  OP_LOGD(context->GetNodeName(), "TilingPrepareForGatherV2 GRAPH_SUCCESS.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GatherV2 and Gather op.
IMPL_OP_OPTILING(GatherV2).Tiling(GatherTiling).TilingParse<GatherV2CompileInfo>(TilingPrepareForGatherV2);
IMPL_OP_OPTILING(Gather).Tiling(GatherTiling).TilingParse<GatherV2CompileInfo>(TilingPrepareForGatherV2);
}  // namespace optiling
