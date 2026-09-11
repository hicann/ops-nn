#pragma once
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "index/common/op_api/median_dim.h"
#include "level0/add.h"
#include "level0/masked_fill.h"
#include "level0/reduce_min.h"
#include "index/common/op_api/gather_elements.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/select.h"
#include "level0/div.h"
#include "level0/right_shift.h"
#include "level0/zero_op.h"
#include "level0/equal.h"
#include "level0/not_equal.h"
#include "level0/fill.h"
#include "op_api/gather_v2.h"
#include "level0/reduce_sum_op.h"
#include "aclnn_kernels/reshape.h"
#include "level0/sort.h"
#include "level0/sub.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "util/math_util.h"
using namespace op;

namespace Ops::NN::MedianCommon {
static const int32_t MAX_INT32 = 2147483647;
static const int32_t MIN_INT32 = -2147483648;
static const int64_t MIN_INT64 = -9223372036854775807LL - 1;
static const int32_t MAX_CONVERT_NUM = 16777216;
static constexpr size_t DIM_ZERO = 0;
static const int64_t SMALLSORTLIMIT = 16;
static constexpr int64_t MEDIAN_DIRECT_AXIS_THRESHOLD = 2048;
static constexpr int64_t MEDIAN_MIN_AXIS_LEN = 2;

using MedianFunction = const std::tuple<aclTensor*, aclTensor*> (*)(const aclTensor*, int64_t, aclOpExecutor*);

const std::initializer_list<op::DataType>& GetDtypeSupportList();

const std::initializer_list<op::DataType>& GetFloatList();

bool CheckNotNull(const aclTensor* self, const aclTensor* valuesOut);

bool CheckDtypeValid(const aclTensor* self, const aclTensor* valuesOut);

bool CheckShape(const aclTensor* self, const aclTensor* valuesOut);

aclnnStatus CheckParams(const aclTensor* self, const aclTensor* valuesOut);

bool CheckTupleNotNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple);

bool CheckDtypeValidDim(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut);

bool CheckNotNullDim(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut);

int64_t GetTensorDim(const aclTensor* self);

bool CheckDimValue(const aclTensor* self, const int64_t dim);

bool CheckShapeDim(const aclTensor* self, bool keepDim, const aclTensor* valuesOut, const aclTensor* indicesOut);

aclnnStatus CheckParamsDim(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                           aclTensor* indicesOut);

aclIntArray* GetPermResult(int64_t dim, int64_t dimSize, aclOpExecutor* executor);

const aclTensor* MedianAdaptInputZeroDimTensor(const aclTensor* self, int64_t dimNum, aclOpExecutor* executor);

const aclTensor* ReduceOneDim(const aclTensor* self, int64_t selfShapeDim, aclOpExecutor* executor);

aclIntArray* GetReduceShape(const aclTensor* self, const int64_t dim, aclOpExecutor* executor);

std::tuple<const aclTensor*, const aclTensor*> SortProcess(const aclTensor* self, const int64_t dim,
                                                           aclOpExecutor* executor);

const aclTensor* GetLast(const aclTensor* sortValues, int64_t dim, aclOpExecutor* executor);

const aclTensor* CreateNanTensor(const aclTensor* self, aclOpExecutor* executor);

const aclTensor* GetNanMask(const aclTensor* self, int64_t dim, aclOpExecutor* executor);

const aclTensor* GetFirstNanIndices(const aclTensor* sortValues, const aclTensor* sortIndices, int64_t dim,
                                    aclOpExecutor* executor);

std::tuple<const aclTensor*, const aclTensor*> GetMedianSortResult(const aclTensor* self, int64_t dim,
                                                                   aclOpExecutor* executor);

bool CanMedianNonLastAxisDirectly(const aclTensor* self, int64_t realDim);

std::tuple<const aclTensor*, const aclTensor*> GetMedianResult(const aclTensor* self, int64_t realDim,
                                                               aclOpExecutor* executor, MedianFunction medianFunction);

aclnnStatus DealMedianEmptyTensor(const aclTensor* self, aclTensor* out, aclOpExecutor* executor);

void CheckFormat(const aclTensor* self);

} // namespace Ops::NN::MedianCommon
