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
 * \file reverse_sequence_tiling_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_REVERSE_SEQUENCE_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_REVERSE_SEQUENCE_TILING_COMMON_H_

#include <array>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling
{
static constexpr int32_t allDims = 5;

struct ReverseInputInfo {
    int64_t batchAxis = 1;
    int64_t seqAxis = 0;
    int64_t batchDim = 0;
    int64_t seqDim = 0;
    int64_t inputDim[allDims] = {0, 0, 0, 0, 0};
    int64_t comBineDims = 0;
    int64_t xDtypeSize = 0;
    int64_t xShapeSize = 0;
    int64_t batchSize = 1;
    int64_t reverseSize = 1;
    int64_t comBineType = 0;
};

ge::graphStatus GetReverseSequencePlatformInfo(gert::TilingContext *context, uint64_t& ubSize, uint64_t& coreNum);
ge::graphStatus GetReverseSequenceShapeAttrsInfo(gert::TilingContext *context, ReverseInputInfo& inputData);

}  // namespace optiling
 
#endif