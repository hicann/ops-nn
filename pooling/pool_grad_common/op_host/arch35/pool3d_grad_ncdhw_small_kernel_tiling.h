
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
 * \file max_pool3d_grad_with_argmax_simd_tiling.cpp
 * \brief
 */
#ifndef POOL3D_GRAD_NCDHW_TILING_H
#define POOL3D_GRAD_NCDHW_TILING_H

#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "../..//op_kernel/arch35/pool3d_grad_struct_common.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "util.h"

namespace optiling {

struct Pool3DGradNCDHWInputInfo {
    int64_t dPad{0};
    int64_t hPad{0};
    int64_t wPad{0};
    int64_t dStride{1};
    int64_t hStride{1};
    int64_t wStride{1};
    int64_t dKernel{1};
    int64_t hKernel{1};
    int64_t wKernel{1};
    int64_t dDilation{1};
    int64_t hDilation{1};
    int64_t wDilation{1};
    int64_t nX{1};
    int64_t cX{1};
    int64_t dX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t nGrad{1};
    int64_t cGrad{1};
    int64_t dGrad{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    bool ceilMode{false};
    int64_t gradShapeSize{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::Format inputFormat{ge::Format::FORMAT_NCDHW};
    int64_t isInt32Meet{1};
};

struct Pool3DGradNCDHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t dProBatchSize{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct Pool3DGradNCDHWSplitInfo {
    // DoUBTiling
    int64_t isCheckRange{0};

    int64_t highAxisInner{0};
    int64_t highAxisTail{0};
    int64_t highAxisOuter{0};

    int64_t dOutputInner{0};
    int64_t dOutputTail{0};
    int64_t dOutputOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t inputBufferSize{0};
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
};

struct Tiling4Pool3DGradCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

class Pool3DGradNCDHWSmallKernelCommonTiling {
public:
    Pool3DGradNCDHWSmallKernelCommonTiling(Pool3DGradNCDHWInputInfo* input) : inputData(input) 
    {}

    void InitializationVars(gert::TilingContext* context_, int64_t ubSize_, int64_t coreNum_);
    ge::graphStatus DoOpTiling(gert::TilingContext* context);
    ge::graphStatus PostTiling(gert::TilingContext* context_, uint64_t key);
    Pool3DGradNCDHWSplitInfo& GetSplitData();
    Pool3DGradNCDHWBaseInfo& GetBaseData();
    void DoBufferCalculate();

private:
    void DoUBTiling();
    bool TrySplitNC();
    bool TrySplitAlignD();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignDHW();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentDWH();
    void SetTilingData(gert::TilingContext* context);
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();

    Pool3DGradNCDHWBaseInfo baseData;
    Pool3DGradNCDHWSplitInfo splitData;
    Pool3DGradNCDHWInputInfo* inputData;
};

} // namespace optiling

#endif
