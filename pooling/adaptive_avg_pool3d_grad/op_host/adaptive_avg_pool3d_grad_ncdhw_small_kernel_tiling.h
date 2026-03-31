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
 * \file adaptive_avg_pool3d_grad_ncdhw_small_kernel_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (27) USING BLANK LINES.
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_Small_KERNEL_TILING_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_Small_KERNEL_TILING_H_
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_avg_pool3d_grad_tiling_arch35.h"
#include "../op_kernel/arch35/adaptive_avg_pool3d_grad_struct.h"

namespace optiling {

constexpr int64_t FLOAT16_SIZE = 2;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t UB_RESVERVED_SIZE = 2048;
constexpr int64_t UB_TEMP_BUFF_SIZE = 256 * 10;
constexpr int64_t T3_INT64 = 10;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t THRESHOLD= 2;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t MAX_INT32 = 2147483647;

struct AdaptiveAvgPool3dGradNCDHWBaseInfo {
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

struct AdaptiveAvgPool3dGradNCDHWSplitInfo {
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
    int64_t inputQueBufferSize{0};
    int64_t transOutQueBufferSize{0};
    int64_t transQueBufferSize{0};
    int64_t totalBufferSize{0};
    int64_t computeSrcBufferSize{0};
    int64_t computeAccumBufferSize{0};

};

class AdaptiveAvgPool3dGradTilingSmallKernel : public AdaptiveAvgPool3dGradTilingBaseV35 {
public:
    explicit AdaptiveAvgPool3dGradTilingSmallKernel(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradTilingBaseV35(context)
    {}

    ~AdaptiveAvgPool3dGradTilingSmallKernel() override
    {}
    
    AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35* tilingData =
        context_->GetTilingData<AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35>();

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;

    bool IsCapable() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    void InitializationVars();
    void DoBufferCalculate();
    bool IsMeetTargetCoreNum();
    bool IsMeetUBSize();
    bool TrySplitNC();
    void SplitAlignDHW();
    void SearchBestTiling();
    void DoUBTiling();
    void DynamicAdjustmentAlignDWH();
    void DoBlockTiling();
    void SetTilingData();
    void PrintSplitData() const;
    void SplitUnalignDHW();
    void DynamicAdjustmentDWH();
    
public:
    int64_t gradInputN;
    int64_t gradInputC;
    int64_t gradInputD;
    int64_t gradInputH;
    int64_t gradInputW;

    int64_t gradOutputN;
    int64_t gradOutputC;
    int64_t gradOutputD;
    int64_t gradOutputH;
    int64_t gradOutputW;

    int64_t kernelD;
    int64_t kernelH;
    int64_t kernelW;

    AdaptiveAvgPool3dGradNCDHWBaseInfo baseData;
    AdaptiveAvgPool3dGradNCDHWSplitInfo splitData;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
