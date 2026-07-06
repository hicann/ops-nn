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
 * \file adaptive_avg_pool2d_grad_nchw_small_kernel_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (27) USING BLANK LINES.
 */

#ifndef ADAPTIVE_AVG_POOL2D_GRAD_SMALL_KERNEL_TILING_H_
#define ADAPTIVE_AVG_POOL2D_GRAD_SMALL_KERNEL_TILING_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_avg_pool2d_grad_base_tiling.h"
#include "../../op_kernel/arch35/adaptive_avg_pool2d_grad_struct.h"

namespace optiling {

constexpr int64_t FLOAT16_SIZE = 2;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t UB_RESVERVED_SIZE = 2048;
constexpr int64_t UB_TEMP_BUFF_SIZE = 256 * 10;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t THRESHOLD = 2;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t MAX_INT32 = 2147483647;

struct AdaptiveAvgPool2dGradNCHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct AdaptiveAvgPool2dGradNCHWSplitInfo {
    int64_t isCheckRange{0};

    int64_t highAxisInner{0};
    int64_t highAxisTail{0};
    int64_t highAxisOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    int64_t inputQueBufferSize{0};
    int64_t transQueBufferSize{0};
    int64_t transOutQueBufferSize{0};
    int64_t totalBufferSize{0};
};

class AdaptiveAvgPool2dGradTilingSmallKernel : public AdaptiveAvgPool2dGradTilingBase {
public:
    explicit AdaptiveAvgPool2dGradTilingSmallKernel(gert::TilingContext* context)
        : AdaptiveAvgPool2dGradTilingBase(context)
    {}

    ~AdaptiveAvgPool2dGradTilingSmallKernel() override {}

    AdaptiveAvgPool2dGradOp::AdaptiveAvgPool2dNCHWGradSmallKernelTilingDataV35* tilingData = context_->GetTilingData<
        AdaptiveAvgPool2dGradOp::AdaptiveAvgPool2dNCHWGradSmallKernelTilingDataV35>();

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

    void SplitAlignHW();
    void DynamicAdjustmentAlignHW();

    void SplitUnalignHW();
    void DynamicAdjustmentHW();

    void SearchBestTiling();

    bool ExhaustiveSearchBestTiling(int64_t computeVl, int64_t ncSearchMax, int64_t& bestHighAxisInner,
                                    int64_t& bestHOutputInner, int64_t& bestWOutputInner, int64_t& bestBlockNum,
                                    int64_t& bestUsedCoreNum, int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail,
                                    int64_t& bestBufferSize, long double& bestCost, bool& found);

    long double EvalTilingCandidate(int64_t highAxisInner, int64_t highAxisOuter, int64_t highAxisTail,
                                    int64_t highAxisPadding, int64_t hOutputInner, int64_t hOutputOuter,
                                    int64_t wOutputInner, int64_t wOutputOuter, int64_t blockNum, int64_t computeVl,
                                    int64_t normalCoreProcessNum);

    long double AddCostPenalties(long double cost, int64_t highAxisInner, int64_t highAxisOuter, int64_t highAxisTail,
                                 int64_t hOutputInner, int64_t wOutputInner, int64_t blockNum, int64_t computeVl,
                                 int64_t normalCoreProcessNum, int64_t oneBlockWork);

    bool TryRecordBetterTiling(long double cost, int64_t hOutputInner, int64_t wOutputInner, int64_t blockNum,
                               int64_t usedCoreNum, int64_t highAxisInner, int64_t highAxisPadding,
                               int64_t highAxisTail, int64_t& bestHighAxisInner, int64_t& bestHOutputInner,
                               int64_t& bestWOutputInner, int64_t& bestBlockNum, int64_t& bestUsedCoreNum,
                               int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail, int64_t& bestBufferSize,
                               long double& bestCost, bool& found);

    void ApplyCoarseFallback();

    void DoUBTiling();
    void DoBlockTiling();
    void SetTilingData();
    void PrintSplitData() const;

public:
    int64_t gradInputN{0};
    int64_t gradInputC{0};
    int64_t gradInputH{0};
    int64_t gradInputW{0};

    int64_t gradOutputN{0};
    int64_t gradOutputC{0};
    int64_t gradOutputH{0};
    int64_t gradOutputW{0};

    int64_t kernelH{0};
    int64_t kernelW{0};

    AdaptiveAvgPool2dGradNCHWBaseInfo baseData;
    AdaptiveAvgPool2dGradNCHWSplitInfo splitData;
};

} // namespace optiling

#endif // ADAPTIVE_AVG_POOL2D_GRAD_SMALL_KERNEL_TILING_H_