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
 * \file avg_pool_v2_grad_nchw_tiling.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_NCHW_TILING_H_
#define AVG_POOL_V2_GRAD_NCHW_TILING_H_

#include <array>
#include <cstdint>

#include "log/log.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "avg_pool_v2_grad_tiling_base.h"
#include "../../op_kernel/arch35/avg_pool_v2_grad_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool_v2_grad_tiling_key.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct AvgPoolV2GradNCHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t dataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct AvgPoolV2GradNCHWSplitInfo {
    // DoUBTiling
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

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t totalBufferSize{0};
};

class AvgPoolV2GradCommonNCHWTiling : public TilingBaseClass {
public:
    explicit AvgPoolV2GradCommonNCHWTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~AvgPoolV2GradCommonNCHWTiling() override
    {}

protected:
    void DoUBTiling();
    void InitializationVars();
    bool TrySplitNC();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignHW();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentWH();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

public:
    AvgPoolV2GradInputInfo inputData;
    AvgPoolV2GradNCHWBaseInfo baseData;
    AvgPoolV2GradNCHWSplitInfo splitData;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
};

class AvgPoolV2GradNCHWTiling : public AvgPoolV2GradCommonNCHWTiling {
public:
    explicit AvgPoolV2GradNCHWTiling(gert::TilingContext* context) : AvgPoolV2GradCommonNCHWTiling(context)
    {}
    ~AvgPoolV2GradNCHWTiling() override
    {}

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

} // namespace optiling

#endif
