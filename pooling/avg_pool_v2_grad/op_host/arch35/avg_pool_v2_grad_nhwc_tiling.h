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
 * \file avg_pool_v2_grad_nhwc_tiling.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_NHWC_TILING_H_
#define AVG_POOL_V2_GRAD_NHWC_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "avg_pool_v2_grad_tiling_common.h"
#include "avg_pool_v2_grad_tiling_base.h"
#include "../op_kernel/arch35/avg_pool_v2_grad_tiling_data.h"
#include "../op_kernel/arch35/avg_pool_v2_grad_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;

struct AvgPoolV2GradNHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t gradBytes{0};
    int64_t availableUb{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t moveDataNumCacheLine{0};
};

struct AvgPoolV2GradNHWCSplitInfo {
    // DoUBTiling
    int64_t isCheckRange{0};

    int64_t nOutputInner{0};
    int64_t nOutputTail{0};
    int64_t nOutputOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    int64_t cOutputInner{0};
    int64_t cOutputTail{0};
    int64_t cOutputOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t outputBufferSize{0};
    int64_t inputGradBufferSize{0};
    int64_t shapeBufferSize{0};
    int64_t totalBufferSize{0};
};

class AvgPoolV2GradCommonNHWCTiling : public TilingBaseClass {
public:
    explicit AvgPoolV2GradCommonNHWCTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~AvgPoolV2GradCommonNHWCTiling() override
    {
    }

protected:
    void DoUBTiling();  // 入口
    void InitializationVars();  // tiling初始化
    bool TrySplitN();  // 尝试一次ub分多个n
    bool TrySplitAlignH();  // 无pad无overlap，尝试一次ub分多个h
    bool TrySplitAlignW();  // 无pad无overlap，ub分多个w
    bool TrySplitAlignC();   // 无pad无overlap，ub分多个c
    void SplitUnalignHWC();  // 非对齐加载
    bool IsMeetTargetCoreNum() const;  // 当前切分是否满足核数 分块过大，块数小于核数会false
    bool IsMeetUBSize();  // 当前切分是否满足ub大小 buffer小于ub大小
    void SearchBestTiling();  // 动态查找各维度outer TrySplitN TrySplitAlignH TrySplitAlignW SplitUnalignHWC
    void DynamicAdjustmentWH();  // SplitUnalignHWC中动态查找h和w切分outer
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();  // 核间block划分
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

public:
    AvgPoolV2GradInputInfo inputData;
    AvgPoolV2GradNHWCBaseInfo baseData;
    AvgPoolV2GradNHWCSplitInfo splitData;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
};

class AvgPoolV2GradNHWCTiling : public AvgPoolV2GradCommonNHWCTiling {
public:
    explicit AvgPoolV2GradNHWCTiling(gert::TilingContext* context) : AvgPoolV2GradCommonNHWCTiling(context)
    {}
    ~AvgPoolV2GradNHWCTiling() override
    {}

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif