
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL3D_GRAD_NDHWC_SMALL_KERNEL_TILING_H
#define POOL3D_GRAD_NDHWC_SMALL_KERNEL_TILING_H

#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/pool3d_grad_struct_common.h"
#include "pool3d_grad_ncdhw_small_kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "util.h"

namespace optiling {

struct Pool3DGradNDHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t moveDataNumCacheLine{0};
    int64_t dProBatchSize{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct Pool3DGradNDHWCSplitInfo {
    int64_t isCheckRange{0};

    int64_t nOutputInner{1};
    int64_t nOutputTail{1};
    int64_t nOutputOuter{1};

    int64_t dOutputInner{1};
    int64_t dOutputTail{1};
    int64_t dOutputOuter{1};

    int64_t hOutputInner{1};
    int64_t hOutputTail{1};
    int64_t hOutputOuter{1};

    int64_t wOutputInner{1};
    int64_t wOutputTail{1};
    int64_t wOutputOuter{1};

    int64_t cOutputInner{1};
    int64_t cOutputTail{1};
    int64_t cOutputOuter{1};

    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    int64_t inputBufferSize{0};
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
    int64_t isBigKernel{0};
};

constexpr int64_t NDHWC_BIG_HELP_BUF_SIZE = 5120;

class Pool3DGradNDHWCSmallKernelCommonTiling {
public:
    Pool3DGradNDHWCSmallKernelCommonTiling(Pool3DGradNCDHWInputInfo* input) : inputData(input) {}
    virtual ~Pool3DGradNDHWCSmallKernelCommonTiling() = default;

    void InitializationVars(gert::TilingContext* context, int64_t ubSize, int64_t coreNum);
    ge::graphStatus DoOpTiling(gert::TilingContext* context);
    ge::graphStatus PostTiling(gert::TilingContext* context, uint64_t key);
    Pool3DGradNDHWCSplitInfo& GetSplitData();
    Pool3DGradNDHWCBaseInfo& GetBaseData();

    virtual void SetTilingData(gert::TilingContext* context) = 0;

protected:
    void SearchBestTiling();
    void DoUBTiling();
    void DoBlockTiling();
    bool TrySplitN();
    bool TrySplitAlignD();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    bool TrySplitAlignC();
    void SplitUnalignDHWC();
    void DynamicAdjustmentDHW();
    virtual bool IsMeetUBSize() = 0;
    bool IsMeetTargetCoreNum() const;

    Pool3DGradNDHWCBaseInfo baseData;
    Pool3DGradNDHWCSplitInfo splitData;
    Pool3DGradNCDHWInputInfo* inputData;
};

} // namespace optiling

#endif
