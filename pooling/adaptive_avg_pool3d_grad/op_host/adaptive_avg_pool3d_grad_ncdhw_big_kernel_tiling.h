/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (27) USING BLANK LINES.
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_BIG_KERNEL_TILING_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_BIG_KERNEL_TILING_H_
#include "adaptive_avg_pool3d_grad_ncdhw_tiling_common.h"

namespace optiling {

constexpr int64_t ADAPTIVE_BIG_KERNEL_SIZE = 256;

struct AdaptiveAvgPool3dGradNCDHWBigKernelSplitInfo : public AdaptiveAvgPool3dGradNCDHWSplitCommon {
    // DoBufferCalculate (big kernel 专属 buffer)
    int64_t gradInputBufferSize{0};
    int64_t outputBufferSize{0};
};

class AdaptiveAvgPool3dGradTilingBigKernel
    : public AdaptiveAvgPool3dGradNCDHWTilingCommon<AdaptiveAvgPool3dGradNCDHWBigKernelSplitInfo> {
public:
    explicit AdaptiveAvgPool3dGradTilingBigKernel(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradNCDHWTilingCommon<AdaptiveAvgPool3dGradNCDHWBigKernelSplitInfo>(context)
    {}

    ~AdaptiveAvgPool3dGradTilingBigKernel() override {}

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;

    bool IsCapable() override;
    ge::graphStatus PostTiling() override;

    void InitializationVars();
    void DoBufferCalculate() override;
    bool TrySplitNC();
    void SplitUnalignDHW();
    void SearchBestTiling() override;
    void DynamicAdjustmentDWH();
    void SplitAlignDHW();
    void DynamicAdjustmentAlignDWH();
    ge::graphStatus SetTilingData();
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
