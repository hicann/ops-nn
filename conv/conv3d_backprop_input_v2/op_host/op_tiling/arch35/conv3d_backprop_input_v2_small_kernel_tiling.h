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
 * \file conv3d_backprop_input_v2_small_kernel_tiling.h
 * \brief small kernel tiling template: N-axis full load, M-axis split, maximize core utilization
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_SMALL_KERNEL_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_SMALL_KERNEL_TILING_H

#include <tiling/tiling_api.h>
#include <register/tilingdata_base.h>
#include "op_host/tiling_base.h"
#include "conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DDXV2SmallKernelTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit Conv3DDXV2SmallKernelTiling(gert::TilingContext* context) : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
    }
    ~Conv3DDXV2SmallKernelTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoLibApiTiling() override;
    void InitBaseMNK(L0TilingParams& l0Params) override;
    void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void SetTilingCondition(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                            const L0TilingParams& l0Params) override;

private:
    void SetSmallKernelCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params);
    uint64_t CalcSmallKernelMaxMByL0C(uint64_t cinAlign, uint32_t cl0Pbuffer) const;
    uint64_t CalcSmallKernelCandidateM(uint64_t hwI, uint64_t mCnt, uint64_t maxMByBuffer, uint64_t m0) const;
    uint64_t CalcSmallKernelCoreScore(uint64_t hwI, uint64_t batchDepth, uint64_t coreNum, uint64_t singleCoreM) const;
    uint64_t SelectSmallKernelCoreM(uint64_t hwI, uint64_t batchDepth, uint64_t coreNum, uint64_t m0,
                                    uint64_t maxMByBuffer) const;
    uint64_t SelectSmallKernelCoreMWithBuffering(uint64_t hwI, uint64_t batchDepth, uint64_t coreNum, uint64_t m0,
                                                 uint64_t maxM, uint64_t maxSingleCoreMByL0C);
    bool HasSupportedSmallKernelDimensions() const;
    bool HasSupportedSmallKernelFormats() const;
    bool HasSupportedSmallKernelPadding() const;
    bool HasSmallKernelComputationBudget() const;
    bool HasSmallKernelBufferBudget() const;
    // for small kernel optimize
    uint64_t CalSmallKernelLocalHo(uint64_t maxM, uint64_t wi, uint64_t hk, uint64_t dilationH, uint64_t hoExpand);
    uint64_t CalcSmallKernelA1Size(uint64_t baseM) const;
    uint64_t CalcSmallKernelL1FixedSize() const;
    uint64_t CalcMaxSingleCoreMByL1(uint64_t maxM, uint32_t a1Pbuffer) const;
    bool CheckSmallKernelEnable();

    bool enableA1Db_ = false;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_SMALL_KERNEL_TILING_H
