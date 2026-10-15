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
 * \file conv3d_backprop_input_v2_small_shape_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_SMALL_SHAPE_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_SMALL_SHAPE_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "op_host/tiling_templates_registry.h"
#include "conv3d_backprop_input_v2_fullLoad_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DDXV2SmallShapeTiling : public Conv3DDXV2FullLoadTiling {
public:
    explicit Conv3DDXV2SmallShapeTiling(gert::TilingContext* context) : Conv3DDXV2FullLoadTiling(context) { Reset(); }
    ~Conv3DDXV2SmallShapeTiling() override = default;

protected:
    bool IsCapable() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    void InitBaseMNK(L0TilingParams& l0Params) override;
    void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params) override;
    void EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;

private:
    void AdjustSingleCoreAndL0Info(CoreTilingParams& coreParams, L0TilingParams& l0Params);
    ge::graphStatus CalcAndValidateTiling(L0TilingParams& l0Params, L1TilingParams& l1Params,
                                          CoreTilingParams& coreParams);
    bool IsEnableFullLoad(const L1TilingParams& l1Params);

    static constexpr uint64_t FULLLOAD_MIN_MCNT = 2U;
    static constexpr uint64_t FULLLOAD_MIN_KVALUE = 64U;
    void EvalBaseMNCandidate(uint64_t i, uint64_t j, uint64_t hwI, uint64_t batchDepth, uint64_t kernelHW,
                             uint64_t& maxTotalCnt, uint64_t& minTotalCnt, uint64_t& minL1LoadSize,
                             L0TilingParams& l0Params, uint64_t& singleCoreM);
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_SMALL_SHAPE_TILING_H
