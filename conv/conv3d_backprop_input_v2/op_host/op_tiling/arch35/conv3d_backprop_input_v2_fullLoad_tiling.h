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
 * \file conv3d_backprop_input_v2_fullLoad_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_FULLLOAD_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_FULLLOAD_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "conv3d_backprop_input_v2_base_tiling.h"
#include "conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DDXV2FullLoadTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit Conv3DDXV2FullLoadTiling(gert::TilingContext* context) : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
    }
    ~Conv3DDXV2FullLoadTiling() override = default;

protected:
    bool IsCapable() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;

    void InitBaseMNK(L0TilingParams& l0Params) override;
    void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params) override;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_FULLLOAD_TILING_H
