/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_FULLLOAD_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_FULLLOAD_TILING_H

#include <tiling/tiling_api.h>
#include <register/tilingdata_base.h>
#include "tiling_base/tiling_base.h"
#include "conv3d_backprop_input_v2_base_tiling.h"
#include "conv3d_backprop_input_v2_kernel_split_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DDXV2KernelSplitFullLoadTiling : public Conv3DDXV2KernelSplitTiling {
public:
    explicit Conv3DDXV2KernelSplitFullLoadTiling(gert::TilingContext* context) : Conv3DDXV2KernelSplitTiling(context)
    {
        Reset();
    }
    ~Conv3DDXV2KernelSplitFullLoadTiling() override = default;

protected:
    bool IsCapable() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;

    void InitBaseMNK(L0TilingParams& tilingParams) override;
    void AdjustBaseMNK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo) override;
    void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_FULLLOAD_TILING_H
