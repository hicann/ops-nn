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
 * \file conv3d_backprop_filter_v2_dload_tiling.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_H
#define CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_H

#include "conv3d_backprop_filter_v2_basic_block_tiling_arch35.h"

namespace Ops {
namespace NN {
namespace Conv {
class Conv3DBackpropFilterV2DLoadTiling : public Conv3DDWV2BasicBlockTilingArch35 {
public:
    // 固定档 tiling：baseK=16（kl0HoWo howo 窗宽，16 倍数）、
    // baseM=128（cout 块宽）、baseN=144（=16 cin × 3×3 hkwk，mmad N 轴）
    static constexpr uint32_t DLOAD_BASE_K = 16;
    static constexpr uint32_t DLOAD_BASE_M = 128;
    static constexpr uint32_t DLOAD_BASE_N_PER_HKWK = 16; // baseN/hkwk = cin 块宽 16
    static constexpr uint32_t DLOAD_KERNEL_SIZE_3 = 3;

    explicit Conv3DBackpropFilterV2DLoadTiling(gert::TilingContext* context) : Conv3DDWV2BasicBlockTilingArch35(context)
    {
        Reset();
    }

    ~Conv3DBackpropFilterV2DLoadTiling() override = default;

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

private:
    bool CheckFormat();
    bool CheckDLoadDtype();
    bool CheckDLoadAttrs();
    bool CheckShape();
};
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_H
