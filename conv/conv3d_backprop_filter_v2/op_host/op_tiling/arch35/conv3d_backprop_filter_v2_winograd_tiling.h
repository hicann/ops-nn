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
 * \file conv3d_backprop_filter_v2_winograd_tiling.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_H
#define CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_H

#include "conv3d_backprop_filter_v2_basic_block_tiling_arch35.h"

namespace Ops {
namespace NN {
namespace Conv {
class Conv3DBackpropFilterV2WinogradTiling : public Conv3DDWV2BasicBlockTilingArch35 {
public:
    // 不去支持超大pad这类场景，kernel当前本身实现无限制，但是pad过大性能可能不比实现了pad跳过的基本块kernel好
    static constexpr uint32_t RECOMMEND_PAD_LIMIT = 8;

    // Winograd当前实现对pad区域也会做正变换和mmad，无法像基本块kernel那样跳过全pad矩阵
    // 当pad产生的无效tile占比过高时，winograd的计算密度优势会被无效开销抵消甚至劣化
    // 经推算，当有效fmap区域产生的tile数占总tile数比例低于该阈值时，退回基本块kernel更优
    // 阈值取0.5的含义：pad浪费不超过一半，结合winograd的cube吞吐优势仍有净收益
    static constexpr float RECOMMEND_FMAP_VALID_TILE_RATIO = 0.5f;

    // 累加轴过大暂时不处理，winograd累加轴比常规实现少了4倍，应该能囊括绝大部分case
    // 有需要可以适当放大
    static constexpr uint32_t RECOMMEND_K_MAX_SIZE = 512000;

    explicit Conv3DBackpropFilterV2WinogradTiling(gert::TilingContext* context)
        : Conv3DDWV2BasicBlockTilingArch35(context)
    {
        Reset();
    }

    ~Conv3DBackpropFilterV2WinogradTiling() override = default;

    enum class SingleShapeTile {
        B16H8W8_B32H4W8,
        B16H4W16_B32H2W16,
    };

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

private:
    bool CheckFormat();

    SingleShapeTile singleShapeTile_;
};
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_H
