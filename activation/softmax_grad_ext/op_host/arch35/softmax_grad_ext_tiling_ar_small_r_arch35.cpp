/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_tiling_ar_small_r.cc
 * \brief
 */
#include "softmax_grad_ext_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling
{

constexpr int64_t UB_RESERVED_BYTE = 1024 * 8;

bool SoftmaxGradExtTilingARSmallR::IsCapable()
{
    OP_TILING_CHECK((a0_ != DIM_NUM_ONE) || (r_ > DATA_BLOCK_COUNT) || (r_ > CONST_EIGHT && yDtype_ == ge::DT_FLOAT), OP_LOGI(context_->GetNodeName(), "AR small r template is not capable. "),
                    return false);
    return true;
}//检查什么条件下使用AR samll r模板

ge::graphStatus SoftmaxGradExtTilingARSmallR::DoOpTiling()
{
    // 转置
    int64_t a1 = a1_;
    int64_t rTileBase;
    if (yDtype_ == ge::DT_FLOAT) {
        rTileBase = DATA_BLOCK_COUNT / CONST_TWO;
    } else {
        rTileBase = DATA_BLOCK_COUNT;
    }
    int64_t rAligned = CeilAlign(r_, rTileBase);

    // rAligned * (grad, x1, x2, yTransposed, y) * DB * sizeof(T) + reduceBuffer * DB * sizeof(float)
    int64_t rFactor = rAligned * (xDtypeSize_ * CONST_FIVE * CONST_TWO) + CONST_TWO * CONST_FOUR;
    
    // ubFactor表示按a轴切分，ub内单次循环计算最大能放多大的切片
    int64_t ubFactor= (aicoreParams_.ubSize - UB_RESERVED_BYTE)/ rFactor;
    // ubFactor 需要按32b对齐，但不能超过ub大小
    ubFactor = CeilAlign(ubFactor, rTileBase);
    if (aicoreParams_.ubSize < static_cast<decltype(aicoreParams_.ubSize)>(rFactor * ubFactor)) {
        ubFactor -= rTileBase;
    }
    OP_TILING_CHECK(
        (ubFactor <= 0),
        OP_LOGI(context_->GetNodeName(), "AR small r template is not capable. r is %ld, rFactor %ld", r_, rFactor),
        return ge::GRAPH_PARAM_INVALID);

    // 按核均分a轴
    int64_t aPerHeadCore = CeilDiv(a1, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums_ = CeilDiv(a1, aPerHeadCore);
    int64_t aPerTailCore = a1 - (usedCoreNums_ - 1) * aPerHeadCore;

    tilingData_.set_a(a1);
    tilingData_.set_r(r_);
    tilingData_.set_ubFactor(ubFactor);
    tilingData_.set_aPerHeadCore(aPerHeadCore);
    tilingData_.set_aPerTailCore(aPerTailCore);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_x2IsScalar(isX2Scalar_ ? 1L : 0L);

    OP_LOGI(context_->GetNodeName(),
            "Do tiling success, a: %ld, r: %ld, ubFactor: %ld, "
            "aPerHeadCore: %ld, aPerTailCore: %ld, usedCoreNums_:%ld",
            a1, r_, ubFactor, aPerHeadCore, aPerTailCore, usedCoreNums_);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxGradExtTilingARSmallR::GetTilingKey() const
{
    return TILINGKEY_AR_SMALL_R;
}

ge::graphStatus SoftmaxGradExtTilingARSmallR::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxGradExt, SoftmaxGradExtTilingARSmallR, TEMPLATE_AR_SMALL_R_PRIORITY);

}  // namespace optiling
