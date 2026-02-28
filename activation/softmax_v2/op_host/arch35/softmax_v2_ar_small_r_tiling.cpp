/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_v2_ar_small_r_tiling.cpp
 * \brief
 */

#include "softmax_v2_tiling.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

using namespace ge;

using namespace Ops::Base;
namespace optiling
{
bool SoftmaxV2TilingArSmallR::IsCapable()
{
    OP_CHECK_IF((a0_ != DIM_NUM_ONE) || (r_ > DATA_BLOCK_COUNT), OP_LOGI(context_->GetNodeName(), "AR small r template is not capable."),
                    return false);
    return true;
}

ge::graphStatus SoftmaxV2TilingArSmallR::DoOpTiling()
{
    // 转置成(r_, a1_)
    int64_t a0New = a1_;

    int64_t a0TileBase = vlFp32_; // 计算过程中均按FP32处理
    int64_t rTileBase;
    if (yDtype_ == ge::DT_FLOAT) {
        rTileBase = DATA_BLOCK_COUNT / CONST_TWO;
    } else {
        rTileBase = DATA_BLOCK_COUNT;
    }
    int64_t rAligned = CeilDiv(r_, rTileBase) * rTileBase;

    int64_t rFactor = rAligned * (xDtypeSize_ * DOUBLE_BUFFER + yDtypeSize_ * DOUBLE_BUFFER +
                                  FLOAT32_BYTES * DOUBLE_BUFFER);  // 输入、tmp、输出buffer的基本大小，单位为字节

    int64_t a0TileNumMax = aicoreParams_.ubSize / (a0TileBase * (rFactor + FLOAT32_BYTES)); // 最大支持a基础块的个数

    OP_CHECK_IF(a0TileNumMax < 1,
                    OP_LOGI(context_->GetNodeName(),
                            "AR small r template is not capable. merged shape is (%ld, %ld, %ld), ub size: %ldB, "
                            "tileBase: %ld, ub factor: %ld.",
                            a1_, r_, a0_, aicoreParams_.ubSize, a0TileBase, a0TileNumMax),
                    return ge::GRAPH_PARAM_INVALID);

    // 切a0
    int64_t a0TileNumTmp = CeilDiv(a0New, a0TileBase);  // a基础块的数量
    int64_t a0TileNum = a0TileNumMax < a0TileNumTmp ? a0TileNumMax : a0TileNumTmp;
    int64_t tileA0Len = a0TileNum * a0TileBase;     // UB内能放下的tiling块，UB块
    int64_t a0Outer = CeilDiv(a0New, tileA0Len);    // UB块的数量
    int64_t tileA0Tail = a0New - tileA0Len * (a0Outer - 1);

    // 按核均分
    int64_t tilesPerCore = CeilDiv(a0Outer, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = CeilDiv(a0Outer, tilesPerCore);

    tilingData_.set_totalTiles(a0Outer);
    tilingData_.set_tilesPerCore(tilesPerCore);
    tilingData_.set_totalRLen(r_);
    tilingData_.set_totalA0Len(a0New);
    tilingData_.set_tileA0Len(tileA0Len);
    tilingData_.set_tileA0Tail(tileA0Tail);
    tilingData_.set_rTileBase(rTileBase);
    tilingData_.set_rAligned(rAligned);

    OP_LOGI(context_->GetNodeName(), "Do tiling success, tileA0Len: %ld, tileA0Tail: %ld, "
                                     "totalTiles: %ld, tilesPerCore: %ld, usedCoreNums_:%ld",
                                     tileA0Len, tileA0Tail, a0Outer, tilesPerCore, usedCoreNums_);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxV2TilingArSmallR::GetTilingKey() const
{
    return TILINGKEY_AR_SMALL_R;
}

ge::graphStatus SoftmaxV2TilingArSmallR::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxV2, SoftmaxV2TilingArSmallR, TEMPLATE_AR_SMALL_R_PRIORITY);
}  // namespace optiling
