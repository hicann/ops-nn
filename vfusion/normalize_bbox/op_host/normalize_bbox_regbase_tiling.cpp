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
 * \file normalize_bbox_regbase_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_key.h"
#include "normalize_bbox_tiling.h"
#include "normalize_bbox_regbase_tiling.h"

namespace optiling {
static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t VREPEAT_SIZE = 256;
static constexpr uint64_t UB_RESERVE = 32 * 1024; // Div reserved UB + queue metadata margin
static constexpr uint64_t MAX_TILE_LEN = 8192;
static constexpr uint64_t INPUT_BOXES = 0;
static constexpr uint64_t INPUT_SHAPE_HW = 1;
static constexpr uint64_t ATTR_REVERSED_BOX = 0;
static constexpr uint64_t BOXES_RANK_MIN = 2; // dim0=batch, coord=4; 中间维可空(num=1)
static constexpr uint64_t BOXES_RANK_MAX = 8; // ND 上限
static constexpr uint64_t SHAPE_HW_RANK = 2;
static constexpr int64_t SHAPE_HW_DIM1 = 3;
static constexpr int64_t COORD_NUM = 4;

bool NormalizeBBoxTilingForRegbase::IsCapable() { return true; }

// 1. platform info
ge::graphStatus NormalizeBBoxTilingForRegbase::GetPlatformInfo()
{
    OP_LOGD(opName_, "NormalizeBBoxTilingForRegbase GetPlatformInfo.");
    auto compileInfo = static_cast<const NormalizeBBoxCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = static_cast<uint64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((totalCoreNum_ == 0 || ubSize_ == 0),
                OP_LOGE(opName_, "invalid platform info: coreNum=%lu ubSize=%lu.", totalCoreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 2. shape / attr info + validation
ge::graphStatus NormalizeBBoxTilingForRegbase::GetDtypeAndAttr()
{
    auto boxesDesc = context_->GetInputDesc(INPUT_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, boxesDesc);
    boxesDType_ = boxesDesc->GetDataType();
    OP_CHECK_IF((boxesDType_ != ge::DT_FLOAT16 && boxesDType_ != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "boxes", Ops::Base::ToString(boxesDType_),
                                                      "only fp16/fp32 supported"),
                return ge::GRAPH_FAILED);
    boxesDtypeSize_ = ge::GetSizeByDataType(boxesDType_);

    auto attrs = context_->GetAttrs();
    reversedBox_ = false;
    if (attrs != nullptr) {
        const bool* reversedPtr = attrs->GetAttrPointer<bool>(ATTR_REVERSED_BOX);
        if (reversedPtr != nullptr) {
            reversedBox_ = *reversedPtr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormalizeBBoxTilingForRegbase::ValidateShapes(const gert::Shape& boxesGeShape,
                                                              const gert::Shape& shapeHwGeShape, uint64_t& boxesRank)
{
    boxesRank = static_cast<uint64_t>(boxesGeShape.GetDimNum());
    OP_CHECK_IF((boxesRank < BOXES_RANK_MIN || boxesRank > BOXES_RANK_MAX),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "boxes", std::to_string(boxesRank) + "D",
                                                         "rank must be in [2, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((shapeHwGeShape.GetDimNum() != SHAPE_HW_RANK || shapeHwGeShape.GetDim(1) != SHAPE_HW_DIM1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName_, "shape_hw",
                    std::to_string(shapeHwGeShape.GetDimNum()) + "D, dim1=" + std::to_string(shapeHwGeShape.GetDim(1)),
                    "shape_hw must be 2-D and dim1 must be 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (boxesGeShape.GetDim(0) != shapeHwGeShape.GetDim(0)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "boxes, shape_hw",
                                               "boxes.batch=" + std::to_string(boxesGeShape.GetDim(0)) +
                                                   ", shape_hw.batch=" + std::to_string(shapeHwGeShape.GetDim(0)),
                                               "boxes.batch must equal shape_hw.batch"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (boxesGeShape.GetDim(0) < 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "boxes", "dim0=" + std::to_string(boxesGeShape.GetDim(0)),
                                              "boxes dim0 (batch) must be non-negative"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t NormalizeBBoxTilingForRegbase::ComputeNum(const gert::Shape& boxesGeShape, uint64_t boxesRank,
                                                   bool reversedBox, const std::string& opName, ge::graphStatus& status)
{
    status = ge::GRAPH_SUCCESS;
    uint64_t num = 1;
    auto fail = [&](const std::string& dim, const std::string& reason) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "boxes", dim, reason);
        status = ge::GRAPH_FAILED;
    };
    if (reversedBox) {
        if (boxesGeShape.GetDim(1) != COORD_NUM) {
            fail("dim1=" + std::to_string(boxesGeShape.GetDim(1)), "boxes dim1 must be 4 when reversedBox=true");
            return 0;
        }
        for (uint64_t i = 2; i < boxesRank; i++) {
            int64_t dimVal = boxesGeShape.GetDim(i);
            if (dimVal < 0) {
                fail("dim[" + std::to_string(i) + "]=" + std::to_string(dimVal), "boxes dims must be non-negative");
                return 0;
            }
            num *= static_cast<uint64_t>(dimVal);
        }
    } else {
        if (boxesGeShape.GetDim(boxesRank - 1) != COORD_NUM) {
            fail("last_dim=" + std::to_string(boxesGeShape.GetDim(boxesRank - 1)), "boxes last dimension must be 4");
            return 0;
        }
        for (uint64_t i = 1; i + 1 < boxesRank; i++) {
            int64_t dimVal = boxesGeShape.GetDim(i);
            if (dimVal < 0) {
                fail("dim[" + std::to_string(i) + "]=" + std::to_string(dimVal), "boxes dims must be non-negative");
                return 0;
            }
            num *= static_cast<uint64_t>(dimVal);
        }
    }
    return num;
}

ge::graphStatus NormalizeBBoxTilingForRegbase::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "NormalizeBBoxTilingForRegbase GetShapeAttrsInfo.");
    const gert::StorageShape* boxesShape = context_->GetInputShape(INPUT_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, boxesShape);
    const gert::StorageShape* shapeHwShape = context_->GetInputShape(INPUT_SHAPE_HW);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHwShape);

    auto ret = GetDtypeAndAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape& boxesGeShape = boxesShape->GetStorageShape();
    const gert::Shape& shapeHwGeShape = shapeHwShape->GetStorageShape();
    uint64_t boxesRank = 0;
    ret = ValidateShapes(boxesGeShape, shapeHwGeShape, boxesRank);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ge::graphStatus numStatus = ge::GRAPH_SUCCESS;
    uint64_t num = ComputeNum(boxesGeShape, boxesRank, reversedBox_, opName_, numStatus);
    if (numStatus != ge::GRAPH_SUCCESS) {
        return numStatus;
    }

    tilingData_.batch = static_cast<uint64_t>(boxesGeShape.GetDim(0));
    tilingData_.coordNum = COORD_NUM;
    tilingData_.num = num;
    return ge::GRAPH_SUCCESS;
}

void NormalizeBBoxTilingForRegbase::ComputeTileLen()
{
    uint64_t divisorBlocks = reversedBox_ ? 2 : 1;
    uint64_t ubBufFactor = 2 + 2 + divisorBlocks;
    uint64_t blockAlignElems = VREPEAT_SIZE / boxesDtypeSize_;
    uint64_t tileLen = 0;
    if (ubSize_ > UB_RESERVE) {
        tileLen = (ubSize_ - UB_RESERVE) / (ubBufFactor * boxesDtypeSize_);
    }
    tileLen = tileLen / blockAlignElems * blockAlignElems;
    if (tileLen > MAX_TILE_LEN) {
        tileLen = MAX_TILE_LEN;
    }
    if (tileLen < blockAlignElems) {
        tileLen = blockAlignElems;
    }
    tilingData_.tileLen = tileLen;
}

void NormalizeBBoxTilingForRegbase::SplitByBatch(uint64_t batch)
{
    tilingData_.splitMode = 1;
    uint64_t usedCoreNum = std::min(batch, totalCoreNum_);
    uint64_t batchPerCore = Ops::Base::CeilDiv(batch, usedCoreNum);
    uint64_t bigCoreNum = batch - usedCoreNum * (batchPerCore - 1);
    tilingData_.usedCoreNum = usedCoreNum;
    tilingData_.batchPerCore = batchPerCore;
    tilingData_.tailBatchNum = batchPerCore - 1;
    tilingData_.bigCoreNum = bigCoreNum;
    blockDim_ = usedCoreNum;
}

void NormalizeBBoxTilingForRegbase::SplitByNum(uint64_t num)
{
    tilingData_.splitMode = 0;
    uint64_t alignFrames = reversedBox_ ? (BLOCK_SIZE / boxesDtypeSize_) : (BLOCK_SIZE / (COORD_NUM * boxesDtypeSize_));
    if (alignFrames == 0) {
        alignFrames = 1;
    }
    uint64_t perCoreRaw = Ops::Base::CeilDiv(num, totalCoreNum_);
    uint64_t numPerCore = Ops::Base::CeilDiv(perCoreRaw, alignFrames) * alignFrames;
    uint64_t usedCoreNum = Ops::Base::CeilDiv(num, numPerCore);
    tilingData_.usedCoreNum = usedCoreNum;
    tilingData_.numPerCore = numPerCore;
    tilingData_.tailNumCore = numPerCore;
    tilingData_.numBigCore = usedCoreNum;
    blockDim_ = usedCoreNum;
}

// 3. data split
ge::graphStatus NormalizeBBoxTilingForRegbase::DoOpTiling()
{
    OP_LOGD(opName_, "NormalizeBBoxTilingForRegbase DoOpTiling.");
    uint64_t batch = tilingData_.batch;
    uint64_t num = tilingData_.num;

    ComputeTileLen();

    tilingData_.splitMode = 0;
    tilingData_.usedCoreNum = 1;
    tilingData_.batchPerCore = 0;
    tilingData_.tailBatchNum = 0;
    tilingData_.bigCoreNum = 0;
    tilingData_.numPerCore = 0;
    tilingData_.tailNumCore = 0;
    tilingData_.numBigCore = 0;

    if (batch == 0 || num == 0) {
        blockDim_ = 1;
        tilingData_.usedCoreNum = 1;
        tilingData_.splitMode = (batch > 1) ? 1 : 0;
        return ge::GRAPH_SUCCESS;
    }

    if (batch > 1) {
        SplitByBatch(batch);
    } else {
        SplitByNum(num);
    }
    return ge::GRAPH_SUCCESS;
}

// 4. high-level api tiling (none)
ge::graphStatus NormalizeBBoxTilingForRegbase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// 5. tiling key (dtype driven by DTYPE_BOXES compile axis; reversedBox via TPL)
uint64_t NormalizeBBoxTilingForRegbase::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(reversedBox_);
    OP_LOGD(opName_, "tilingKey=%lu reversedBox=%d.", tilingKey, static_cast<int>(reversedBox_));
    return tilingKey;
}

// 6. workspace
ge::graphStatus NormalizeBBoxTilingForRegbase::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

// 7. save tiling data
ge::graphStatus NormalizeBBoxTilingForRegbase::PostTiling()
{
    OP_LOGD(opName_, "NormalizeBBoxTilingForRegbase PostTiling.");
    PrintTilingData();

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);
    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    auto* tilingData = context_->GetTilingData<NormalizeBBoxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    *tilingData = tilingData_;
    return ge::GRAPH_SUCCESS;
}

void NormalizeBBoxTilingForRegbase::PrintTilingData()
{
    OP_LOGD(opName_, "batch=%lu num=%lu coordNum=%lu splitMode=%lu usedCoreNum=%lu.", tilingData_.batch,
            tilingData_.num, tilingData_.coordNum, tilingData_.splitMode, tilingData_.usedCoreNum);
    OP_LOGD(opName_, "batchPerCore=%lu tailBatchNum=%lu bigCoreNum=%lu.", tilingData_.batchPerCore,
            tilingData_.tailBatchNum, tilingData_.bigCoreNum);
    OP_LOGD(opName_, "numPerCore=%lu tailNumCore=%lu numBigCore=%lu tileLen=%lu.", tilingData_.numPerCore,
            tilingData_.tailNumCore, tilingData_.numBigCore, tilingData_.tileLen);
}

} // namespace optiling
