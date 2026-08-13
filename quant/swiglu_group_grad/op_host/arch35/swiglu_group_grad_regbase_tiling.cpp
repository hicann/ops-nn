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
 * \file swiglu_group_grad_tiling.cpp
 * \brief SwigluGroupGrad tiling implementation for Ascend950 (DAV_3510)
 */

#include "swiglu_group_grad_regbase_tiling.h"
#include <algorithm>
#include <limits>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

constexpr uint64_t WS_SYS_SIZE = 0U;
constexpr int64_t FP32_BYTES_HOST = 4;
constexpr int64_t FP32_ALIGN_HOST = 8;
constexpr int64_t VEC_ALIGN_HOST = 64;
constexpr int64_t UB_MARGIN_BYTES = 8 * 1024;

namespace {
inline int64_t AlignUpHost(int64_t n, int64_t a)
{
    if (a <= 0) {
        return n;
    }
    return ((n + a - 1) / a) * a;
}

inline int64_t FloorAlignHost(int64_t n, int64_t a)
{
    if (a <= 0) {
        return n;
    }
    return (n / a) * a;
}

inline int64_t CeilDivHost(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

} // namespace

ge::graphStatus SwigluGroupGradArch35Tiling::GetPlatformInfo()
{
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNumAll_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    } else {
        auto compileInfoPtr = tilingContext->GetCompileInfo<SwigluGroupGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(tilingContext->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNumAll_ = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize_ = compileInfoPtr->ubSize;
    }
    OP_CHECK_IF(coreNumAll_ == 0, OP_LOGE(tilingContext->GetNodeName(), "coreNumAll is 0"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext->GetNodeName(), "SwigluGroupGradArch35Tiling GetPlatformInfo: coreNum=%ld, ubSize=%lu",
            coreNumAll_, ubSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::CalcDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    gradYDtype = inputDesc->GetDataType();
    if (gradYDtype != ge::DT_BF16 && gradYDtype != ge::DT_FLOAT16 && gradYDtype != ge::DT_FLOAT) {
        OP_LOGE(tilingContext->GetNodeName(), "grad_y dtype[%s] not supported",
                ge::TypeUtils::DataTypeToSerialString(gradYDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::CheckShape()
{
    auto gradYStorageShape = tilingContext->GetInputShape(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradYStorageShape);
    const gert::Shape& gradYShape = gradYStorageShape->GetStorageShape();
    OP_CHECK_IF(gradYShape.GetDimNum() != DIM_TWO && gradYShape.GetDimNum() != DIM_THREE,
                OP_LOGE(tilingContext->GetNodeName(), "grad_y must be 2D (T, H) or 3D (B, S, H), got %ld dims.",
                        gradYShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    if (gradYShape.GetDimNum() == DIM_TWO) {
        totalRows_ = gradYShape.GetDim(0);
    } else {
        const int64_t gradYDim0 = gradYShape.GetDim(0);
        const int64_t gradYDim1 = gradYShape.GetDim(1);
        OP_CHECK_IF(
            gradYDim0 > 0 && gradYDim1 > std::numeric_limits<int64_t>::max() / gradYDim0,
            OP_LOGE(tilingContext->GetNodeName(), "grad_y rows overflow: dim0=%ld, dim1=%ld", gradYDim0, gradYDim1),
            return ge::GRAPH_FAILED);
        totalRows_ = gradYDim0 * gradYDim1;
    }
    H_ = gradYShape.GetDim(gradYShape.GetDimNum() - 1);

    auto xStorageShape = tilingContext->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != gradYShape.GetDimNum(),
                OP_LOGE(tilingContext->GetNodeName(), "x dims=%ld must match grad_y dims=%ld", xShape.GetDimNum(),
                        gradYShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    int64_t xTotalRows = xShape.GetDim(0);
    if (xShape.GetDimNum() == DIM_THREE) {
        const int64_t xDim0 = xShape.GetDim(0);
        const int64_t xDim1 = xShape.GetDim(1);
        OP_CHECK_IF(xDim0 != gradYShape.GetDim(0) || xDim1 != gradYShape.GetDim(1),
                    OP_LOGE(tilingContext->GetNodeName(), "x [B,S] must match grad_y [B,S]"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(xDim0 > 0 && xDim1 > std::numeric_limits<int64_t>::max() / xDim0,
                    OP_LOGE(tilingContext->GetNodeName(), "x rows overflow: dim0=%ld, dim1=%ld", xDim0, xDim1),
                    return ge::GRAPH_FAILED);
        xTotalRows = xDim0 * xDim1;
    }
    dim2H_ = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_CHECK_IF(xTotalRows != totalRows_,
                OP_LOGE(tilingContext->GetNodeName(), "x rows=%ld != grad_y rows=%ld", xTotalRows, totalRows_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim2H_ != H_ * DIM_TWO,
                OP_LOGE(tilingContext->GetNodeName(), "x.shape[-1]=%ld != 2*H=%ld", dim2H_, H_ * DIM_TWO),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H_ <= 0, OP_LOGE(tilingContext->GetNodeName(), "H=%ld must be > 0", H_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::ParseOptionalInputs()
{
    isWeight_ = 0;
    isYOrigin_ = 0;
    isGroupIndex_ = 0;
    groupIndexG_ = 0;

    auto weightStorageShape = tilingContext->GetOptionalInputShape(WEIGHT_INDEX);
    auto yOriginStorageShape = tilingContext->GetOptionalInputShape(YORIGIN_INDEX);
    auto groupIndexStorageShape = tilingContext->GetOptionalInputShape(GROUP_INDEX_INDEX);
    isWeight_ = (weightStorageShape != nullptr && weightStorageShape->GetStorageShape().GetDimNum() > 0) ? 1 : 0;
    isYOrigin_ = (yOriginStorageShape != nullptr && yOriginStorageShape->GetStorageShape().GetDimNum() > 0) ? 1 : 0;
    isGroupIndex_ = (groupIndexStorageShape != nullptr && groupIndexStorageShape->GetStorageShape().GetDimNum() > 0) ?
                        1 :
                        0;

    OP_CHECK_IF(isWeight_ != isYOrigin_,
                OP_LOGE(tilingContext->GetNodeName(), "weight and y_origin must be provided together"),
                return ge::GRAPH_FAILED);

    if (isWeight_ == 1) {
        const gert::Shape& weightShape = weightStorageShape->GetStorageShape();
        auto weightElementNum = weightShape.GetShapeSize();
        if (weightElementNum != totalRows_) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                tilingContext->GetNodeName(), "weight", std::to_string(weightElementNum).c_str(),
                "The element num of weight must be equal to the product of grad_y leading dims.");
            return ge::GRAPH_FAILED;
        }
        auto gradWeightOut = tilingContext->GetOutputShape(1);
        OP_CHECK_IF(gradWeightOut == nullptr,
                    OP_LOGE(tilingContext->GetNodeName(), "grad_weight must be non-null when weight is present"),
                    return ge::GRAPH_FAILED);
    }

    if (isYOrigin_ == 1) {
        const gert::Shape& yOriginShape = yOriginStorageShape->GetStorageShape();
        const gert::Shape& gradYShape = tilingContext->GetInputShape(GRAD_Y_INDEX)->GetStorageShape();
        OP_CHECK_IF(yOriginShape.GetDimNum() != gradYShape.GetDimNum(),
                    OP_LOGE(tilingContext->GetNodeName(), "y_origin dims=%ld must match grad_y dims=%ld",
                            yOriginShape.GetDimNum(), gradYShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(gradYShape.GetDimNum()); ++i) {
            OP_CHECK_IF(
                yOriginShape.GetDim(i) != gradYShape.GetDim(i),
                OP_LOGE(tilingContext->GetNodeName(), "y_origin.shape[%ld]=%ld must match grad_y.shape[%ld]=%ld", i,
                        yOriginShape.GetDim(i), i, gradYShape.GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    }

    if (isGroupIndex_ == 1) {
        const gert::Shape& groupIndexShape = groupIndexStorageShape->GetStorageShape();
        OP_CHECK_IF(groupIndexShape.GetDimNum() != 1 || groupIndexShape.GetDim(0) < 1,
                    OP_LOGE(tilingContext->GetNodeName(), "group_index must be a non-empty 1D tensor"),
                    return ge::GRAPH_FAILED);
        groupIndexG_ = groupIndexShape.GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::ParseAttrs()
{
    hasClamp_ = 0;
    clampLimit_ = 0.0f;
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* clampLimitAttr = attrs->GetAttrPointer<float>(CLAMPLIMIT_ATTR_INDEX);
    if (clampLimitAttr != nullptr) {
        OP_CHECK_IF(!(*clampLimitAttr >= 0.0f),
                    OP_LOGE(tilingContext->GetNodeName(), "clamp_limit must be greater than or equal to 0, but got %f",
                            *clampLimitAttr),
                    return ge::GRAPH_FAILED);
        if (*clampLimitAttr > 0.0f) {
            hasClamp_ = 1;
            clampLimit_ = *clampLimitAttr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::CalcTilingStrategy()
{
    int64_t HA = AlignUpHost(H_, VEC_ALIGN_HOST);
    int64_t dim2HA = HA * 2;
    int64_t dt = 0;
    if (gradYDtype == ge::DT_FLOAT) {
        dt = 4;
    } else if (gradYDtype == ge::DT_FLOAT16 || gradYDtype == ge::DT_BF16) {
        dt = 2;
    } else {
        dt = 4;
    }

    int64_t ubAvailable = static_cast<int64_t>(ubSize_);
    int64_t ubUsable = ubAvailable - UB_MARGIN_BYTES;
    OP_CHECK_IF(
        ubUsable <= 0,
        OP_LOGE(tilingContext->GetNodeName(), "UB too small: available=%ld, margin=%ld", ubAvailable, UB_MARGIN_BYTES),
        return ge::GRAPH_FAILED);

    auto calcNormalBytes = [&](int64_t rows) -> int64_t {
        int64_t dyQBytes = 2 * rows * HA * dt;
        int64_t xQBytes = 2 * rows * dim2HA * dt;
        int64_t dxOutQBytes = rows * dim2HA * dt;
        int64_t veccalcBytes = 0;
        if (dt < FP32_BYTES_HOST) {
            veccalcBytes += rows * HA * FP32_BYTES_HOST;
            veccalcBytes += rows * dim2HA * FP32_BYTES_HOST;
        }
        int64_t yOriginBytes = 0;
        if (isWeight_ && isYOrigin_) {
            yOriginBytes += rows * HA * dt;
            if (dt < FP32_BYTES_HOST) {
                yOriginBytes += rows * HA * FP32_BYTES_HOST;
            }
        }
        int64_t weightBytes = isWeight_ ?
                                  AlignUpHost((rows - 1) * FP32_BYTES_HOST + VEC_ALIGN_HOST * FP32_BYTES_HOST, 32) :
                                  0;
        int64_t dwOutBytes = isWeight_ ?
                                 AlignUpHost((rows - 1) * FP32_BYTES_HOST + VEC_ALIGN_HOST * FP32_BYTES_HOST, 32) :
                                 0;
        return dyQBytes + xQBytes + dxOutQBytes + veccalcBytes + yOriginBytes + weightBytes + dwOutBytes;
    };

    int64_t dyQPerRow = 2 * HA * dt;
    int64_t xQPerRow = 2 * dim2HA * dt;
    int64_t dxOutQPerRow = dim2HA * dt;
    int64_t veccalcPerRow = 0;
    if (dt < FP32_BYTES_HOST) {
        veccalcPerRow += HA * FP32_BYTES_HOST;
        veccalcPerRow += dim2HA * FP32_BYTES_HOST;
    }
    int64_t yOriginPerRow = 0;
    if (isWeight_ && isYOrigin_) {
        yOriginPerRow += HA * dt;
        if (dt < FP32_BYTES_HOST) {
            yOriginPerRow += HA * FP32_BYTES_HOST;
        }
    }
    int64_t totalPerRowNormalLowerBound = dyQPerRow + xQPerRow + dxOutQPerRow + veccalcPerRow + yOriginPerRow +
                                          (isWeight_ ? 2 * FP32_BYTES_HOST : 0);
    OP_CHECK_IF(
        totalPerRowNormalLowerBound <= 0,
        OP_LOGE(tilingContext->GetNodeName(), "totalPerRowNormal is invalid, value=%ld", totalPerRowNormalLowerBound),
        return ge::GRAPH_FAILED);

    int64_t splitHidden = 0;
    int64_t ubChunkH = HA;
    int64_t numChunksPerRow = 1;
    int64_t blkH = 0;

    int64_t rawBlkH = ubUsable / totalPerRowNormalLowerBound;
    if (rawBlkH <= 0) {
        splitHidden = 1;
    } else {
        blkH = std::min(rawBlkH, totalRows_);
        if (blkH >= FP32_ALIGN_HOST) {
            blkH = FloorAlignHost(blkH, FP32_ALIGN_HOST);
        }
        if (blkH <= 0) {
            blkH = 1;
        }
        while (blkH > 0 && calcNormalBytes(blkH) > ubUsable) {
            if (blkH > FP32_ALIGN_HOST) {
                blkH = FloorAlignHost(blkH - 1, FP32_ALIGN_HOST);
            } else {
                blkH--;
            }
        }
        if (blkH <= 0) {
            splitHidden = 1;
        }
    }

    if (splitHidden == 1) {
        auto calcSplitBytes = [&](int64_t chunkH) -> int64_t {
            int64_t chunks = CeilDivHost(HA, chunkH);
            int64_t dyQBytes = chunkH * dt;
            int64_t xQBytes = 2 * chunkH * dt;
            int64_t dxOutQBytes = 2 * chunkH * dt;
            int64_t veccalcBytes = 0;
            if (dt < FP32_BYTES_HOST) {
                veccalcBytes += chunkH * FP32_BYTES_HOST;
                veccalcBytes += 2 * chunkH * FP32_BYTES_HOST;
            }
            int64_t yOriginBytes = 0;
            if (isWeight_ && isYOrigin_) {
                yOriginBytes += chunkH * dt;
                if (dt < FP32_BYTES_HOST) {
                    yOriginBytes += chunkH * FP32_BYTES_HOST;
                }
            }
            int64_t dwAccumBytes = AlignUpHost(AlignUpHost(chunks, VEC_ALIGN_HOST) * FP32_BYTES_HOST, 32);
            int64_t weightBytes = isWeight_ ? AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, 32) : 0;
            int64_t dwOutBytes = isWeight_ ? AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, 32) : 0;
            return dyQBytes + xQBytes + dxOutQBytes + veccalcBytes + yOriginBytes + dwAccumBytes + weightBytes +
                   dwOutBytes;
        };

        int64_t perElementBytes = 5 * dt;
        if (dt < FP32_BYTES_HOST) {
            perElementBytes += 3 * FP32_BYTES_HOST;
        }
        if (isWeight_ && isYOrigin_) {
            perElementBytes += dt;
            if (dt < FP32_BYTES_HOST) {
                perElementBytes += FP32_BYTES_HOST;
            }
        }
        int64_t minFixedBytes = AlignUpHost(VEC_ALIGN_HOST * FP32_BYTES_HOST, 32) +
                                (isWeight_ ? 2 * AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, 32) : 0);
        OP_CHECK_IF(perElementBytes <= 0 || ubUsable <= minFixedBytes,
                    OP_LOGE(tilingContext->GetNodeName(),
                            "invalid chunk ub params, ubUsable=%ld, minFixedBytes=%ld, perElementBytes=%ld", ubUsable,
                            minFixedBytes, perElementBytes),
                    return ge::GRAPH_FAILED);

        int64_t rawUbChunkH = (ubUsable - minFixedBytes) / perElementBytes;
        ubChunkH = FloorAlignHost(rawUbChunkH, VEC_ALIGN_HOST);
        if (ubChunkH > HA) {
            ubChunkH = HA;
        }
        if (ubChunkH < VEC_ALIGN_HOST) {
            ubChunkH = VEC_ALIGN_HOST;
        }
        while (ubChunkH >= VEC_ALIGN_HOST && calcSplitBytes(ubChunkH) > ubUsable) {
            ubChunkH -= VEC_ALIGN_HOST;
        }
        OP_CHECK_IF(
            ubChunkH < VEC_ALIGN_HOST,
            OP_LOGE(tilingContext->GetNodeName(), "ubChunkH is invalid, ubUsable=%ld, H=%ld, HA=%ld", ubUsable, H_, HA),
            return ge::GRAPH_FAILED);

        numChunksPerRow = CeilDivHost(HA, ubChunkH);
        blkH = 1;
    } else {
        ubChunkH = HA;
        numChunksPerRow = 1;
    }

    int64_t coreNum = coreNumAll_;
    int64_t rFB = CeilDivHost(totalRows_, coreNum);
    rFB = CeilDivHost(rFB, blkH) * blkH;
    if (rFB <= 0) {
        rFB = 1;
    }
    int64_t usedCoreNum = CeilDivHost(totalRows_, rFB);

    tiling->coreNumAll = coreNum;
    tiling->totalRows = totalRows_;
    tiling->H = H_;
    tiling->blkH = blkH;
    tiling->splitHidden = splitHidden;
    tiling->clampLimit = clampLimit_;
    tiling->groupIndexG = groupIndexG_;
    tiling->ubChunkH = ubChunkH;
    tiling->numChunksPerRow = numChunksPerRow;
    tiling->blockFactor = rFB;

    schMode_ = TPL_REGBASE_KERNEL;

    OP_LOGD(tilingContext->GetNodeName(),
            "CalcTilingStrategy: coreNum=%ld, T=%ld, H=%ld, HA=%ld, blkH=%ld, rFB=%ld, usedCores=%ld, splitHidden=%ld, "
            "ubChunkH=%ld, numChunksPerRow=%ld, isWeight=%ld, isYOrigin=%ld, isGroupIndex=%ld, hasClamp=%ld, "
            "clampLimit=%.2f, schMode=%ld",
            coreNum, totalRows_, H_, HA, blkH, rFB, usedCoreNum, splitHidden, ubChunkH, numChunksPerRow, isWeight_,
            isYOrigin_, isGroupIndex_, hasClamp_, clampLimit_, schMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcDtype() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CalcDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseOptionalInputs() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "ParseOptionalInputs failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "ParseAttrs failed"),
                return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<SwigluGroupGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SwigluGroupGradTilingData), 0, sizeof(SwigluGroupGradTilingData)) != EOK,
                OP_LOGE(tilingContext->GetNodeName(), "memset tiling data error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool SwigluGroupGradArch35Tiling::IsCapable()
{
    bool preferSimt = (totalRows_ < coreNumAll_ / 2) || (H_ < VL_FP32_ARCH35 / 2);
    return !preferSimt;
}

ge::graphStatus SwigluGroupGradArch35Tiling::DoOpTiling()
{
    if (totalRows_ == 0 || H_ == 0) {
        tiling->coreNumAll = coreNumAll_;
        tiling->totalRows = 0;
        tiling->H = 0;
        tiling->blkH = 0;
        tiling->splitHidden = 0;
        tiling->clampLimit = 0.0f;
        tiling->blockFactor = 0;
        tiling->groupIndexG = 0;
        tiling->ubChunkH = 0;
        tiling->numChunksPerRow = 0;
        schMode_ = TPL_REGBASE_KERNEL;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CalcTilingStrategy() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "CalcTilingStrategy failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SwigluGroupGradArch35Tiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    tilingContext->SetTilingKey(tilingKey);

    if (totalRows_ == 0 || H_ == 0) {
        tilingContext->SetBlockDim(1);
    } else {
        int64_t usedCoreNum = CeilDivHost(totalRows_, tiling->blockFactor);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tilingContext->SetBlockDim(usedCoreNum);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t SwigluGroupGradArch35Tiling::GetTilingKey() const
{
    uint64_t key = GET_TPL_TILING_KEY(static_cast<uint32_t>(schMode_), static_cast<uint32_t>(hasClamp_),
                                      static_cast<uint32_t>(isWeight_), static_cast<uint32_t>(isYOrigin_),
                                      static_cast<uint32_t>(isGroupIndex_));
    OP_LOGI(tilingContext->GetNodeName(),
            "GetTilingKey: key=%lu, schMode=%ld, hasClamp=%ld, isWeight=%ld, isYOrigin=%ld, isGroupIndex=%ld", key,
            schMode_, hasClamp_, isWeight_, isYOrigin_, isGroupIndex_);
    return key;
}

REGISTER_OPS_TILING_TEMPLATE(SwigluGroupGrad, SwigluGroupGradArch35Tiling, 0);

} // namespace optiling
