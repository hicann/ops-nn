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
 * \file swiglu_group_grad_regbase_tiling.cpp
 * \brief brief Tiling implementation for SwigluGroupGrad_regbase_tiling
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
constexpr int64_t FP16_BYTES_HOST = 2;
constexpr int64_t FP32_ALIGN_HOST = 8;
constexpr int64_t VEC_ALIGN_HOST = 64;
constexpr int64_t UB_MARGIN_BYTES = 8 * 1024;
// SwiGLU splits the input into gate/up halves, so every row is 2x hidden.
constexpr int64_t SWIGLU_HALF_COUNT = 2;

// Dedicated RegBase path for very small T and ultra-wide H. The general
// split-hidden path remains unchanged for every shape outside this domain.
// Cover the very-small-T family, including T=4/8 with multi-million H.
// Keeping the boundary at one eighth of a 64-core vector cluster confines the
// new fixed-tree path to shapes whose row dimension cannot provide enough
// independent reduction work by itself.
constexpr int64_t SIMD_ULTRAWIDE_MAX_ROWS_HOST = 8;
constexpr int64_t SIMD_ULTRAWIDE_MIN_H_HOST = 256 * 1024;
constexpr int64_t SIMD_ULTRAWIDE_SPLIT_MODE_HOST = 2;
constexpr int64_t NUMPY_SPLIT_ALIGNMENT_HOST = 8;

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

// Return the largest NumPy pairwise-reduction subtree after a fixed number
// of binary levels. NumPy aligns every left child to eight FP32 elements.
// Following the larger child at each level gives a safe upper bound for all
// nodes at that depth and therefore for every per-core UB tile.
inline int64_t CalcMaxFixedDepthNodeSize(int64_t count, int64_t taskCount)
{
    while (taskCount > 1) {
        int64_t leftCount = FloorAlignHost(count / 2, NUMPY_SPLIT_ALIGNMENT_HOST);
        int64_t rightCount = count - leftCount;
        count = std::max(leftCount, rightCount);
        taskCount >>= 1;
    }
    return count;
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
    ubAlignBytes_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(tilingContext));
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
    OP_CHECK_IF(
        gradYShape.GetDimNum() < 1,
        OP_LOGE(tilingContext->GetNodeName(), "grad_y must be at least 1D, got %ld dims.", gradYShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    totalRows_ = 1;
    for (size_t i = 0; i < gradYShape.GetDimNum() - 1; ++i) {
        totalRows_ *= gradYShape.GetDim(i);
    }
    hiddenSize_ = gradYShape.GetDim(gradYShape.GetDimNum() - 1);

    auto xStorageShape = tilingContext->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() < 1,
                OP_LOGE(tilingContext->GetNodeName(), "x must be at least 1D, got %ld dims.", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    int64_t xTotalRows = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        xTotalRows *= xShape.GetDim(i);
    }
    doubleHiddenSize_ = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_CHECK_IF(xTotalRows != totalRows_,
                OP_LOGE(tilingContext->GetNodeName(), "x rows=%ld != grad_y rows=%ld", xTotalRows, totalRows_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        doubleHiddenSize_ != hiddenSize_ * DIM_TWO,
        OP_LOGE(tilingContext->GetNodeName(), "x.shape[-1]=%ld != 2*H=%ld", doubleHiddenSize_, hiddenSize_ * DIM_TWO),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(hiddenSize_ <= 0, OP_LOGE(tilingContext->GetNodeName(), "H=%ld must be > 0", hiddenSize_),
                return ge::GRAPH_FAILED);
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
        OP_CHECK_IF(yOriginShape.GetDimNum() < 1, OP_LOGE(tilingContext->GetNodeName(), "y_origin must be at least 1D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(yOriginShape.GetDim(yOriginShape.GetDimNum() - 1) != hiddenSize_,
                    OP_LOGE(tilingContext->GetNodeName(), "y_origin.shape[-1]=%ld must equal H=%ld",
                            yOriginShape.GetDim(yOriginShape.GetDimNum() - 1), hiddenSize_),
                    return ge::GRAPH_FAILED);
        int64_t yOriginTotalRows = 1;
        for (size_t i = 0; i < yOriginShape.GetDimNum() - 1; ++i) {
            yOriginTotalRows *= yOriginShape.GetDim(i);
        }
        OP_CHECK_IF(yOriginTotalRows != totalRows_,
                    OP_LOGE(tilingContext->GetNodeName(), "y_origin outer numel(%ld) must equal totalRows(%ld)",
                            yOriginTotalRows, totalRows_),
                    return ge::GRAPH_FAILED);
    }

    if (isGroupIndex_ == 1) {
        const gert::Shape& groupIndexShape = groupIndexStorageShape->GetStorageShape();
        OP_CHECK_IF(groupIndexShape.GetDimNum() != 1 || groupIndexShape.GetDim(0) < 1,
                    OP_LOGE(tilingContext->GetNodeName(),
                            "group_index must be a non-empty 1D tensor, got dimNum=%ld, dim[0]=%ld",
                            groupIndexShape.GetDimNum(), groupIndexShape.GetDim(0)),
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
        OP_CHECK_IF(*clampLimitAttr != -1.0f && !(*clampLimitAttr > 0.0f),
                    OP_LOGE(tilingContext->GetNodeName(), "clamp_limit must be -1.0 (no clamp) or > 0.0, but got %f",
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
    int64_t alignedHiddenSize = AlignUpHost(hiddenSize_, VEC_ALIGN_HOST);
    int64_t dim2HA = alignedHiddenSize * SWIGLU_HALF_COUNT;
    int64_t dt = 0;
    if (gradYDtype == ge::DT_FLOAT) {
        dt = FP32_BYTES_HOST;
    } else if (gradYDtype == ge::DT_FLOAT16 || gradYDtype == ge::DT_BF16) {
        dt = FP16_BYTES_HOST;
    } else {
        dt = FP32_BYTES_HOST;
    }

    const bool useUltraWideSimd = totalRows_ > 0 && totalRows_ <= SIMD_ULTRAWIDE_MAX_ROWS_HOST &&
                                  hiddenSize_ >= SIMD_ULTRAWIDE_MIN_H_HOST && isWeight_ == 1 && isYOrigin_ == 1;

    int64_t ubAvailable = static_cast<int64_t>(ubSize_);
    int64_t ubUsable = ubAvailable - UB_MARGIN_BYTES;
    OP_CHECK_IF(
        ubUsable <= 0,
        OP_LOGE(tilingContext->GetNodeName(), "UB too small: available=%ld, margin=%ld", ubAvailable, UB_MARGIN_BYTES),
        return ge::GRAPH_FAILED);

    auto calcNormalBytes = [&](int64_t rows) -> int64_t {
        int64_t dyQBytes = 2 * rows * alignedHiddenSize * dt;
        int64_t xQBytes = 2 * rows * dim2HA * dt;
        int64_t dxOutQBytes = rows * dim2HA * dt;
        int64_t veccalcBytes = 0;
        if (dt < FP32_BYTES_HOST) {
            veccalcBytes += rows * alignedHiddenSize * FP32_BYTES_HOST;
            veccalcBytes += rows * dim2HA * FP32_BYTES_HOST;
        }
        int64_t yOriginBytes = 0;
        if (isWeight_ && isYOrigin_) {
            yOriginBytes += rows * alignedHiddenSize * dt;
            if (dt < FP32_BYTES_HOST) {
                yOriginBytes += rows * alignedHiddenSize * FP32_BYTES_HOST;
            }
        }
        int64_t weightBytes = isWeight_ ? AlignUpHost((rows - 1) * FP32_BYTES_HOST + VEC_ALIGN_HOST * FP32_BYTES_HOST,
                                                      ubAlignBytes_) :
                                          0;
        int64_t dwOutBytes = isWeight_ ? AlignUpHost((rows - 1) * FP32_BYTES_HOST + VEC_ALIGN_HOST * FP32_BYTES_HOST,
                                                     ubAlignBytes_) :
                                         0;
        return dyQBytes + xQBytes + dxOutQBytes + veccalcBytes + yOriginBytes + weightBytes + dwOutBytes;
    };

    int64_t dyQPerRow = 2 * alignedHiddenSize * dt;
    int64_t xQPerRow = 2 * dim2HA * dt;
    int64_t dxOutQPerRow = dim2HA * dt;
    int64_t veccalcPerRow = 0;
    if (dt < FP32_BYTES_HOST) {
        veccalcPerRow += alignedHiddenSize * FP32_BYTES_HOST;
        veccalcPerRow += dim2HA * FP32_BYTES_HOST;
    }
    int64_t yOriginPerRow = 0;
    if (isWeight_ && isYOrigin_) {
        yOriginPerRow += alignedHiddenSize * dt;
        if (dt < FP32_BYTES_HOST) {
            yOriginPerRow += alignedHiddenSize * FP32_BYTES_HOST;
        }
    }
    int64_t totalPerRowNormalLowerBound = dyQPerRow + xQPerRow + dxOutQPerRow + veccalcPerRow + yOriginPerRow +
                                          (isWeight_ ? 2 * FP32_BYTES_HOST : 0);
    OP_CHECK_IF(
        totalPerRowNormalLowerBound <= 0,
        OP_LOGE(tilingContext->GetNodeName(), "totalPerRowNormal is invalid, value=%ld", totalPerRowNormalLowerBound),
        return ge::GRAPH_FAILED);

    int64_t splitHidden = 0;
    int64_t ubChunkH = alignedHiddenSize;
    int64_t numChunksPerRow = 1;
    int64_t blkH = 0;

    // Small-T/large-H shapes need hidden-axis tasks to reach the target AIV
    // count. This keeps IsCapable's hiddenCanFillAllCores promise true in the
    // actual launch instead of merely checking it at admission time.
    int64_t hiddenSlicesNeededForTargetCore = CeilDivHost(coreNumAll_, totalRows_);
    int64_t maxHiddenSlicesByVector = CeilDivHost(alignedHiddenSize, VEC_ALIGN_HOST);
    int64_t targetHiddenSlices = std::min(hiddenSlicesNeededForTargetCore, maxHiddenSlicesByVector);
    if (targetHiddenSlices <= 0) {
        targetHiddenSlices = 1;
    }
    bool forceHiddenCoreSplit = targetHiddenSlices > 1;

    int64_t rawBlkH = ubUsable / totalPerRowNormalLowerBound;
    int64_t rowParallelCoreNum = std::min(coreNumAll_, totalRows_);
    if (rowParallelCoreNum <= 0) {
        rowParallelCoreNum = 1;
    }
    int64_t maxRowsPerCore = CeilDivHost(totalRows_, rowParallelCoreNum);
    if (forceHiddenCoreSplit || rawBlkH <= 0) {
        splitHidden = 1;
    } else {
        // A balanced row partition never gives one core more than
        // maxRowsPerCore rows, so reserving a larger tile only wastes UB.
        blkH = std::min(rawBlkH, maxRowsPerCore);
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
            int64_t chunks = CeilDivHost(alignedHiddenSize, chunkH);
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
            int64_t dwAccumBytes = useUltraWideSimd ? 0 :
                                                      AlignUpHost(AlignUpHost(chunks, VEC_ALIGN_HOST) * FP32_BYTES_HOST,
                                                                  ubAlignBytes_);
            int64_t weightBytes = isWeight_ ? AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, ubAlignBytes_) : 0;
            int64_t dwOutBytes = isWeight_ ? AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, ubAlignBytes_) : 0;
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
        int64_t minFixedBytes = AlignUpHost(VEC_ALIGN_HOST * FP32_BYTES_HOST, ubAlignBytes_) +
                                (isWeight_ ? 2 * AlignUpHost(FP32_ALIGN_HOST * FP32_BYTES_HOST, ubAlignBytes_) : 0);
        OP_CHECK_IF(perElementBytes <= 0 || ubUsable <= minFixedBytes,
                    OP_LOGE(tilingContext->GetNodeName(),
                            "split-hidden UB budget is invalid: ubUsable=%ld, fixedBytes=%ld, perElementBytes=%ld",
                            ubUsable, minFixedBytes, perElementBytes),
                    return ge::GRAPH_FAILED);

        int64_t rawUbChunkH = (ubUsable - minFixedBytes) / perElementBytes;
        ubChunkH = FloorAlignHost(rawUbChunkH, VEC_ALIGN_HOST);
        if (ubChunkH > alignedHiddenSize) {
            ubChunkH = alignedHiddenSize;
        }
        if (ubChunkH < VEC_ALIGN_HOST) {
            ubChunkH = VEC_ALIGN_HOST;
        }
        while (ubChunkH >= VEC_ALIGN_HOST && calcSplitBytes(ubChunkH) > ubUsable) {
            ubChunkH -= VEC_ALIGN_HOST;
        }
        OP_CHECK_IF(
            ubChunkH < VEC_ALIGN_HOST,
            OP_LOGE(tilingContext->GetNodeName(), "ubChunkH is invalid, ubUsable=%ld, H=%ld, alignedHiddenSize=%ld",
                    ubUsable, hiddenSize_, alignedHiddenSize),
            return ge::GRAPH_FAILED);

        if (forceHiddenCoreSplit) {
            // Use the largest aligned chunk that still creates enough hidden
            // tasks to occupy the target cores.
            int64_t targetChunkH = alignedHiddenSize;
            if (targetHiddenSlices > 1) {
                targetChunkH = FloorAlignHost((alignedHiddenSize - 1) / (targetHiddenSlices - 1), VEC_ALIGN_HOST);
            }
            targetChunkH = std::max(targetChunkH, VEC_ALIGN_HOST);
            targetChunkH = std::min(targetChunkH, alignedHiddenSize);
            ubChunkH = std::min(ubChunkH, targetChunkH);
        }

        numChunksPerRow = CeilDivHost(alignedHiddenSize, ubChunkH);
        if (useUltraWideSimd) {
            // Cut the exact NumPy pairwise tree at one fixed depth. A fixed
            // depth gives a power-of-two task count, so every per-task partial
            // is a real subtree and can later be merged without changing a
            // single FP32 addition in the reference reduction order.
            int64_t minTaskCount = CeilDivHost(coreNumAll_, totalRows_);
            int64_t fixedTreeTaskCount = 1;
            while (fixedTreeTaskCount < minTaskCount) {
                OP_CHECK_IF(fixedTreeTaskCount > std::numeric_limits<int64_t>::max() / 2,
                            OP_LOGE(tilingContext->GetNodeName(), "ultra-wide task count overflow"),
                            return ge::GRAPH_FAILED);
                fixedTreeTaskCount <<= 1;
            }

            int64_t maxTaskElements = CalcMaxFixedDepthNodeSize(hiddenSize_, fixedTreeTaskCount);
            int64_t alignedMaxTaskElements = AlignUpHost(maxTaskElements, VEC_ALIGN_HOST);
            while (alignedMaxTaskElements > ubChunkH) {
                OP_CHECK_IF(fixedTreeTaskCount > std::numeric_limits<int64_t>::max() / 2,
                            OP_LOGE(tilingContext->GetNodeName(), "ultra-wide task count overflow while fitting UB"),
                            return ge::GRAPH_FAILED);
                fixedTreeTaskCount <<= 1;
                maxTaskElements = CalcMaxFixedDepthNodeSize(hiddenSize_, fixedTreeTaskCount);
                alignedMaxTaskElements = AlignUpHost(maxTaskElements, VEC_ALIGN_HOST);
            }

            OP_CHECK_IF(fixedTreeTaskCount <= 0 || alignedMaxTaskElements <= 0 ||
                            calcSplitBytes(alignedMaxTaskElements) > ubUsable,
                        OP_LOGE(tilingContext->GetNodeName(),
                                "ultra-wide fixed-tree tile does not fit UB: tasks=%ld, tile=%ld, ub=%ld",
                                fixedTreeTaskCount, alignedMaxTaskElements, ubUsable),
                        return ge::GRAPH_FAILED);
            splitHidden = SIMD_ULTRAWIDE_SPLIT_MODE_HOST;
            ubChunkH = alignedMaxTaskElements;
            numChunksPerRow = fixedTreeTaskCount;
        }
        blkH = 1;
    } else {
        ubChunkH = alignedHiddenSize;
        numChunksPerRow = 1;
    }

    int64_t coreNum = coreNumAll_;
    int64_t totalTasks = totalRows_;
    if (splitHidden != 0) {
        if (numChunksPerRow > 0 && totalRows_ > std::numeric_limits<int64_t>::max() / numChunksPerRow) {
            totalTasks = std::numeric_limits<int64_t>::max();
        } else {
            totalTasks = totalRows_ * numChunksPerRow;
        }
    }
    int64_t usedCoreNum = std::min(coreNum, totalTasks);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }

    tiling->coreNumAll = coreNum;
    tiling->totalRows = totalRows_;
    tiling->hiddenSize = hiddenSize_;
    tiling->rowsPerTile = blkH;
    tiling->splitHiddenMode = splitHidden;
    tiling->clampLimit = clampLimit_;
    tiling->groupIndexG = groupIndexG_;
    tiling->hiddenChunkSize = ubChunkH;
    tiling->chunksPerRow = numChunksPerRow;
    // launchedCoreNum stores the actual launched AIV core count. The RegBase
    // kernel uses it for balanced row ownership and hidden-task strides.
    tiling->launchedCoreNum = usedCoreNum;

    schMode_ = TPL_REGBASE_KERNEL;

    OP_LOGD(tilingContext->GetNodeName(),
            "CalcTilingStrategy: coreNum=%ld, T=%ld, H=%ld, alignedHiddenSize=%ld, blkH=%ld, usedCores=%ld, "
            "splitHidden=%ld, "
            "ubChunkH=%ld, numChunksPerRow=%ld, totalTasks=%ld, isWeight=%ld, isYOrigin=%ld, isGroupIndex=%ld, "
            "hasClamp=%ld, clampLimit=%.2f, schMode=%ld",
            coreNum, totalRows_, hiddenSize_, alignedHiddenSize, blkH, usedCoreNum, splitHidden, ubChunkH,
            numChunksPerRow, totalTasks, isWeight_, isYOrigin_, isGroupIndex_, hasClamp_, clampLimit_, schMode_);
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
    bool useUltraWideSimd = totalRows_ > 0 && totalRows_ <= SIMD_ULTRAWIDE_MAX_ROWS_HOST &&
                            hiddenSize_ >= SIMD_ULTRAWIDE_MIN_H_HOST && isWeight_ == 1 && isYOrigin_ == 1;
    if (!useUltraWideSimd) {
        bool vectorLaneUsable = hiddenSize_ >= VL_FP32_ARCH35 / 2;
        if (!vectorLaneUsable || totalRows_ <= 0) {
            return false;
        }

        int64_t usefulCoreTarget = std::max<int64_t>(1, CeilDivHost(coreNumAll_, 2));
        int64_t hiddenVectorTileNum = CeilDivHost(hiddenSize_, VEC_ALIGN_HOST);
        int64_t hiddenTilesNeededPerRow = CeilDivHost(usefulCoreTarget, totalRows_);
        if (hiddenVectorTileNum < hiddenTilesNeededPerRow) {
            return false;
        }
    }

    return CalcTilingStrategy() == ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradArch35Tiling::DoOpTiling()
{
    if (totalRows_ == 0 || hiddenSize_ == 0) {
        tiling->coreNumAll = coreNumAll_;
        tiling->totalRows = 0;
        tiling->hiddenSize = 0;
        tiling->rowsPerTile = 0;
        tiling->splitHiddenMode = 0;
        tiling->clampLimit = 0.0f;
        tiling->launchedCoreNum = 0;
        tiling->groupIndexG = 0;
        tiling->hiddenChunkSize = 0;
        tiling->chunksPerRow = 0;
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

    if (totalRows_ == 0 || hiddenSize_ == 0) {
        tilingContext->SetBlockDim(1);
    } else {
        int64_t usedCoreNum = tiling->launchedCoreNum;
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tilingContext->SetBlockDim(usedCoreNum);
        if (tiling->splitHiddenMode == SIMD_ULTRAWIDE_SPLIT_MODE_HOST && usedCoreNum > 1) {
            // The fixed-tree path uses hard inter-core synchronization around
            // its dxOut scratch region. Batch scheduling prevents a partially
            // resident launch from waiting forever at SyncAll.
            tilingContext->SetScheduleMode(1);
        }
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
