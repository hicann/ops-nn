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
 * \file sparse_apply_adagrad_tiling_arch35.cpp
 * \brief tiling implementation for sparse_apply_adagrad (arch35)
 */

#include "sparse_apply_adagrad_tiling_arch35.h"

#include <algorithm>
#include <string>
#include <vector>
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "optim/sparse_apply_adagrad/op_kernel/arch35/sparse_apply_adagrad_tiling_data.h"
#include "optim/sparse_apply_adagrad/op_kernel/arch35/sparse_apply_adagrad_tiling_key.h"

namespace optiling {

constexpr int32_t IDX_VAR = 0;
constexpr int32_t IDX_ACCUM = 1;
constexpr int32_t IDX_LR = 2;
constexpr int32_t IDX_GRAD = 3;
constexpr int32_t IDX_INDICES = 4;
constexpr int32_t ATTR_USE_LOCKING = 0;
constexpr int32_t ATTR_UPDATE_SLOTS = 1;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t UB_RESERVED_BYTE = 8 * 1024;
constexpr int64_t CONST_FIVE = 5;
constexpr int64_t MAX_DIMS = 8;
constexpr int64_t BLOCK_LENGTH = 32;
constexpr int64_t BLOCK_NUM = 8;
constexpr int64_t REG_LENGTH = 256;
constexpr int64_t REG_FLOAT_NUM = 64;
constexpr int64_t MAX_INDICES_BLOCK = 1024;

struct SparseApplyAdagradShapeInfo {
    int64_t firstDim = 0;
    int64_t rowSize = 0;
    int64_t gradDim0 = 0;
    int64_t indicesDim0 = 0;
};

struct SparseApplyAdagradBaseTiling {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t firstDim = 0;
    int64_t rowSize = 0;
    int64_t gradDim0 = 0;
    int64_t indicesDim0 = 0;
    int64_t indicesDim0Align = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    bool useLocking = false;
    bool updateSlots = true;
};

static int64_t AlignUp(int64_t value, int64_t align) { return ((value + align - 1) / align) * align; }

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "coreNum must be positive.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizeTemp = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
    ubSize = static_cast<int64_t>(ubSizeTemp);
    if (ubSize <= UB_RESERVED_BYTE + BLOCK_LENGTH) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                              "ubSize is too small for sparse_apply_adagrad.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static bool GetBoolAttr(gert::TilingContext* context, int32_t idx, bool defaultValue)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return defaultValue;
    }
    const bool* attr = attrs->GetAttrPointer<bool>(idx);
    return attr == nullptr ? defaultValue : *attr;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    auto varDesc = context->GetInputDesc(IDX_VAR);
    OP_CHECK_IF(varDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input var"),
                return ge::GRAPH_FAILED);
    auto accumDesc = context->GetInputDesc(IDX_ACCUM);
    OP_CHECK_IF(accumDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input accum"),
                return ge::GRAPH_FAILED);
    auto lrDesc = context->GetInputDesc(IDX_LR);
    OP_CHECK_IF(lrDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input lr"),
                return ge::GRAPH_FAILED);
    auto gradDesc = context->GetInputDesc(IDX_GRAD);
    OP_CHECK_IF(gradDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input grad"),
                return ge::GRAPH_FAILED);
    auto indicesDesc = context->GetInputDesc(IDX_INDICES);
    OP_CHECK_IF(indicesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input indices"),
                return ge::GRAPH_FAILED);

    auto varDtype = varDesc->GetDataType();
    auto accumDtype = accumDesc->GetDataType();
    auto lrDtype = lrDesc->GetDataType();
    auto gradDtype = gradDesc->GetDataType();
    auto indicesDtype = indicesDesc->GetDataType();
    OP_CHECK_IF(varDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "var",
                                          ge::TypeUtils::DataTypeToSerialString(varDtype).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                          ge::TypeUtils::DataTypeToSerialString(indicesDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != accumDtype || varDtype != lrDtype || varDtype != gradDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "accum, lr and grad",
                                                       ge::TypeUtils::DataTypeToSerialString(accumDtype) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(lrDtype) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(gradDtype),
                                                       "accum, lr and grad must have the same dtype as var (float32)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, SparseApplyAdagradShapeInfo& shapeInfo)
{
    auto varInput = context->GetInputShape(IDX_VAR);
    OP_CHECK_IF(varInput == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input var"),
                return ge::GRAPH_FAILED);
    auto accumInput = context->GetInputShape(IDX_ACCUM);
    OP_CHECK_IF(accumInput == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input accum"),
                return ge::GRAPH_FAILED);
    auto lrInput = context->GetInputShape(IDX_LR);
    OP_CHECK_IF(lrInput == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input lr"),
                return ge::GRAPH_FAILED);
    auto gradInput = context->GetInputShape(IDX_GRAD);
    OP_CHECK_IF(gradInput == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input grad"),
                return ge::GRAPH_FAILED);
    auto indicesInput = context->GetInputShape(IDX_INDICES);
    OP_CHECK_IF(indicesInput == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input indices"),
                return ge::GRAPH_FAILED);

    auto varShape = varInput->GetStorageShape();
    auto accumShape = accumInput->GetStorageShape();
    auto lrShape = lrInput->GetStorageShape();
    auto gradShape = gradInput->GetStorageShape();
    auto indicesShape = indicesInput->GetStorageShape();

    OP_CHECK_IF(varShape.GetDimNum() == 0 || varShape.GetDimNum() > MAX_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "var",
                                                         std::to_string(varShape.GetDimNum()) + "D",
                                                         "the dim num of var must be in [1, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "indices",
                                                         std::to_string(indicesShape.GetDimNum()) + "D",
                                                         "indices must be a 1D tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(accumShape.GetDimNum() != varShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "accum and var",
                    std::to_string(accumShape.GetDimNum()) + "D and " + std::to_string(varShape.GetDimNum()) + "D",
                    "accum must have the same dim num as var"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradShape.GetDimNum() != varShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "grad and var",
                    std::to_string(gradShape.GetDimNum()) + "D and " + std::to_string(varShape.GetDimNum()) + "D",
                    "grad must have the same dim num as var"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(lrShape.GetDimNum() != 0 && !(lrShape.GetDimNum() == 1 && lrShape.GetDim(0) == 1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "lr",
                                                      "dim num = " + std::to_string(lrShape.GetDimNum()),
                                                      "lr must be a scalar (shape empty or [1])"),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < varShape.GetDimNum(); ++i) {
        OP_CHECK_IF(
            varShape.GetDim(i) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "var", "dim[" + std::to_string(i) + "] = " + std::to_string(varShape.GetDim(i)),
                "each dim of var must be positive"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(accumShape.GetDim(i) != varShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "accum and var",
                        "accum dim[" + std::to_string(i) + "] = " + std::to_string(accumShape.GetDim(i)) +
                            ", var dim[" + std::to_string(i) + "] = " + std::to_string(varShape.GetDim(i)),
                        "accum must have the same shape as var"),
                    return ge::GRAPH_FAILED);
        if (i > 0) {
            OP_CHECK_IF(gradShape.GetDim(i) != varShape.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context->GetNodeName(), "grad and var",
                            "grad dim[" + std::to_string(i) + "] = " + std::to_string(gradShape.GetDim(i)) +
                                ", var dim[" + std::to_string(i) + "] = " + std::to_string(varShape.GetDim(i)),
                            "grad must have the same shape as var (except dim 0)"),
                        return ge::GRAPH_FAILED);
        }
    }

    shapeInfo.gradDim0 = gradShape.GetDim(0);
    shapeInfo.indicesDim0 = indicesShape.GetDim(0);
    OP_CHECK_IF(shapeInfo.gradDim0 <= 0 || shapeInfo.indicesDim0 <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context->GetNodeName(), "grad and indices",
                    std::to_string(shapeInfo.gradDim0) + " and " + std::to_string(shapeInfo.indicesDim0),
                    "grad dim0 and indices length must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shapeInfo.gradDim0 != shapeInfo.indicesDim0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context->GetNodeName(), "grad and indices",
                    std::to_string(shapeInfo.gradDim0) + " and " + std::to_string(shapeInfo.indicesDim0),
                    "grad dim0 must be equal to the length of indices"),
                return ge::GRAPH_FAILED);

    shapeInfo.firstDim = varShape.GetDim(0);
    shapeInfo.rowSize = 1;
    for (size_t i = 1; i < varShape.GetDimNum(); ++i) {
        shapeInfo.rowSize *= varShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildBaseTiling(gert::TilingContext* context, SparseApplyAdagradBaseTiling& base)
{
    OP_CHECK_IF(GetPlatformInfo(context, base.ubSize, base.coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "GetPlatformInfo failed; propagate the lower-layer first error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CheckDtype failed; propagate the lower-layer first error."),
        return ge::GRAPH_FAILED);

    SparseApplyAdagradShapeInfo shapeInfo;
    OP_CHECK_IF(
        CheckShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CheckShape failed; propagate the lower-layer first error."),
        return ge::GRAPH_FAILED);

    base.firstDim = shapeInfo.firstDim;
    base.rowSize = shapeInfo.rowSize;
    base.gradDim0 = shapeInfo.gradDim0;
    base.indicesDim0 = shapeInfo.indicesDim0;
    base.useLocking = GetBoolAttr(context, ATTR_USE_LOCKING, false);
    base.updateSlots = GetBoolAttr(context, ATTR_UPDATE_SLOTS, true);

    int64_t rowBytes = base.rowSize * FLOAT32_BYTES;
    base.blockFormer = AlignUp((rowBytes + base.coreNum - 1) / base.coreNum, REG_LENGTH);
    if (base.blockFormer <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockFormer",
                                              std::to_string(base.blockFormer).c_str(),
                                              "blockFormer must be positive.");
        return ge::GRAPH_FAILED;
    }
    base.blockNum = (rowBytes + base.blockFormer - 1) / base.blockFormer;
    base.blockTail = rowBytes - (base.blockNum - 1) * base.blockFormer;
    base.indicesDim0Align = AlignUp(base.indicesDim0, BLOCK_NUM);
    int64_t ubFactorNumerator = base.ubSize - UB_RESERVED_BYTE - base.indicesDim0Align * FLOAT32_BYTES;
    base.ubFactor = ubFactorNumerator / (base.blockFormer * FLOAT32_BYTES);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillOptimizeTiling(gert::TilingContext* context, const SparseApplyAdagradBaseTiling& base)
{
    int64_t loopNumTailBlock = base.gradDim0 / base.coreNum;
    int64_t loopNumFormerBlock = loopNumTailBlock + 1;
    int64_t formerCoreNum = base.gradDim0 % base.coreNum;
    int64_t blockNum = loopNumTailBlock == 0 ? formerCoreNum : base.coreNum;
    if (blockNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockNum", std::to_string(blockNum).c_str(),
                                              "optimize blockNum must be positive.");
        return ge::GRAPH_FAILED;
    }

    int64_t loopNumFormerBlockAlign = (loopNumFormerBlock + BLOCK_NUM - 1) / BLOCK_NUM;
    if (loopNumFormerBlock > MAX_INDICES_BLOCK) {
        loopNumFormerBlockAlign = MAX_INDICES_BLOCK;
    }
    int64_t availableUB = base.ubSize - UB_RESERVED_BYTE - loopNumFormerBlockAlign * FLOAT32_BYTES - BLOCK_LENGTH;
    int64_t ubFormer = (availableUB / 10) / REG_LENGTH * REG_LENGTH;
    if (ubFormer <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubFormer", std::to_string(ubFormer).c_str(),
                                              "optimize ubFormer must be positive.");
        return ge::GRAPH_FAILED;
    }

    int64_t rowBytes = base.rowSize * FLOAT32_BYTES;
    int64_t ubLoop = 1;
    int64_t ubTailFormer = ubFormer;
    if (ubFormer < rowBytes) {
        ubLoop += rowBytes / ubFormer;
        ubTailFormer = rowBytes - (ubLoop - 1) * ubFormer;
    } else {
        ubFormer = rowBytes;
        ubTailFormer = rowBytes;
    }

    auto tiling = context->GetTilingData<SparseApplyAdagradOptimizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalCoreNum = base.coreNum;
    tiling->blockNum = blockNum;
    tiling->a1_ = base.firstDim;
    tiling->r_ = base.rowSize;
    tiling->gradDim0 = base.gradDim0;
    tiling->loopNumTailBlock = loopNumTailBlock;
    tiling->loopNumFormerBlock = loopNumFormerBlock;
    tiling->formerCoreNum = formerCoreNum;
    tiling->ubFormer = ubFormer;
    tiling->ubLoop = ubLoop;
    tiling->ubTailFormer = ubTailFormer;
    tiling->useLocking = base.useLocking;
    tiling->updateSlots = base.updateSlots;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    context->SetTilingKey(SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillOptimizeSplitRTiling(gert::TilingContext* context, const SparseApplyAdagradBaseTiling& base)
{
    int64_t splitblockNum = base.coreNum / base.gradDim0;
    if (splitblockNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "splitblockNum",
                                              std::to_string(splitblockNum).c_str(), "splitblockNum must be positive.");
        return ge::GRAPH_FAILED;
    }
    int64_t formerBlockDataNum = base.rowSize / splitblockNum;
    if (formerBlockDataNum < REG_FLOAT_NUM) {
        formerBlockDataNum = REG_FLOAT_NUM;
    } else {
        formerBlockDataNum = AlignUp(formerBlockDataNum, REG_FLOAT_NUM);
    }
    splitblockNum = (base.rowSize + formerBlockDataNum - 1) / formerBlockDataNum;
    int64_t tailBlockDataNum = base.rowSize - formerBlockDataNum * (splitblockNum - 1);
    int64_t blockNum = splitblockNum * base.gradDim0;

    int64_t availableUB = base.ubSize - UB_RESERVED_BYTE - BLOCK_LENGTH;
    int64_t ubFormer = availableUB / CONST_FIVE / FLOAT32_BYTES / REG_FLOAT_NUM * REG_FLOAT_NUM;
    if (ubFormer <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubFormer", std::to_string(ubFormer).c_str(),
                                              "split-r ubFormer must be positive.");
        return ge::GRAPH_FAILED;
    }
    int64_t formerBlockUbLoop = 1;
    int64_t formerBlocktailNum = formerBlockDataNum;
    int64_t tailBlockUbLoop = 1;
    int64_t tailBlocktailNum = tailBlockDataNum;
    if (ubFormer < formerBlockDataNum) {
        formerBlockUbLoop = (formerBlockDataNum + ubFormer - 1) / ubFormer;
        formerBlocktailNum = formerBlockDataNum - (formerBlockUbLoop - 1) * ubFormer;
        tailBlockUbLoop = (tailBlockDataNum + ubFormer - 1) / ubFormer;
        tailBlocktailNum = tailBlockDataNum - (tailBlockUbLoop - 1) * ubFormer;
    }

    auto tiling = context->GetTilingData<SparseApplyAdagradOptimizeSplitRTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalCoreNum = base.coreNum;
    tiling->blockNum = blockNum;
    tiling->a1_ = base.firstDim;
    tiling->r_ = base.rowSize;
    tiling->gradDim0 = base.gradDim0;
    tiling->splitblockNum = splitblockNum;
    tiling->tailBlockDataNum = tailBlockDataNum;
    tiling->formerBlockDataNum = formerBlockDataNum;
    tiling->ubFormer = ubFormer;
    tiling->formerBlockUbLoop = formerBlockUbLoop;
    tiling->tailBlockUbLoop = tailBlockUbLoop;
    tiling->formerBlocktailNum = formerBlocktailNum;
    tiling->tailBlocktailNum = tailBlocktailNum;
    tiling->useLocking = base.useLocking;
    tiling->updateSlots = base.updateSlots;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    context->SetTilingKey(SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE_SPLIT_R);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillSmallTiling(gert::TilingContext* context, const SparseApplyAdagradBaseTiling& base)
{
    auto tiling = context->GetTilingData<SparseApplyAdagradSmallDimTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalCoreNum = base.coreNum;
    tiling->a1_ = base.firstDim;
    tiling->r_ = base.rowSize;
    tiling->gradDim0 = base.gradDim0;
    tiling->blockFormer = base.blockFormer;
    tiling->blockNum = base.blockNum;
    tiling->blockTail = base.blockTail;
    tiling->useLocking = base.useLocking;
    tiling->updateSlots = base.updateSlots;
    context->SetBlockDim(static_cast<uint32_t>(base.blockNum));
    context->SetTilingKey(SPARSE_APPLY_ADAGRAD_TILING_KEY_SMALL);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillBigTiling(gert::TilingContext* context, SparseApplyAdagradBaseTiling base)
{
    int64_t indicesLoopNum = 1;
    int64_t indicesTailBlock = base.indicesDim0;
    if (base.indicesDim0Align > MAX_INDICES_BLOCK) {
        indicesLoopNum += base.indicesDim0Align / MAX_INDICES_BLOCK;
        indicesTailBlock = base.indicesDim0Align - (indicesLoopNum - 1) * MAX_INDICES_BLOCK;
        base.indicesDim0Align = MAX_INDICES_BLOCK;
    }
    int64_t availableUB = base.ubSize - UB_RESERVED_BYTE - base.indicesDim0Align * FLOAT32_BYTES - BLOCK_LENGTH;
    int64_t ubFormer = (availableUB / BLOCK_NUM) / REG_LENGTH * REG_LENGTH;
    if (ubFormer <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubFormer", std::to_string(ubFormer).c_str(),
                                              "big ubFormer must be positive.");
        return ge::GRAPH_FAILED;
    }
    ubFormer = std::min(ubFormer, base.blockFormer);
    int64_t ubLoopOfFormerBlock = (base.blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubLoopOfTailBlock = (base.blockTail + ubFormer - 1) / ubFormer;
    int64_t ubTailOfFormerBlock = base.blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubTailOfTailBlock = base.blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    auto tiling = context->GetTilingData<SparseApplyAdagradBigDimTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalCoreNum = base.coreNum;
    tiling->a1_ = base.firstDim;
    tiling->r_ = base.rowSize;
    tiling->gradDim0 = base.gradDim0;
    tiling->blockFormer = base.blockFormer;
    tiling->blockNum = base.blockNum;
    tiling->blockTail = base.blockTail;
    tiling->ubFormer = ubFormer;
    tiling->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling->ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling->ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling->ubTailOfTailBlock = ubTailOfTailBlock;
    tiling->indicesLoopNum = indicesLoopNum;
    tiling->indicesTailBlock = indicesTailBlock;
    tiling->useLocking = base.useLocking;
    tiling->updateSlots = base.updateSlots;
    context->SetBlockDim(static_cast<uint32_t>(base.blockNum));
    context->SetTilingKey(SPARSE_APPLY_ADAGRAD_TILING_KEY_BIG);
    return ge::GRAPH_SUCCESS;
}

static bool IsOptimizeSplitRCapable(const SparseApplyAdagradBaseTiling& base)
{
    return !base.useLocking && base.gradDim0 <= (base.coreNum / 2) && base.rowSize >= REG_LENGTH;
}

static ge::graphStatus SparseApplyAdagradTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter TilingSparseApplyAdagrad");

    SparseApplyAdagradBaseTiling base;
    OP_CHECK_IF(BuildBaseTiling(context, base) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "BuildBaseTiling failed; propagate the lower-layer first error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SetWorkspace failed; propagate the lower-layer first error."),
        return ge::GRAPH_FAILED);

    if (IsOptimizeSplitRCapable(base)) {
        return FillOptimizeSplitRTiling(context, base);
    }
    if (!base.useLocking) {
        return FillOptimizeTiling(context, base);
    }
    if (base.ubFactor > 0) {
        return FillSmallTiling(context, base);
    }
    return FillBigTiling(context, base);
}

IMPL_OP_OPTILING(SparseApplyAdagrad)
    .Tiling(SparseApplyAdagradTilingFunc)
    .TilingParse<SparseApplyAdagradCompileInfo>([](gert::TilingParseContext* context) {
        (void)context;
        return ge::GRAPH_SUCCESS;
    });

} // namespace optiling
