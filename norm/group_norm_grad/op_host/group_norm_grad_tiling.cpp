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
 * \file group_norm_grad.cc
 * \brief
 */
#include "group_norm_grad_tiling_arch35.h"
#include "group_norm_grad_tiling.h"

namespace {
constexpr int64_t FP32_MODE = 0;
constexpr int64_t FP16_MODE = 1;
constexpr int64_t BF16_MODE = 2;
constexpr int64_t MODE_0 = 0;
constexpr int64_t MODE_1 = 1;
constexpr int64_t MODE_2 = 2;
constexpr int64_t MODE_3 = 3;
constexpr int64_t MODE_4 = 4;
constexpr int64_t MODE_5 = 5;
constexpr int64_t INPUT_0 = 0;
constexpr int64_t INPUT_1 = 1;
constexpr int64_t INPUT_2 = 2;
constexpr int64_t INPUT_3 = 3;
constexpr int64_t INPUT_4 = 4;
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIM3 = 3;
constexpr int64_t DX_IS_REQUIRE_IDX = 2;
constexpr int64_t DGAMMA_IS_REQUIRE_IDX = 3;
constexpr int64_t DBETA_IS_REQUIRE_IDX = 4;
constexpr int64_t UPPER_CARRYING_LIMIT = 8000;
constexpr int64_t MAX_C_SIZE = 240000;
constexpr uint64_t WORKSPACE_REVERSE = 16 * 1024 * 1024;
constexpr uint64_t TEN = 10;
constexpr uint32_t UB_COPIES_4 = 4;
constexpr uint32_t UB_COPIES_1 = 3;
constexpr uint32_t UB_COPIES_2 = 2;
constexpr uint32_t WORKSPACE_COPIES = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t RESERVE_SAPCE = 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
constexpr uint32_t BFLOAT16_DTYPE_BYTES = 2;
constexpr uint32_t FLOAT16_DTYPE_BYTES = 2;
constexpr uint32_t EIGHT_BLOCK = 8;
constexpr uint32_t SPLIT_COUNT = 2;
constexpr uint32_t STEP_SIZE = 64;
} // namespace

namespace optiling {

class GroupNormGradTiling {
public:
    explicit GroupNormGradTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    ge::graphStatus SetTilingKeyMode(ge::DataType dtypeStr, uint32_t isDeterministicKey) const;
    ge::graphStatus ComputeAllocUBStage2(uint32_t coreBatchCounts, uint32_t availableSpace);
    uint32_t GetDataTypeSize(ge::DataType dtypeStr) const;
    uint32_t GetElePerBlock(uint32_t dtypeBytes) const;
    uint32_t Ceil(uint32_t a, uint32_t b) const;
    uint32_t DivCeil(uint32_t a, uint32_t b) const;
    uint32_t Floor(uint32_t a, uint32_t b) const;
    ge::graphStatus CalStage2TilingInfo(
        uint64_t UB_size, ge::DataType dtypeStr, uint32_t isDeterministicKey, size_t sysWorkspaceSize);
    ge::graphStatus CalStage1TilingInfo(uint32_t reserveSpace);
    bool CheckCompileInfo();
    bool CheckInputDtype();
    bool CheckInputShape();
    bool PlanStepCoreUsage();
    GroupNormGradTilingData tilingData;
    const GroupNormGradCompileInfo* compileInfo = nullptr;
    ge::DataType dtypeStr;
    uint32_t dtypeBytes = 0;
    uint32_t elePerBlock = 0;
    std::unique_ptr<TilingCalculationParameters> tilingParams;
    gert::TilingContext* tilingContext = nullptr;
    const gert::RuntimeAttrs* attrs = nullptr;
};

bool GroupNormGradTiling::CheckCompileInfo()
{
    OP_TILING_CHECK(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(tilingContext->GetNodeName(), "core num should be greater than zero."), return false);
    return true;
}

bool GroupNormGradTiling::CheckInputDtype()
{
    OP_TILING_CHECK(
        (tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetInputDesc(INPUT_1) == nullptr ||
         tilingContext->GetInputDesc(INPUT_2) == nullptr || tilingContext->GetInputDesc(INPUT_3) == nullptr ||
         tilingContext->GetInputDesc(INPUT_4) == nullptr),
        OP_LOGE(
            tilingContext->GetNodeName(),
            "tilingContext->GetInputDesc(INPUT_0) is nullptr   \
    or tilingContext->GetInputDesc(INPUT_1) is nullptr \
    or tilingContext->GetInputDesc(INPUT_2) is nullptr \
    or tilingContext->GetInputDesc(INPUT_3) is nullptr \
    or tilingContext->GetInputDesc(INPUT_4) is nullptr "),
        return false);
    auto inputDesc = tilingContext->GetInputDesc(INPUT_0);
    this->dtypeStr = inputDesc->GetDataType();
    OP_TILING_CHECK(
        !(this->dtypeStr == tilingContext->GetInputDesc(INPUT_1)->GetDataType() &&
          this->dtypeStr == tilingContext->GetInputDesc(INPUT_2)->GetDataType() &&
          this->dtypeStr == tilingContext->GetInputDesc(INPUT_3)->GetDataType() &&
          this->dtypeStr == tilingContext->GetInputDesc(INPUT_4)->GetDataType()),
        OP_LOGE(tilingContext->GetNodeName(), "Input dtypeBytes is not same"), return false);
    this->dtypeBytes = GetDataTypeSize(this->dtypeStr);
    OP_TILING_CHECK((this->dtypeBytes == 0), OP_LOGE(tilingContext->GetNodeName(), "dtypeBytes is zero"), return false);
    // Calculate elePerBlock based on the value of dtypeBytes
    this->elePerBlock = GetElePerBlock(this->dtypeBytes);
    OP_TILING_CHECK(
        (this->elePerBlock == 0), OP_LOGE(tilingContext->GetNodeName(), "Calculated elePerBlock is zero"),
        return false);
    return true;
}

bool GroupNormGradTiling::CheckInputShape()
{
    auto InputShape0 = tilingContext->GetInputShape(INPUT_0);
    OP_TILING_CHECK(
        InputShape0 == nullptr,
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext->GetInputDesc(INPUT_0) is nullptr"), return false);
    auto InputShape1 = tilingContext->GetInputShape(INPUT_1);
    OP_TILING_CHECK(
        InputShape1 == nullptr,
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext->GetInputDesc(INPUT_1) is nullptr"), return false);
    auto InputShape2 = tilingContext->GetInputShape(INPUT_2);
    OP_TILING_CHECK(
        InputShape2 == nullptr,
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext->GetInputDesc(INPUT_2) is nullptr"), return false);
    auto InputShape3 = tilingContext->GetInputShape(INPUT_3);
    OP_TILING_CHECK(
        InputShape3 == nullptr,
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext->GetInputDesc(INPUT_3) is nullptr"), return false);
    auto InputShape4 = tilingContext->GetInputShape(INPUT_4);
    OP_TILING_CHECK(
        InputShape4 == nullptr,
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext->GetInputDesc(INPUT_4) is nullptr"), return false);
    const gert::Shape xShape = tilingContext->GetInputShape(INPUT_3)->GetStorageShape();
    const gert::Shape gammaShape = tilingContext->GetInputShape(INPUT_4)->GetStorageShape();
    const gert::Shape dyShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    const gert::Shape meanShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    const gert::Shape rstdShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto dimNum = dyShape.GetDimNum();
    OP_TILING_CHECK(
        dyShape != xShape,
        OP_LOGE(tilingContext->GetNodeName(), "Input shape check Failed, dy shape should be same with x shape"),
        return false);
    attrs = tilingContext->GetAttrs();
    OP_TILING_CHECK((attrs == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get attrs Failed."), return false);
    tilingParams->g = *(attrs->GetAttrPointer<int32_t>(0));
    OP_TILING_CHECK(
        meanShape.GetDim(DIM1) != tilingParams->g,
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, group_num shuold be same with mean.shape[1]"),
        return false);
    tilingParams->n = dyShape.GetDim(DIM0);
    tilingParams->c = dyShape.GetDim(DIM1);
    OP_TILING_CHECK(
        meanShape.GetDim(DIM0) != tilingParams->n,
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, mean.shape[0] should be same with N(x.shape[0])"),
        return false);
    OP_TILING_CHECK(
        meanShape != rstdShape,
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, Shape of mean and rstd not same."), return false);
    OP_TILING_CHECK(
        (meanShape.GetDimNum() != 2 || rstdShape.GetDimNum() != 2),
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, Dim of mean and rstd should be 2."), return false);
    OP_TILING_CHECK(
        (gammaShape.GetDimNum() != 1 || gammaShape.GetDim(DIM0) != tilingParams->c),
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, Shape of gamma should be (C,)."), return false);
    tilingParams->nxg = tilingParams->n * tilingParams->g;
    OP_TILING_CHECK(
        tilingParams->nxg == 0, OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, N x G should not be 0."),
        return false);
    tilingParams->channelPerGroup = tilingParams->c / tilingParams->g;
    // C / G must not be zero, and C must be an integer multiple of G
    OP_TILING_CHECK(
        (tilingParams->channelPerGroup == 0 || tilingParams->c % tilingParams->g != 0),
        OP_LOGE(tilingContext->GetNodeName(), "Group_num or Channel num is invalid"), return false);
    OP_TILING_CHECK(
        (tilingParams->c > MAX_C_SIZE),
        OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, C should not exceed 240000."), return false);
    tilingParams->hxw = 1;
    for (uint32_t dimIdx = 2; dimIdx < dimNum; dimIdx++) {
        tilingParams->hxw *= dyShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(
        tilingParams->hxw == 0, OP_LOGE(tilingContext->GetNodeName(), "Check shape failed, HxW should not be 0."),
        return false);
    return true;
}

ge::graphStatus GroupNormGradTiling::ComputeAllocUBStage2(uint32_t coreBatchCounts, uint32_t availableSpace)
{
    OP_TILING_CHECK(
        tilingParams->castEleNum == 0,
        OP_LOGE(tilingContext->GetNodeName(), "Error:[ComputeAllocUBStage2] castEleNum is zero!"),
        return ge::GRAPH_FAILED);
    tilingParams->coreBatchParts =
        std::min(availableSpace / (tilingParams->castEleNum * FLOAT_DTYPE_BYTES), coreBatchCounts);
    OP_TILING_CHECK(
        tilingParams->coreBatchParts == 0,
        OP_LOGE(tilingContext->GetNodeName(), "Error:[ComputeAllocUBStage2] coreBatchCounts is zero!"),
        return ge::GRAPH_FAILED);
    tilingParams->coreBatchPartsTailRepeat = (coreBatchCounts % tilingParams->coreBatchParts == 0) ?
                                                 tilingParams->coreBatchParts :
                                                 coreBatchCounts % tilingParams->coreBatchParts;
    tilingParams->repeatTime4Stage2 = DivCeil(coreBatchCounts, tilingParams->coreBatchParts);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTiling::CalStage2TilingInfo(
    uint64_t UB_size, ge::DataType dtypeStrLocal, uint32_t isDeterministicKey, size_t sysWorkspaceSize)
{
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    size_t availableSpace = (UB_size - RESERVE_SAPCE) / SPLIT_COUNT;
    OP_TILING_CHECK(
        (currentWorkSpace == nullptr), OP_LOGE(tilingContext->GetNodeName(), "currentWorkSpace is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t coreBatchCounts = 0;
    size_t usrWorkspaceSize = 0;
    if (isDeterministicKey == 1) {
        tilingParams->workSpaceSize = DivCeil(tilingParams->n, SPLIT_COUNT) * tilingParams->c;
        usrWorkspaceSize = WORKSPACE_COPIES * static_cast<int64_t>(tilingParams->workSpaceSize) * FLOAT_DTYPE_BYTES;
        if (dtypeStrLocal == ge::DT_FLOAT) {
            // task ReduceSum
            tilingParams->castEleNum =
                Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed / SPLIT_COUNT), STEP_SIZE);
            tilingParams->stage2CoreUsed = DivCeil(tilingParams->c, tilingParams->castEleNum);
            tilingParams->tailCastNum =
                (tilingParams->stage2CoreUsed == 1) ?
                    tilingParams->c :
                    tilingParams->c - (tilingParams->stage2CoreUsed - 1) * tilingParams->castEleNum;
            // UB allocation
            availableSpace = availableSpace > tilingParams->castEleNum * FLOAT_DTYPE_BYTES ?
                                 availableSpace - tilingParams->castEleNum * FLOAT_DTYPE_BYTES :
                                 0;
            coreBatchCounts = DivCeil(DivCeil(tilingParams->n, SPLIT_COUNT), SPLIT_COUNT);
            OP_TILING_CHECK(
                ComputeAllocUBStage2(coreBatchCounts, availableSpace) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "fail to Alloc UB for Stage2"), return ge::GRAPH_FAILED);
        } else {
            // task ReduceSum and Cast
            tilingParams->castEleNum = Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed), STEP_SIZE);
            tilingParams->stage2CoreUsed = DivCeil(tilingParams->c, tilingParams->castEleNum);
            tilingParams->tailCastNum =
                (tilingParams->stage2CoreUsed == 1) ?
                    tilingParams->c :
                    tilingParams->c - (tilingParams->stage2CoreUsed - 1) * tilingParams->castEleNum;
            // UB allocation
            availableSpace = availableSpace - tilingParams->castEleNum * (FLOAT_DTYPE_BYTES + FLOAT16_DTYPE_BYTES);
            coreBatchCounts = DivCeil(tilingParams->n, SPLIT_COUNT);
            OP_TILING_CHECK(
                ComputeAllocUBStage2(coreBatchCounts, availableSpace) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "fail to Alloc UB for Stage2"), return ge::GRAPH_FAILED);
        }
    } else {
        if (dtypeStrLocal == ge::DT_FLOAT) {
            // no stage2 task
            usrWorkspaceSize = 0;
        } else {
            // task Cast
            tilingParams->castEleNum = Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed), STEP_SIZE);
            tilingParams->workSpaceSize = tilingParams->c;
            usrWorkspaceSize = WORKSPACE_COPIES * static_cast<int64_t>(tilingParams->workSpaceSize) * FLOAT_DTYPE_BYTES;
            tilingParams->stage2CoreUsed = DivCeil(tilingParams->c, tilingParams->castEleNum);
            tilingParams->tailCastNum =
                (tilingParams->stage2CoreUsed == 1) ?
                    tilingParams->c :
                    tilingParams->c - (tilingParams->stage2CoreUsed - 1) * tilingParams->castEleNum;
        }
    }
    currentWorkSpace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTiling::CalStage1TilingInfo(uint32_t reserveSpace)
{
    uint32_t unalignedExtraSpace = (tilingParams->hxw % EIGHT_BLOCK == 0 || tilingParams->channelPerGroup == 1) ?
                                       0 :
                                       UB_COPIES_2 * Ceil(tilingParams->hxw, elePerBlock) * FLOAT_DTYPE_BYTES;
    // Prevent wrapping errors in uint32 subtraction
    tilingParams->mode0UbCapGNum =
        (compileInfo->ubSizePlatForm < reserveSpace + unalignedExtraSpace) ?
            0 :
            (compileInfo->ubSizePlatForm - reserveSpace - unalignedExtraSpace) /
                (Ceil(tilingParams->channelPerGroup * tilingParams->hxw, elePerBlock) * dtypeBytes * UB_COPIES_1);
    tilingParams->mode1UbCapCNum = (compileInfo->ubSizePlatForm - reserveSpace) /
                                   (Ceil(tilingParams->hxw, elePerBlock) * dtypeBytes * UB_COPIES_1);
    if (tilingParams->mode1UbCapCNum > 0) {
        tilingParams->mode1UbIterCNum =
            Ceil(tilingParams->channelPerGroup, tilingParams->mode1UbCapCNum) / tilingParams->mode1UbCapCNum;
        tilingParams->mode1UbTailCNum =
            (tilingParams->mode1UbIterCNum * tilingParams->mode1UbCapCNum - tilingParams->channelPerGroup == 0) ?
                tilingParams->mode1UbCapCNum :
                (tilingParams->channelPerGroup - ((tilingParams->mode1UbIterCNum - 1) * tilingParams->mode1UbCapCNum));
    }
    if (tilingParams->hxw == 1) {
        tilingParams->tilingKey = MODE_5;
    } else if (tilingParams->mode0UbCapGNum > 0) {
        tilingParams->tilingKey = MODE_0;
    } else if (tilingParams->mode0UbCapGNum <= 0 && tilingParams->mode1UbCapCNum == 1) {
        tilingParams->tilingKey = MODE_1;
    } else if (tilingParams->mode1UbCapCNum > 1) {
        tilingParams->tilingKey = MODE_2;
    } else if (tilingParams->mode1UbCapCNum <= 0) {
        tilingParams->tilingKey = MODE_3;
        tilingParams->mode2UbCapacityEle =
            Floor((compileInfo->ubSizePlatForm - reserveSpace) / (dtypeBytes * UB_COPIES_1), elePerBlock * EIGHT_BLOCK);
        OP_TILING_CHECK(
            tilingParams->mode2UbCapacityEle == 0,
            OP_LOGE(tilingContext->GetNodeName(), "tilingParams->mode2UbCapacityEle should not be zero!"),
            return ge::GRAPH_FAILED);
        tilingParams->mode2UbIterationNum =
            Ceil(tilingParams->hxw, tilingParams->mode2UbCapacityEle) / tilingParams->mode2UbCapacityEle;
        tilingParams->mode2UbTailNum =
            (tilingParams->hxw - tilingParams->mode2UbIterationNum * tilingParams->mode2UbCapacityEle == 0) ?
                tilingParams->mode2UbCapacityEle :
                (tilingParams->hxw - (tilingParams->mode2UbIterationNum - 1) * tilingParams->mode2UbCapacityEle);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTiling::SetTilingKeyMode(ge::DataType dtypeStrLocal, uint32_t isDeterministicKey) const
{
    switch (dtypeStrLocal) {
        case ge::DT_BF16:
            tilingContext->SetTilingKey(BF16_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(FP32_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(tilingContext->GetNodeName(), "inputdtype must be in [float32, float16, bfloat16]");
            return ge::GRAPH_FAILED;
    }
}

uint32_t GroupNormGradTiling::GetDataTypeSize(ge::DataType dtypeStrLocal) const
{
    switch (dtypeStrLocal) {
        case ge::DT_FLOAT:
            return FLOAT_DTYPE_BYTES;
        case ge::DT_BF16:
            return BFLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES;
        case ge::DT_FLOAT16:
            return FLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES;
        default:
            OP_LOGE(tilingContext->GetNodeName(), "inputdtype must be in [float32, float16, bfloat16]");
            return 0;
    }
}

uint32_t GroupNormGradTiling::GetElePerBlock(uint32_t dtypeBytesLocal) const
{
    switch (dtypeBytesLocal) {
        case FLOAT_DTYPE_BYTES:
            return BLOCK_BYTES / FLOAT_DTYPE_BYTES;
        case FLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES:
            return BLOCK_BYTES / FLOAT16_DTYPE_BYTES;
        default:
            return 0;
    }
}

uint32_t GroupNormGradTiling::Ceil(uint32_t a, uint32_t b) const
{
    OP_TILING_CHECK(b == 0, OP_LOGE(tilingContext->GetNodeName(), "Error:[Ceil] Division by zero!"), return 0);
    return ((a - 1) / b + 1) * b;
}

uint32_t GroupNormGradTiling::DivCeil(uint32_t a, uint32_t b) const
{
    OP_TILING_CHECK(b == 0, OP_LOGE(tilingContext->GetNodeName(), "Error:[DivCeil] Division by zero!"), return 0);
    return (a - 1) / b + 1;
}

uint32_t GroupNormGradTiling::Floor(uint32_t a, uint32_t b) const
{
    OP_TILING_CHECK(b == 0, OP_LOGE(tilingContext->GetNodeName(), "Error:[Floor] Division by zero!"), return 0);
    return a / b * b;
}

bool GroupNormGradTiling::PlanStepCoreUsage()
{
    tilingParams->taskNumPerCore = Ceil(tilingParams->nxg, compileInfo->totalCoreNum) / compileInfo->totalCoreNum;
    tilingParams->coreNumUsed = (tilingParams->nxg - 1) / tilingParams->taskNumPerCore + 1;
    OP_TILING_CHECK(
        tilingParams->coreNumUsed == 0, OP_LOGE(tilingContext->GetNodeName(), "coreNumUsed cannot be 0."),
        return false);
    tilingParams->taskNumPerTailCore = tilingParams->taskNumPerCore;
    tilingParams->tailCore = tilingParams->coreNumUsed;
    if (tilingParams->nxg % tilingParams->coreNumUsed != 0) {
        tilingParams->taskNumPerTailCore = tilingParams->taskNumPerCore - 1;
        tilingParams->tailCore = tilingParams->nxg % tilingParams->coreNumUsed;
    }
    return true;
}

ge::graphStatus GroupNormGradTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling initing.");
    try {
        tilingParams = std::make_unique<TilingCalculationParameters>();
    } catch (const std::bad_alloc& e) {
        OP_LOGE(tilingContext->GetNodeName(), "tilingParams memory allocation failed.");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        tilingParams == nullptr, OP_LOGE(tilingContext->GetNodeName(), "failed to instantiate tilingParams"),
        return ge::GRAPH_FAILED);
    // Get compileInfo
    compileInfo = static_cast<const GroupNormGradCompileInfo*>(tilingContext->GetCompileInfo());
    OP_TILING_CHECK(
        !CheckCompileInfo(), OP_LOGE(tilingContext->GetNodeName(), "CompileInfo Check Failed."),
        return ge::GRAPH_FAILED);
    // Get Inputs dtype
    OP_TILING_CHECK(
        !CheckInputDtype(), OP_LOGE(tilingContext->GetNodeName(), "InputDtype Check Failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckInputShape(), OP_LOGE(tilingContext->GetNodeName(), "InputShape Check Failed."), return ge::GRAPH_FAILED);
    // Allocate computing core
    uint32_t channelPerGroupOnceProcess = Ceil(tilingParams->channelPerGroup, elePerBlock);
    // Check channelPerGroup not exceeding the operator's current carrying capacity.
    OP_TILING_CHECK(
        (channelPerGroupOnceProcess > UPPER_CARRYING_LIMIT),
        OP_LOGE(
            tilingContext->GetNodeName(), "channelPerGroup is %u over the operator's current carrying capacity %ld.",
            tilingParams->channelPerGroup, UPPER_CARRYING_LIMIT),
        return ge::GRAPH_FAILED);
    uint32_t reserveSpace = RESERVE_SAPCE + channelPerGroupOnceProcess * this->dtypeBytes * UB_COPIES_1;
    // The function PlanStepCoreUsage is to plan the usage of cores for each step.
    OP_TILING_CHECK(
        !PlanStepCoreUsage(), OP_LOGE(tilingContext->GetNodeName(), "PlanStepCoreUsage failed."),
        return ge::GRAPH_FAILED);
    // Select UB allocation mode for stage_1
    OP_TILING_CHECK(
        CalStage1TilingInfo(reserveSpace) != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext->GetNodeName(), "fail to calculate Stage1 TilingInfo"), return ge::GRAPH_FAILED);
    // Because the accumulation axis is the N axis, there is no deterministic problem when N<=2
    uint32_t isDeterministicKey = (tilingParams->n > 2) ? 1 : 0;
    OP_TILING_CHECK(
        !(isDeterministicKey == 0 || isDeterministicKey == 1),
        OP_LOGE(tilingContext->GetNodeName(), "Error: isDeterministicKey must be 0 or 1!"), return ge::GRAPH_FAILED);
    // Set TilingKey mode
    OP_TILING_CHECK(
        SetTilingKeyMode(this->dtypeStr, isDeterministicKey) != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext->GetNodeName(), "fail to Set TilingKey"), return ge::GRAPH_FAILED);
    // Calculate workspace space
    OP_TILING_CHECK(
        CalStage2TilingInfo(
            compileInfo->ubSizePlatForm, this->dtypeStr, isDeterministicKey, compileInfo->sysWorkspaceSize) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext->GetNodeName(), "fail to calculate Stage2 TilingInfo"), return ge::GRAPH_FAILED);
    // Get ATTR
    tilingParams->dxIsRequire = *(attrs->GetAttrPointer<bool>(DX_IS_REQUIRE_IDX));
    tilingParams->dgammaIsRequire = *(attrs->GetAttrPointer<bool>(DGAMMA_IS_REQUIRE_IDX));
    tilingParams->dbetaIsRequire = *(attrs->GetAttrPointer<bool>(DBETA_IS_REQUIRE_IDX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTiling::SetKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");
    tilingContext->SetBlockDim(tilingParams->coreNumUsed);
    tilingData.set_Tiling_key(tilingParams->tilingKey);                              // 0
    tilingData.set_N(tilingParams->n);                                               // 1
    tilingData.set_C(tilingParams->c);                                               // 2
    tilingData.set_HXW(tilingParams->hxw);                                           // 3
    tilingData.set_G(tilingParams->g);                                               // 4
    tilingData.set_NXG(tilingParams->nxg);                                           // 5
    tilingData.set_C_G(tilingParams->channelPerGroup);                               // 6
    tilingData.set_task_num_per_core(tilingParams->taskNumPerCore);                  // 7
    tilingData.set_task_num_per_tail_core(tilingParams->taskNumPerTailCore);         // 8
    tilingData.set_tail_core(tilingParams->tailCore);                                // 9
    tilingData.set_mode1_ub_cap_C_num(tilingParams->mode1UbCapCNum);                 // 10
    tilingData.set_mode1_ub_iter_C_num(tilingParams->mode1UbIterCNum);               // 11
    tilingData.set_mode1_ub_tail_C_num(tilingParams->mode1UbTailCNum);               // 12
    tilingData.set_mode2_ub_capacity_ele(tilingParams->mode2UbCapacityEle);          // 13
    tilingData.set_mode2_ub_iteration_num(tilingParams->mode2UbIterationNum);        // 14
    tilingData.set_mode2_ub_tail_num(tilingParams->mode2UbTailNum);                  // 15
    tilingData.set_workSpaceSize(tilingParams->workSpaceSize);                       // 16
    tilingData.set_stage2CoreUsed(tilingParams->stage2CoreUsed);                     // 17
    tilingData.set_castEleNum(tilingParams->castEleNum);                             // 18
    tilingData.set_tailCastNum(tilingParams->tailCastNum);                           // 19
    tilingData.set_coreBatchParts(tilingParams->coreBatchParts);                     // 20
    tilingData.set_coreBatchPartsTailRepeat(tilingParams->coreBatchPartsTailRepeat); // 21
    tilingData.set_repeatTime4Stage2(tilingParams->repeatTime4Stage2);               // 22
    tilingData.set_dx_is_require(tilingParams->dxIsRequire);                         // 23
    tilingData.set_dgamma_is_require(tilingParams->dgammaIsRequire);                 // 24
    tilingData.set_dbeta_is_require(tilingParams->dbetaIsRequire);                   // 25
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey:               %d.", tilingParams->tilingKey);
    OP_LOGD(tilingContext->GetNodeName(), "N:                       %d.", tilingParams->n);
    OP_LOGD(tilingContext->GetNodeName(), "C:                       %d.", tilingParams->c);
    OP_LOGD(tilingContext->GetNodeName(), "HXW:                     %d.", tilingParams->hxw);
    OP_LOGD(tilingContext->GetNodeName(), "G:                       %d.", tilingParams->g);
    OP_LOGD(tilingContext->GetNodeName(), "NXG:                     %d.", tilingParams->nxg);
    OP_LOGD(tilingContext->GetNodeName(), "channelPerGroup:         %d.", tilingParams->channelPerGroup);
    OP_LOGD(tilingContext->GetNodeName(), "taskNumPerCore:          %d.", tilingParams->taskNumPerCore);
    OP_LOGD(tilingContext->GetNodeName(), "taskNumPerTailCore:      %d.", tilingParams->taskNumPerTailCore);
    OP_LOGD(tilingContext->GetNodeName(), "tailCore:                %d.", tilingParams->tailCore);
    OP_LOGD(tilingContext->GetNodeName(), "mode1UbCapCNum:          %d.", tilingParams->mode1UbCapCNum);
    OP_LOGD(tilingContext->GetNodeName(), "mode1UbIterCNum:         %d.", tilingParams->mode1UbIterCNum);
    OP_LOGD(tilingContext->GetNodeName(), "mode1UbTailCNum:         %d.", tilingParams->mode1UbTailCNum);
    OP_LOGD(tilingContext->GetNodeName(), "mode2UbCapacityEle:      %d.", tilingParams->mode2UbCapacityEle);
    OP_LOGD(tilingContext->GetNodeName(), "mode2UbIterationNum:     %d.", tilingParams->mode2UbIterationNum);
    OP_LOGD(tilingContext->GetNodeName(), "mode2UbTailNum:          %d.", tilingParams->mode2UbTailNum);
    OP_LOGD(tilingContext->GetNodeName(), "workSpaceSize:           %d.", tilingParams->workSpaceSize);
    OP_LOGD(tilingContext->GetNodeName(), "stage2CoreUsed:          %d.", tilingParams->stage2CoreUsed);
    OP_LOGD(tilingContext->GetNodeName(), "castEleNum:              %d.", tilingParams->castEleNum);
    OP_LOGD(tilingContext->GetNodeName(), "tailCastNum:             %d.", tilingParams->tailCastNum);
    OP_LOGD(tilingContext->GetNodeName(), "coreBatchParts:          %d.", tilingParams->coreBatchParts);
    OP_LOGD(tilingContext->GetNodeName(), "coreBatchPartsTailRepeat:%d.", tilingParams->coreBatchPartsTailRepeat);
    OP_LOGD(tilingContext->GetNodeName(), "repeatTime4Stage2        %d.", tilingParams->repeatTime4Stage2);
    OP_LOGD(tilingContext->GetNodeName(), "dxIsRequire:             %d.", tilingParams->dxIsRequire);
    OP_LOGD(tilingContext->GetNodeName(), "dgammaIsRequire:         %d.", tilingParams->dgammaIsRequire);
    OP_LOGD(tilingContext->GetNodeName(), "dbetaIsRequire:          %d.", tilingParams->dbetaIsRequire);
}

ge::graphStatus TilingGroupNormGrad(gert::TilingContext* context)
{
    auto compileInfo = context->GetCompileInfo<GroupNormGradCompileInfo>();
    if (compileInfo->isRegBase) {
        GroupNormGradRegBaseTiling tilingObj(context);
        return tilingObj.DoTiling();
    }
    GroupNormGradTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.SetKernelTiling();
}

ge::graphStatus TilingPrepareForGroupNormGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForGroupNormGrad start.");
    auto compileInfo = context->GetCompiledInfo<GroupNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo->isRegBase =
        ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95 ? true : false;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_TILING_CHECK(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    compileInfo->vectorLen = Ops::Base::GetVRegSize(context);
    OP_TILING_CHECK(
        (compileInfo->vectorLen <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get vector Length failed, vector Length: %u",
            static_cast<uint32_t>(compileInfo->vectorLen)),
        return ge::GRAPH_FAILED);
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_TILING_CHECK(
        (compileInfo->blockSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get block Size failed, block size: %u",
            static_cast<uint32_t>(compileInfo->blockSize)),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForGroupNormGrad end.");
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(GroupNormGrad)
    .Tiling(TilingGroupNormGrad)
    .TilingParse<GroupNormGradCompileInfo>(TilingPrepareForGroupNormGrad);
} // namespace optiling