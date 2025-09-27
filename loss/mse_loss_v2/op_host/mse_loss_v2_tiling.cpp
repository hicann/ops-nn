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
 * \file mse_loss_v2_tiling.cpp
 * \brief
 */

#include "mse_loss_v2_tiling.h"

namespace optiling {

constexpr uint32_t BYTES_PER_CORE = 4096u; // 4 * 1024
constexpr uint32_t BYTES_PER_BLOCK = 32u;
constexpr uint32_t BLOCK_PER_CORE = BYTES_PER_CORE / BYTES_PER_BLOCK;
constexpr uint32_t FOUR_BYTES = 4u;
constexpr uint32_t TWO_BYTES = 2u;
constexpr const char* REDUCTION_MEAN = "mean";
constexpr const char* REDUCTION_SUM = "sum";
constexpr const char* REDUCTION_NONE = "none";

// tiling key
constexpr uint64_t TILING_KEY_DTYPE_FLOAT32 = 1ULL;
constexpr uint64_t TILING_KEY_DTYPE_FLOAT16 = 2ULL;
constexpr uint64_t TILING_KEY_DTYPE_BFLOAT16 = 3ULL;
constexpr uint64_t TILING_KEY_ATTR_MEAN = 30ULL;
constexpr uint64_t TILING_KEY_ATTR_SUM = 20ULL;
constexpr uint64_t TILING_KEY_ATTR_NONE = 10ULL;
constexpr uint64_t CAST_BUFFER_NUM = 4ULL;

constexpr uint32_t INPUT_DATA_IDX = 0u;
constexpr uint32_t INPUT_TARGET_IDX = 1u;
constexpr uint32_t ATTR_REDUCTION_IDX = 0u;

// variable stored globally
MSELossV2TilingData mseLossV2TilingData;
std::string reduction = optiling::REDUCTION_MEAN;
ge::DataType dtype = ge::DT_MAX;
uint32_t elemsPerBlock = 0u;
uint64_t tilingKey = 0ULL;
uint32_t kernelNumber = 0u;
uint64_t blockPerQue = 0ULL;
uint64_t blockPerCore = 0ULL;
uint64_t tailBlocks = 0ULL;

static inline uint64_t Mod(uint64_t a, uint64_t b)
{
    if (b == 0UL) {
        return a;
    }
    return b == 0UL ? 0UL : a % b;
}

static ge::graphStatus CheckInputDtype(const gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDtype = inputDesc->GetDataType();

    auto targetDesc = context->GetInputDesc(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    auto targetDtype = targetDesc->GetDataType();

    OP_CHECK_IF(
        inputDtype != targetDtype,
        OP_LOGE(context->GetNodeName(), "Input dtype and target dtype should be consistent."),
        return ge::GRAPH_FAILED);

    optiling::dtype = inputDtype;

    switch (inputDtype) {
        case ge::DT_FLOAT:
            optiling::elemsPerBlock = optiling::BYTES_PER_BLOCK / optiling::FOUR_BYTES;
            optiling::tilingKey += optiling::TILING_KEY_DTYPE_FLOAT32;
            break;
        case ge::DT_FLOAT16:
            optiling::elemsPerBlock = optiling::BYTES_PER_BLOCK / optiling::TWO_BYTES;
            optiling::tilingKey += optiling::TILING_KEY_DTYPE_FLOAT16;
            break;
        case ge::DT_BF16:
            optiling::elemsPerBlock = optiling::BYTES_PER_BLOCK / optiling::TWO_BYTES;
            optiling::tilingKey += optiling::TILING_KEY_DTYPE_BFLOAT16;
            break;

        default:
            OP_LOGE(
                context->GetNodeName(), "The input or target dtype must be one of float32, float16, or bfloat16.");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingAttr(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    optiling::reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    if (optiling::reduction == optiling::REDUCTION_NONE) {
        optiling::tilingKey += TILING_KEY_ATTR_NONE;
    } else if (optiling::reduction == optiling::REDUCTION_SUM) {
        optiling::tilingKey += TILING_KEY_ATTR_SUM;
    } else if (optiling::reduction == optiling::REDUCTION_MEAN) {
        optiling::tilingKey += TILING_KEY_ATTR_MEAN;
    } else {
        OP_LOGD(context, "The reduction attribute must be 'none', 'mean', or 'sum'.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void CoreSplit(gert::TilingContext* context, const platform_ascendc::PlatformAscendC* ascendcPlatform)
{
    uint64_t totalLength = context->GetInputShape(INPUT_DATA_IDX)->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        totalLength == 0,
        OP_LOGE(context->GetNodeName(), "The shape size of input should not be 0"), return);
    double scale = 1. / totalLength;
    optiling::mseLossV2TilingData.set_scale(static_cast<float>(scale));

    uint64_t totalBlocks = Ops::Base::FloorDiv(totalLength, static_cast<uint64_t>(optiling::elemsPerBlock));
    uint8_t tailElems = optiling::Mod(totalLength, optiling::elemsPerBlock);
    optiling::mseLossV2TilingData.set_tailElems(tailElems);
    uint64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (totalBlocks < coreNum * optiling::BLOCK_PER_CORE) {
        coreNum = Ops::Base::CeilDiv(totalBlocks, static_cast<uint64_t>(optiling::BLOCK_PER_CORE));
    }
    if (coreNum == 0u) {
        coreNum = 1u;
    }

    context->SetBlockDim(coreNum);
    optiling::kernelNumber = coreNum;

    optiling::blockPerCore = Ops::Base::FloorDiv(totalBlocks, coreNum);
    optiling::tailBlocks = optiling::Mod(totalBlocks, coreNum);
    optiling::mseLossV2TilingData.set_coreLength(optiling::blockPerCore * optiling::elemsPerBlock);
    optiling::mseLossV2TilingData.set_coreNum(coreNum);
}

static void UBSplit(const platform_ascendc::PlatformAscendC* ascendcPlatform)
{
    uint64_t UBSize = 0u;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, UBSize);

    auto CalcBlockPerQue = [UBSize](uint32_t bufferNum) {
        constexpr uint64_t INQUE_NUM = 2u;

        if (optiling::reduction == optiling::REDUCTION_NONE) {
            uint64_t UBTotalBlocks = UBSize / optiling::BYTES_PER_BLOCK;
            constexpr uint64_t OUTQUE_NUM = 1u;

            if (optiling::dtype == ge::DT_FLOAT) {
                optiling::blockPerQue = Ops::Base::FloorDiv(UBTotalBlocks, ((INQUE_NUM + OUTQUE_NUM) * bufferNum));
            } else {
                optiling::blockPerQue =
                    Ops::Base::FloorDiv(UBTotalBlocks, ((INQUE_NUM + OUTQUE_NUM) * bufferNum + CAST_BUFFER_NUM));
            }
        } else {
            uint64_t preservedSize = static_cast<uint64_t>(optiling::BYTES_PER_BLOCK * (optiling::kernelNumber + 1u));
            uint64_t UBTotalBlocks = Ops::Base::FloorDiv((UBSize - preservedSize), static_cast<uint64_t>(optiling::BYTES_PER_BLOCK));

            if (optiling::dtype == ge::DT_FLOAT) {
                optiling::blockPerQue = Ops::Base::FloorDiv(UBTotalBlocks, (INQUE_NUM * bufferNum));
            } else {
                optiling::blockPerQue = Ops::Base::FloorDiv(UBTotalBlocks, (INQUE_NUM * bufferNum + CAST_BUFFER_NUM));
            }
        }
    };
    uint8_t bufferNum = 1u;
    CalcBlockPerQue(bufferNum);
    uint32_t epochs = Ops::Base::FloorDiv(optiling::blockPerCore, optiling::blockPerQue);
    if (epochs > 1u) {
        bufferNum = 2u; // double buffer
        CalcBlockPerQue(bufferNum);
    }

    optiling::mseLossV2TilingData.set_bufferNum(bufferNum);
    optiling::mseLossV2TilingData.set_tileLength(optiling::blockPerQue * optiling::elemsPerBlock);
}

static void TileSplit()
{
    uint32_t epochs = Ops::Base::FloorDiv(optiling::blockPerCore, optiling::blockPerQue);
    uint32_t tailBlocksPerCore = optiling::Mod(optiling::blockPerCore, optiling::blockPerQue);
    optiling::mseLossV2TilingData.set_epochs(epochs);
    optiling::mseLossV2TilingData.set_tailTileLength(tailBlocksPerCore * optiling::elemsPerBlock);

    uint64_t blockForLastCore = optiling::blockPerCore + optiling::tailBlocks;
    uint64_t epochsForLastCore = Ops::Base::FloorDiv(blockForLastCore, optiling::blockPerQue);
    uint64_t tailBlocksForLastCore = optiling::Mod(blockForLastCore, optiling::blockPerQue);
    optiling::mseLossV2TilingData.set_epochsForLastCore(epochsForLastCore);
    optiling::mseLossV2TilingData.set_tailTileLengthForLastCore(tailBlocksForLastCore * optiling::elemsPerBlock);
}

static ge::graphStatus GetTilingData(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    optiling::CoreSplit(context, &ascendcPlatform);

    optiling::UBSplit(&ascendcPlatform);

    optiling::TileSplit();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    if (optiling::reduction == optiling::REDUCTION_NONE) {
        currentWorkspace[0] = 0u;
    } else {
        uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint32_t synchronizedWorkspaceSize = optiling::BYTES_PER_BLOCK * optiling::kernelNumber;
        uint32_t syncAllWorkspaceSize = optiling::BYTES_PER_BLOCK * optiling::kernelNumber;
        currentWorkspace[0] = sysWorkspaceSize + syncAllWorkspaceSize + synchronizedWorkspaceSize;
    }

    return ge::GRAPH_SUCCESS;
}

static void PrintInfo(const gert::TilingContext* context)
{
    OP_LOGD(context, ">>>>>>>>>>>>> mse_loss_v2 tiling data begin <<<<<<<<<<<<<");
    OP_LOGD(context, "tailElems = %lu.", optiling::mseLossV2TilingData.get_tailElems());
    OP_LOGD(context, "bufferNum = %lu.", optiling::mseLossV2TilingData.get_bufferNum());
    OP_LOGD(context, "scale = %f.", optiling::mseLossV2TilingData.get_scale());
    OP_LOGD(context, "epochs = %lu.", optiling::mseLossV2TilingData.get_epochs());
    OP_LOGD(context, "epochsForLastCore = %lu.", optiling::mseLossV2TilingData.get_epochsForLastCore());
    OP_LOGD(context, "coreLength = %lu.", optiling::mseLossV2TilingData.get_coreLength());
    OP_LOGD(context, "tileLength = %lu.", optiling::mseLossV2TilingData.get_tileLength());
    OP_LOGD(context, "tailTileLength = %lu.", optiling::mseLossV2TilingData.get_tailTileLength());
    OP_LOGD(
        context, "tailTileLengthForLastCore = %lu.", optiling::mseLossV2TilingData.get_tailTileLengthForLastCore());
    OP_LOGD(context, ">>>>>>>>>>>>> mse_loss_v2 tiling data end <<<<<<<<<<<<<");
}

ge::graphStatus Tiling4MSELossV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling For MSELossV2 starts");
    // init tiling data
    optiling::reduction = optiling::REDUCTION_MEAN;
    optiling::dtype = ge::DT_MAX;
    optiling::elemsPerBlock = 0u;
    optiling::tilingKey = 0ULL;
    optiling::kernelNumber = 0u;
    optiling::blockPerQue = 0ULL;
    optiling::blockPerCore = 0ULL;
    optiling::tailBlocks = 0ULL;

    OP_CHECK_IF(
        CheckInputDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckInputDtype failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetTilingAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "GetTilingAttr failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetTilingData(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "GetTilingData failed."),
        return ge::GRAPH_FAILED);

    context->GetRawTilingData()->SetDataSize(optiling::mseLossV2TilingData.GetDataSize());
    context->SetTilingKey(optiling::tilingKey);
    optiling::mseLossV2TilingData.SaveToBuffer(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    PrintInfo(context);
    OP_LOGD(context, "Tiling4MSELossV2 end");
    return ge::GRAPH_SUCCESS;
}

struct MSELossV2CompileInfo {
};

ge::graphStatus TilingParse4MSELossV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MSELossV2).Tiling(Tiling4MSELossV2).TilingParse<MSELossV2CompileInfo>(TilingParse4MSELossV2);
} // namespace optiling