/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bn_training_reduce_tiling_arch35.h"

#include "bn_training_reduce_tiling_public.h"
#include <algorithm>
#include <limits>

#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

#include "../../op_kernel/arch35/bn_training_reduce_tiling_data.h"

namespace optiling {
namespace {
constexpr int64_t kSmallRTileChannels = 64;
constexpr int64_t kBlockBytes = 32;

struct BNTrainingReduceCompileInfo {};

ge::graphStatus TilingParseForBNTrainingReduce([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

int64_t DTypeBytes(BNTrainingReducePublicDType dtype)
{
    if (dtype == BNTrainingReducePublicDType::FLOAT32) {
        return 4;
    }
    if (dtype == BNTrainingReducePublicDType::FLOAT16 || dtype == BNTrainingReducePublicDType::BFLOAT16) {
        return 2;
    }
    return 0;
}

int64_t AlignBytes(int64_t bytes) { return (bytes + kBlockBytes - 1) / kBlockBytes * kBlockBytes; }

bool IsFirstRoundSmallR(int64_t reduceLen)
{
    return reduceLen == 1 || reduceLen == 3 || reduceLen == 10 || reduceLen == 17;
}

bool TryBuildSmallRConfig(const BNTrainingReducePublicInputs& inputs, BNTrainingReducePublicResult& result)
{
    const int64_t n = inputs.shape[0];
    const int64_t channels = inputs.shape[1];
    const int64_t h = inputs.shape[2];
    const int64_t w = inputs.shape[3];
    const int64_t dtypeBytes = DTypeBytes(inputs.inputDtype);
    if (!inputs.inputPresent || inputs.rank != 4 || inputs.format != BNTrainingReducePublicFormat::NCHW || n != 1 ||
        channels <= 0 || h <= 0 || w <= 0 || dtypeBytes == 0 || inputs.coreNum <= 0 || inputs.ubSize <= 0 ||
        h > std::numeric_limits<int64_t>::max() / w) {
        return false;
    }
    const int64_t reduceLen = h * w;
    if (!IsFirstRoundSmallR(reduceLen)) {
        return false;
    }

    const int64_t inputBytes = AlignBytes(kSmallRTileChannels * reduceLen * dtypeBytes);
    const int64_t outputBytes = AlignBytes(kSmallRTileChannels * static_cast<int64_t>(sizeof(float)));
    if (inputBytes > inputs.ubSize || outputBytes > inputs.ubSize - inputBytes ||
        outputBytes > inputs.ubSize - inputBytes - outputBytes) {
        return false;
    }

    const int64_t tileCount = (channels + kSmallRTileChannels - 1) / kSmallRTileChannels;
    const int32_t usedCores = static_cast<int32_t>(std::min<int64_t>(tileCount, inputs.coreNum));
    if (usedCores <= 0) {
        return false;
    }
    const int64_t smallLoops = tileCount / usedCores;
    const int32_t bigCores = static_cast<int32_t>(tileCount % usedCores);

    result.status = BNTrainingReducePublicStatus::SUCCESS;
    result.tilingKey = static_cast<int64_t>(BNTrainingReduceTilingKey::SMALL_R);
    result.blockDim = static_cast<uint32_t>(usedCores);
    result.workspaceSize = inputs.systemWorkspaceSize;
    result.scheduleMode = 0;
    auto& td = result.tilingData;
    td.axisNum = 2;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td.axisShape[i] = 1;
        td.axisStride[i] = 0;
    }
    td.axisShape[0] = channels;
    td.axisShape[1] = reduceLen;
    td.axisStride[0] = reduceLen;
    td.axisStride[1] = 1;
    td.aLoopCntTotal = tileCount;
    td.aSplitChunkCnt = tileCount;
    td.aBigCoreLoopCnt = smallLoops + (bigCores > 0 ? 1 : 0);
    td.aSmallCoreLoopCnt = smallLoops;
    td.aBigCoreCnt = bigCores;
    td.usedCoreNum = usedCores;
    td.aSplitAxisIdx = 0;
    td.rSplitAxisIdx = 1;
    td.aUbFactor = kSmallRTileChannels;
    td.aUbFactorAlign = kSmallRTileChannels;
    td.rUbFactor = reduceLen;
    td.rUbFactorAlign = reduceLen;
    td.innerAProd = 1;
    td.innerAProdAlign = 1;
    td.innerRProd = 1;
    td.innerRProdAlign = 1;
    td.rLoopCntTotal = 1;
    td.preReduceUbSize = inputBytes;
    td.postReduceUbSize = outputBytes;
    td.tmpBufUbSize = outputBytes;
    td.cacheBufUbSize = 0;
    td.rGroupCnt = 0;
    return true;
}

BNTrainingReducePublicDType ConvertDType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return BNTrainingReducePublicDType::FLOAT16;
        case ge::DT_BF16:
            return BNTrainingReducePublicDType::BFLOAT16;
        case ge::DT_FLOAT:
            return BNTrainingReducePublicDType::FLOAT32;
        default:
            return BNTrainingReducePublicDType::INT32;
    }
}

bool NormalizeOutputShape(const gert::Shape& outputShape, BNTrainingReducePublicFormat format, int32_t& rank,
                          int64_t& channel)
{
    const size_t outputRank = outputShape.GetDimNum();
    if (outputRank > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    rank = static_cast<int32_t>(outputRank);
    channel = rank == 1 ? outputShape.GetDim(0) : 0;
    // ACLNN expands the public one-dimensional output to the input layout
    // before the AICore launch and squeezes it back afterwards.
    const bool isNchwOutput = format == BNTrainingReducePublicFormat::NCHW && rank == 4 && outputShape.GetDim(0) == 1 &&
                              outputShape.GetDim(2) == 1 && outputShape.GetDim(3) == 1;
    const bool isNhwcOutput = format == BNTrainingReducePublicFormat::NHWC && rank == 4 && outputShape.GetDim(0) == 1 &&
                              outputShape.GetDim(1) == 1 && outputShape.GetDim(2) == 1;
    if (isNchwOutput || isNhwcOutput) {
        rank = 1;
        channel = outputShape.GetDim(format == BNTrainingReducePublicFormat::NCHW ? 1 : 3);
    }
    return true;
}

bool PopulateInterfaceInputs(gert::TilingContext* context, BNTrainingReducePublicInputs& inputs)
{
    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    inputs.inputPresent = inputShape != nullptr && inputDesc != nullptr;
    if (!inputs.inputPresent) {
        return true;
    }

    const auto& xShape = inputShape->GetStorageShape();
    const size_t inputRank = xShape.GetDimNum();
    if (inputRank > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    inputs.rank = static_cast<int32_t>(inputRank);
    if (inputs.rank == 4) {
        for (size_t i = 0; i < inputs.shape.size(); ++i) {
            inputs.shape[i] = xShape.GetDim(i);
        }
    }
    const ge::Format storageFormat = inputDesc->GetStorageFormat();
    if (storageFormat == ge::FORMAT_NCHW) {
        inputs.format = BNTrainingReducePublicFormat::NCHW;
    } else if (storageFormat == ge::FORMAT_NHWC) {
        inputs.format = BNTrainingReducePublicFormat::NHWC;
    } else {
        OP_LOGE(context, "BNTrainingReduce only supports NCHW and NHWC, but got format %d.",
                static_cast<int32_t>(storageFormat));
        return false;
    }
    inputs.inputDtype = ConvertDType(inputDesc->GetDataType());

    const auto* sumShape = context->GetOutputShape(0);
    const auto* squareSumShape = context->GetOutputShape(1);
    const auto* sumDesc = context->GetOutputDesc(0);
    const auto* squareSumDesc = context->GetOutputDesc(1);
    if (sumShape == nullptr || squareSumShape == nullptr || sumDesc == nullptr || squareSumDesc == nullptr) {
        return false;
    }

    const auto& sumStorageShape = sumShape->GetStorageShape();
    if (!NormalizeOutputShape(sumStorageShape, inputs.format, inputs.sumRank, inputs.sumDim0)) {
        return false;
    }
    inputs.sumDtype = ConvertDType(sumDesc->GetDataType());
    const auto& squareSumStorageShape = squareSumShape->GetStorageShape();
    if (!NormalizeOutputShape(squareSumStorageShape, inputs.format, inputs.squareSumRank, inputs.squareSumDim0)) {
        return false;
    }
    inputs.squareSumDtype = ConvertDType(squareSumDesc->GetDataType());
    inputs.deterministic = context->GetDeterministic() == 1;
    return true;
}

bool PopulatePlatformInputs(gert::TilingContext* context, BNTrainingReducePublicInputs& inputs)
{
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return false;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    inputs.coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    inputs.ubSize = static_cast<int64_t>(ubSize);
    inputs.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    inputs.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));
    inputs.vectorSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    inputs.systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    return true;
}

bool IsLegalTilingKey(int64_t tilingKey)
{
    const auto key = static_cast<BNTrainingReduceTilingKey>(tilingKey);
    return key == BNTrainingReduceTilingKey::NORMAL_TAIL_A || key == BNTrainingReduceTilingKey::GROUP_TAIL_A ||
           key == BNTrainingReduceTilingKey::EMPTY || key == BNTrainingReduceTilingKey::NORMAL_TAIL_R ||
           key == BNTrainingReduceTilingKey::GROUP_TAIL_R || key == BNTrainingReduceTilingKey::SMALL_R ||
           key == BNTrainingReduceTilingKey::DETERMINISTIC_GROUP_TAIL_A ||
           key == BNTrainingReduceTilingKey::DETERMINISTIC_GROUP_TAIL_R;
}

} // namespace

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context->GetNodeName(), "Enter TilingFunc");
    BNTrainingReducePublicInputs inputs;
    if (!PopulateInterfaceInputs(context, inputs) || !PopulatePlatformInputs(context, inputs)) {
        OP_LOGE(context, "Failed to populate BNTrainingReduce tiling inputs.");
        return ge::GRAPH_FAILED;
    }
    BNTrainingReducePublicResult result;
    if (!TryBuildSmallRConfig(inputs, result)) {
        result = ComputeBNTrainingReducePublicTiling(inputs);
    }
    if (result.status != BNTrainingReducePublicStatus::SUCCESS || !IsLegalTilingKey(result.tilingKey)) {
        OP_LOGE(context, "Failed to compute a legal BNTrainingReduce tiling result.");
        return ge::GRAPH_FAILED;
    }

    auto* tilingData = context->GetTilingData<BNTrainingReduceTilingData>();
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (tilingData == nullptr || workspaceSizes == nullptr) {
        OP_LOGE(context, "Failed to get BNTrainingReduce tiling data or workspace.");
        return ge::GRAPH_FAILED;
    }
    if (result.scheduleMode == 1 && context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Failed to set BNTrainingReduce schedule mode.");
        return ge::GRAPH_FAILED;
    }
    if (context->SetTilingKey(static_cast<uint64_t>(result.tilingKey)) != ge::GRAPH_SUCCESS ||
        context->SetBlockDim(result.blockDim) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Failed to set BNTrainingReduce tiling key or block dim.");
        return ge::GRAPH_FAILED;
    }

    *tilingData = result.tilingData;
    workspaceSizes[0] = result.workspaceSize;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BNTrainingReduce)
    .Tiling(TilingFunc)
    .TilingParse<BNTrainingReduceCompileInfo>(TilingParseForBNTrainingReduce);
} // namespace optiling
