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
 * \file in_training_update_grad_gamma_beta_tiling_arch35.cpp
 * \brief Deterministic axis-zero reduction tiling for Ascend 950.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string>

#include "graph/utils/type_utils.h"
#include "in_training_update_grad_gamma_beta_tiling_arch35.h"
#include "norm/in_training_update_grad_gamma_beta/op_kernel/arch35/in_training_update_grad_gamma_beta_tiling_data.h"
#include "norm/in_training_update_grad_gamma_beta/op_kernel/arch35/in_training_update_grad_gamma_beta_tiling_key.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr size_t INPUT_RES_GAMMA_INDEX = 0U;
constexpr size_t INPUT_RES_BETA_INDEX = 1U;
constexpr size_t OUTPUT_PD_GAMMA_INDEX = 0U;
constexpr size_t OUTPUT_PD_BETA_INDEX = 1U;
constexpr size_t REDUCE_AXIS = 0U;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t RESERVED_UB_BYTES = 4096UL;
constexpr uint32_t DTYPE_MODE_FLOAT32 = 0U;
constexpr uint64_t DATA_COPY_MAX_BYTES = 2097151UL;
constexpr uint64_t DATA_COPY_MAX_BLOCK_COUNT = 4095UL;
constexpr uint64_t DATA_COPY_MAX_SRC_STRIDE_BYTES = (1UL << 40U) - 1UL;
constexpr uint64_t MAX_VECTOR_LOOPS = std::numeric_limits<uint16_t>::max();
constexpr const char* OP_TYPE = "INTrainingUpdateGradGammaBeta";

bool SafeMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs != 0U && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

uint64_t CeilDiv(uint64_t value, uint64_t factor)
{
    return value / factor + static_cast<uint64_t>(value % factor != 0U);
}

const char* EntityNameOf(const gert::TilingContext* context)
{
    if (context == nullptr || context->GetNodeName() == nullptr) {
        return OP_TYPE;
    }
    return context->GetNodeName();
}

bool IsSupportedFormat(ge::Format format)
{
    static const std::set<ge::Format> SUPPORTED_FORMATS = {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW,
                                                           ge::FORMAT_NDHWC, ge::FORMAT_ND};
    return SUPPORTED_FORMATS.find(format) != SUPPORTED_FORMATS.end();
}

bool FormatMatchesRank(ge::Format format, size_t rank)
{
    if (format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC) {
        return rank == 4U;
    }
    if (format == ge::FORMAT_NCDHW || format == ge::FORMAT_NDHWC) {
        return rank == 5U;
    }
    return format == ge::FORMAT_ND && (rank == 4U || rank == 5U);
}

ge::graphStatus CheckTensorDesc(const gert::TilingContext* context, const gert::CompileTimeTensorDesc* desc,
                                const char* tensorName)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    EntityNameOf(context), tensorName,
                    ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), "only float32 is supported"),
                return ge::GRAPH_FAILED);
    const ge::Format storageFormat = desc->GetStorageFormat();
    OP_CHECK_IF(!IsSupportedFormat(storageFormat),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(EntityNameOf(context), tensorName,
                                                       ge::TypeUtils::FormatToSerialString(storageFormat),
                                                       "format must be NCHW, NHWC, NCDHW, NDHWC or ND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateContract(gert::TilingContext* context, const gert::Shape*& inputShape,
                                 ge::Format& storageFormat)
{
    const auto* gammaDesc = context->GetInputDesc(INPUT_RES_GAMMA_INDEX);
    const auto* betaDesc = context->GetInputDesc(INPUT_RES_BETA_INDEX);
    const auto* pdGammaDesc = context->GetOutputDesc(OUTPUT_PD_GAMMA_INDEX);
    const auto* pdBetaDesc = context->GetOutputDesc(OUTPUT_PD_BETA_INDEX);
    if (CheckTensorDesc(context, gammaDesc, "res_gamma") != ge::GRAPH_SUCCESS ||
        CheckTensorDesc(context, betaDesc, "res_beta") != ge::GRAPH_SUCCESS ||
        CheckTensorDesc(context, pdGammaDesc, "pd_gamma") != ge::GRAPH_SUCCESS ||
        CheckTensorDesc(context, pdBetaDesc, "pd_beta") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    storageFormat = gammaDesc->GetStorageFormat();
    OP_CHECK_IF(betaDesc->GetStorageFormat() != storageFormat || pdGammaDesc->GetStorageFormat() != storageFormat ||
                    pdBetaDesc->GetStorageFormat() != storageFormat,
                OP_LOGE(context, "all inputs and outputs must use the same storage format"), return ge::GRAPH_FAILED);

    const auto* gammaStorageShape = context->GetInputShape(INPUT_RES_GAMMA_INDEX);
    const auto* betaStorageShape = context->GetInputShape(INPUT_RES_BETA_INDEX);
    const auto* pdGammaStorageShape = context->GetOutputShape(OUTPUT_PD_GAMMA_INDEX);
    const auto* pdBetaStorageShape = context->GetOutputShape(OUTPUT_PD_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, pdGammaStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, pdBetaStorageShape);

    const gert::Shape& gammaShape = gammaStorageShape->GetStorageShape();
    const gert::Shape& betaShape = betaStorageShape->GetStorageShape();
    const gert::Shape& pdGammaShape = pdGammaStorageShape->GetStorageShape();
    const gert::Shape& pdBetaShape = pdBetaStorageShape->GetStorageShape();
    const size_t rank = gammaShape.GetDimNum();
    OP_CHECK_IF(rank != 4U && rank != 5U,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(EntityNameOf(context), "res_gamma",
                                                         std::to_string(rank).c_str(), "rank must be 4 or 5"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaShape.GetDimNum() != rank || pdGammaShape.GetDimNum() != rank || pdBetaShape.GetDimNum() != rank,
                OP_LOGE(context, "all input and output ranks must match"), return ge::GRAPH_FAILED);

    const ge::Format gammaOriginFormat = gammaDesc->GetOriginFormat();
    const ge::Format betaOriginFormat = betaDesc->GetOriginFormat();
    OP_CHECK_IF(!IsSupportedFormat(gammaOriginFormat) || !IsSupportedFormat(betaOriginFormat) ||
                    gammaOriginFormat != betaOriginFormat,
                OP_LOGE(context, "input origin formats must be identical and supported"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!FormatMatchesRank(gammaOriginFormat, rank),
                OP_LOGE(context, "input origin format does not match rank %zu", rank), return ge::GRAPH_FAILED);

    for (size_t index = 0U; index < rank; ++index) {
        const int64_t gammaDim = gammaShape.GetDim(index);
        OP_CHECK_IF(gammaDim < 0, OP_LOGE(context, "res_gamma dim[%zu]=%ld must be nonnegative", index, gammaDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(betaShape.GetDim(index) != gammaDim,
                    OP_LOGE(context, "res_beta dim[%zu]=%ld differs from res_gamma dim[%zu]=%ld", index,
                            betaShape.GetDim(index), index, gammaDim),
                    return ge::GRAPH_FAILED);
        const int64_t expectedOutputDim = index == REDUCE_AXIS ? 1 : gammaDim;
        OP_CHECK_IF(pdGammaShape.GetDim(index) != expectedOutputDim || pdBetaShape.GetDim(index) != expectedOutputDim,
                    OP_LOGE(context, "output dim[%zu] must be %ld", index, expectedOutputDim), return ge::GRAPH_FAILED);
    }
    inputShape = &gammaShape;
    return ge::GRAPH_SUCCESS;
}

uint32_t ComputeScaleExponent(uint64_t reduceCount)
{
    if (reduceCount == 0U) {
        return 0U;
    }
    uint32_t ceilLog2 = 0U;
    uint64_t coveredValues = 1U;
    while (coveredValues < reduceCount) {
        coveredValues <<= 1U;
        ++ceilLog2;
    }
    return ceilLog2 + 1U;
}
} // namespace

namespace optiling {
ge::graphStatus INTrainingUpdateGradGammaBetaTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = nullptr;
    ge::Format storageFormat = ge::FORMAT_RESERVED;
    OP_CHECK_IF(ValidateContract(context, inputShape, storageFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "operator contract validation failed"), return ge::GRAPH_FAILED);

    uint64_t outputElements = 1U;
    for (size_t index = REDUCE_AXIS + 1U; index < inputShape->GetDimNum(); ++index) {
        uint64_t multiplied = 0U;
        OP_CHECK_IF(!SafeMultiply(outputElements, static_cast<uint64_t>(inputShape->GetDim(index)), multiplied),
                    OP_LOGE(context, "output element count overflows uint64"), return ge::GRAPH_FAILED);
        outputElements = multiplied;
    }
    const uint64_t reduceCount = static_cast<uint64_t>(inputShape->GetDim(REDUCE_AXIS));
    uint64_t inputElements = 0U;
    uint64_t inputBytes = 0U;
    uint64_t outputBytes = 0U;
    OP_CHECK_IF(!SafeMultiply(reduceCount, outputElements, inputElements) ||
                    !SafeMultiply(inputElements, FP32_BYTES, inputBytes) ||
                    !SafeMultiply(outputElements, FP32_BYTES, outputBytes) ||
                    inputBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    outputBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "input or output size exceeds the signed 64-bit GM range"), return ge::GRAPH_FAILED);

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t vectorLength = Ops::Base::GetVRegSize(context);
    const uint64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(coreNum == 0U || coreNum > std::numeric_limits<uint32_t>::max() || ubSize <= RESERVED_UB_BYTES ||
                    vectorLength < FP32_BYTES || ubBlockSize < FP32_BYTES || vectorLength < ubBlockSize ||
                    vectorLength % FP32_BYTES != 0U || ubBlockSize % FP32_BYTES != 0U ||
                    vectorLength % ubBlockSize != 0U || vectorLength > DATA_COPY_MAX_BYTES,
                OP_LOGE(context, "invalid Ascend 950 vector platform resources"), return ge::GRAPH_FAILED);

    const uint64_t vectorElements = vectorLength / FP32_BYTES;
    const uint64_t blockElements = ubBlockSize / FP32_BYTES;
    OP_CHECK_IF(blockElements > std::numeric_limits<uint8_t>::max(),
                OP_LOGE(context, "UB block element count does not fit DataCopyPad padding fields"),
                return ge::GRAPH_FAILED);

    uint64_t outputBlocks = 0U;
    uint64_t usedCoreNum = 1U;
    uint64_t baseBlocksPerCore = 0U;
    uint64_t extraBlockCoreCount = 0U;
    uint64_t maxCoreElements = blockElements;
    if (outputElements != 0U) {
        outputBlocks = CeilDiv(outputElements, blockElements);
        usedCoreNum = std::min(outputBlocks, coreNum);
        baseBlocksPerCore = outputBlocks / usedCoreNum;
        extraBlockCoreCount = outputBlocks % usedCoreNum;
        const uint64_t maxBlocksPerCore = baseBlocksPerCore + static_cast<uint64_t>(extraBlockCoreCount != 0U);
        OP_CHECK_IF(!SafeMultiply(maxBlocksPerCore, blockElements, maxCoreElements),
                    OP_LOGE(context, "per-core output ownership overflows uint64"), return ge::GRAPH_FAILED);
    }

    const uint64_t partialCount = std::min(std::max(reduceCount, uint64_t{1}),
                                           uint64_t{IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_MAX_PARTIALS});
    // N <= 2 uses one direct accumulator. Larger reductions reserve at least
    // four accumulator slots for sum, compensation, retry flags, and retry
    // reduction work. The expansion reuses those slots and may require more;
    // its per-column length and the magnitude scan use two additional buffers.
    const uint64_t stateBufferCount = reduceCount <= 2U ? 1U : std::max(partialCount, uint64_t{4}) + 2U;
    const uint64_t inputRowsForSizing = reduceCount <= 2U ? std::max(reduceCount, uint64_t{1}) : 1U;
    const uint64_t usableUbElements = (ubSize - RESERVED_UB_BYTES) / FP32_BYTES;
    const uint64_t maxAlignmentSlackElements = vectorElements - blockElements;
    uint64_t totalAlignmentSlackElements = 0U;
    OP_CHECK_IF(!SafeMultiply(stateBufferCount + 1U, maxAlignmentSlackElements, totalAlignmentSlackElements) ||
                    usableUbElements <= totalAlignmentSlackElements,
                OP_LOGE(context, "UB cannot hold vector-safe tail padding"), return ge::GRAPH_FAILED);
    const uint64_t tileBufferCount = stateBufferCount + inputRowsForSizing;
    uint64_t maxTileElements = (usableUbElements - totalAlignmentSlackElements) / tileBufferCount;
    maxTileElements = std::min(maxTileElements, DATA_COPY_MAX_BYTES / FP32_BYTES);
    maxTileElements = std::min(maxTileElements, vectorElements * MAX_VECTOR_LOOPS);
    maxTileElements = maxTileElements / blockElements * blockElements;
    OP_CHECK_IF(maxTileElements == 0U || maxTileElements > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "no valid UB tile can be formed"), return ge::GRAPH_FAILED);

    const uint64_t tileElements = std::min(maxCoreElements, maxTileElements);
    const uint64_t accumulatorStrideElements = CeilDiv(tileElements, vectorElements) * vectorElements;
    uint64_t accumulatorElements = 0U;
    OP_CHECK_IF(!SafeMultiply(stateBufferCount, accumulatorStrideElements, accumulatorElements),
                OP_LOGE(context, "accumulator allocation overflows uint64"), return ge::GRAPH_FAILED);
    const uint64_t inputTailSlackElements = vectorElements - blockElements;
    OP_CHECK_IF(accumulatorElements + inputTailSlackElements >= usableUbElements,
                OP_LOGE(context, "UB cannot hold accumulators and one input row"), return ge::GRAPH_FAILED);
    const uint64_t inputCapacityElements = usableUbElements - accumulatorElements - inputTailSlackElements;
    const uint64_t tileBytes = tileElements * FP32_BYTES;
    const uint64_t inputTailSlackBytes = inputTailSlackElements * FP32_BYTES;

    uint64_t reduceRowsPerTile = 1U;
    if (reduceCount != 0U) {
        reduceRowsPerTile = std::min(reduceCount, DATA_COPY_MAX_BLOCK_COUNT);
        reduceRowsPerTile = std::min(reduceRowsPerTile, inputCapacityElements / tileElements);
        reduceRowsPerTile = std::min(
            reduceRowsPerTile,
            (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - inputTailSlackBytes) / tileBytes);
        if (outputBytes > DATA_COPY_MAX_SRC_STRIDE_BYTES) {
            reduceRowsPerTile = 1U;
        }
    }
    OP_CHECK_IF(reduceRowsPerTile == 0U || reduceRowsPerTile > std::numeric_limits<uint16_t>::max() ||
                    extraBlockCoreCount > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "kernel tiling fields exceed their encoded ranges"), return ge::GRAPH_FAILED);

    float safeMagnitude = std::numeric_limits<float>::max();
    float inputScale = 1.0F;
    float outputScale = 1.0F;
    if (reduceCount > 1U) {
        const uint32_t scaleExponent = ComputeScaleExponent(reduceCount);
        // Bound each unscaled addend by a power of two. The absolute sum is
        // at most 2^127, leaving headroom for FP32 accumulation roundoff.
        safeMagnitude = std::ldexp(1.0F, std::numeric_limits<float>::max_exponent - static_cast<int>(scaleExponent));
        inputScale = std::ldexp(1.0F, -static_cast<int>(scaleExponent));
        outputScale = std::ldexp(1.0F, static_cast<int>(scaleExponent));
    }

    INTrainingUpdateGradGammaBetaTilingData tilingData{};
    tilingData.reduceCount = static_cast<int64_t>(reduceCount);
    tilingData.outputElements = static_cast<int64_t>(outputElements);
    tilingData.baseBlocksPerCore = static_cast<int64_t>(baseBlocksPerCore);
    tilingData.tileElements = static_cast<uint32_t>(tileElements);
    tilingData.reduceRowsPerTile = static_cast<uint32_t>(reduceRowsPerTile);
    tilingData.extraBlockCoreCount = static_cast<uint32_t>(extraBlockCoreCount);
    tilingData.blockElements = static_cast<uint32_t>(blockElements);
    tilingData.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tilingData.safeMagnitude = safeMagnitude;
    tilingData.inputScale = inputScale;
    tilingData.outputScale = outputScale;

    OP_CHECK_IF(context->SetBlockDim(static_cast<uint32_t>(usedCoreNum)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to set block dimension"), return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(context, DTYPE_MODE_FLOAT32);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1U);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();

    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(sizeof(tilingData) > rawTilingData->GetCapacity(), OP_LOGE(context, "raw tiling buffer is too small"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tilingData, sizeof(tilingData)) != EOK,
        OP_LOGE(context, "failed to serialize tiling data"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    OP_LOGI(context,
            "tiling format=%s reduceCount=%ld outputElements=%ld tileElements=%u rowsPerTile=%u usedCoreNum=%u "
            "dtypeMode=%u",
            ge::TypeUtils::FormatToSerialString(storageFormat).c_str(), tilingData.reduceCount,
            tilingData.outputElements, tilingData.tileElements, tilingData.reduceRowsPerTile, tilingData.usedCoreNum,
            DTYPE_MODE_FLOAT32);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForINTrainingUpdateGradGammaBeta(gert::TilingParseContext* context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INTrainingUpdateGradGammaBeta)
    .Tiling(INTrainingUpdateGradGammaBetaTilingFunc)
    .TilingParse<INTrainingUpdateGradGammaBetaCompileInfo>(TilingParseForINTrainingUpdateGradGammaBeta);
} // namespace optiling
