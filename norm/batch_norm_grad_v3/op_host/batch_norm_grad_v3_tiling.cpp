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
 * \file batch_norm_grad_v3_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cstdint>
#include "batch_norm_grad_v3_tiling.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成E89999/EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "BatchNormGradV3", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

ge::graphStatus ParseBatchNormGradV3OutputMask(gert::TilingContext* context, int64_t& outputMask)
{
    constexpr size_t OUTPUT_MASK_ATTR_INDEX = 2;
    outputMask = BNG_V3_OUTPUT_MASK_ALL;
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "GetAttrs() returned null; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto mask = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_MASK_ATTR_INDEX);
    if (mask == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        mask->GetSize() != BNG_V3_OUTPUT_MASK_NUM,
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "output_mask", std::to_string(mask->GetSize()).c_str(),
                                     std::to_string(BNG_V3_OUTPUT_MASK_NUM).c_str()),
        return ge::GRAPH_FAILED);
    auto values = static_cast<const bool*>(mask->GetData());
    outputMask = BNG_V3_OUTPUT_MASK_NONE;
    for (size_t i = 0; i < BNG_V3_OUTPUT_MASK_NUM; i++) {
        if (values[i]) {
            outputMask |= int64_t{1} << i;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static constexpr size_t BNG_WORKSPACE_RESERVED = 0;
static constexpr int64_t DIM_NUM_4 = 4;
static constexpr int64_t DIM_NUM_5 = 5;
static constexpr int64_t INPUT_NUM = 7;
static constexpr int64_t INDEX_0 = 0;
static constexpr int64_t INDEX_1 = 1;
static constexpr int64_t INDEX_2 = 2;
static constexpr int64_t INDEX_3 = 3;
static constexpr int64_t INDEX_4 = 4;
static constexpr int64_t INDEX_5 = 5;
static constexpr int64_t INDEX_6 = 6;
static constexpr int64_t FLOAT_BYTE_NUMBER = 4;
static constexpr int64_t FLOAT16_BYTE_NUMBER = 2;
static constexpr int64_t BNG_PER_CORE_PROCESS_MIN_UB_SIZE = 1024;
static constexpr int64_t BNG_NEED_RESERVE_UB_PARAM_NUM = 5; // mean rstd gamma dbeta dgamma
static constexpr uint64_t BNG_V3_TK_DEFAULT = 0;
static constexpr uint64_t BNG_V3_RAR_ALL_LOAD_TK_BASE = 10000000;
static constexpr uint64_t BNG_V3_RA_ALL_LOAD_TK_BASE = 20000000;
static constexpr uint64_t BNG_V3_RA_PACKED_ALL_LOAD_TK_BASE = 21000000;
static constexpr int64_t RA_FULL_LOAD_UNROLLED_ROWS = 4;
static constexpr int64_t RA_PACKED_MAX_CHANNELS = 32;
static constexpr uint64_t BNG_V3_RAR_RECOMPUTE_TK_BASE = 30000000;
static constexpr uint64_t BNG_V3_RA_RECOMPUTE_TK_BASE = 40000000;
static constexpr uint64_t BNG_V3_RAR_RECOMPUTE_SPLIT_R1 = 1000000;
static constexpr uint64_t BNG_V3_RAR_RECOMPUTE_SPLIT_R0 = 2000000;
static constexpr uint64_t BNG_V3_TK_FLOAT_SUBTYPE = 1;
static constexpr uint64_t BNG_V3_TK_FLOAT16_SUBTYPE = 2;
static constexpr uint64_t BNG_V3_TK_BFLOAT16_SUBTYPE = 3;
static constexpr uint64_t BNG_V3_TK_MIXED_FLOAT16_FLOAT32_SUBTYPE = 4;
static constexpr uint64_t BNG_V3_TK_MIXED_BFLOAT16_FLOAT32_SUBTYPE = 5;
static constexpr int ULONG_BIT_LEN = 64;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t TWO = 2;
static constexpr int64_t THREE = 3;
static constexpr int64_t BASIC_FACTOR = 64;
static constexpr int64_t RA_PACKED_MIN_REDUCE = TWO * BASIC_FACTOR;

// File-local capacity calculations used by the RA tiling implementations.
namespace {
inline int64_t RARecomputeBytesPerChannel(int64_t rDim, int64_t rFactor)
{
    int64_t blockCount = Ops::Base::CeilDiv(rDim, rFactor);
    int64_t cacheCount = blockCount <= 1 ? 1 : ULONG_BIT_LEN - __builtin_clzl(blockCount - 1);
    // All queues reserve FP32 storage: three R-by-C feature tiles, three C-sized
    // parameter queues, and two C-by-cacheCount reduction queues, each double buffered.
    return FLOAT_BYTE_NUMBER * DOUBLE_BUFFER * (THREE * rFactor + THREE + TWO * cacheCount);
}

inline void CalcRARecomputeLoopFactors(int64_t blockFactor, int64_t rDim, uint64_t ubSize, int64_t& aFactor,
                                       int64_t& aFactorAligned, int64_t& rFactor)
{
    if (blockFactor <= BASIC_FACTOR || rDim < BASIC_FACTOR) {
        return;
    }
    constexpr int64_t MIN_R_FACTOR = 16;
    int64_t maxChannels = static_cast<int64_t>(ubSize) / RARecomputeBytesPerChannel(rDim, MIN_R_FACTOR);
    maxChannels = maxChannels / BASIC_FACTOR * BASIC_FACTOR;
    if (maxChannels < BASIC_FACTOR) {
        return;
    }
    // Maximize C using the R=16 UB budget, then retain the largest R in {64, 32, 16}
    // that fits this C tile. The cache refinement may narrow C afterward.
    aFactor = std::min(blockFactor, maxChannels);
    aFactorAligned = Ops::Base::CeilAlign(aFactor, BASIC_FACTOR);
    while (rFactor > MIN_R_FACTOR &&
           aFactorAligned * RARecomputeBytesPerChannel(rDim, rFactor) > static_cast<int64_t>(ubSize)) {
        rFactor /= TWO;
    }
}

inline uint64_t RARecomputeLoopCount(int64_t blockFactor, int64_t rDim, int64_t aFactor, int64_t rFactor)
{
    uint64_t aLoops = blockFactor / aFactor + (blockFactor % aFactor != 0);
    uint64_t rLoops = rDim / rFactor + (rDim % rFactor != 0);
    return aLoops > UINT64_MAX / rLoops ? UINT64_MAX : aLoops * rLoops;
}

inline void CalcRARecomputeCacheFactors(int64_t blockFactor, int64_t rDim, int64_t aDim, int64_t dtypeSize,
                                        int64_t numBlocks, uint64_t ubSize, uint64_t l2Size, int64_t& aFactor,
                                        int64_t& aFactorAligned, int64_t& rFactor)
{
    constexpr int64_t COPY_ALIGN_BYTES = 256;
    int64_t alignedChannels = COPY_ALIGN_BYTES / dtypeSize;
    // 256B alignment is a performance gate for strided copies, not an API requirement.
    // Require both the GM row pitch and the per-core C stride to be aligned.
    // Unaligned layouts retain the wider UB tile to avoid inefficient narrow copies.
    if (blockFactor <= BASIC_FACTOR || rDim < BASIC_FACTOR || numBlocks <= 0 || l2Size == 0 ||
        aDim % alignedChannels != 0 || blockFactor % alignedChannels != 0) {
        return;
    }
    // Estimate the shared-L2 working set as 3 * rDim * Ctile * numBlocks * dtypeSize:
    // x/dy are read again for dx, whose writes may evict their cache lines.
    // Use the full R dimension to model reuse between passes, rather than one R tile.
    // This is a capacity heuristic; the platform reports total L2, not free space.
    uint64_t cacheChannels = l2Size / THREE / dtypeSize / rDim / numBlocks;
    cacheChannels = std::min(cacheChannels, static_cast<uint64_t>(blockFactor));
    cacheChannels = cacheChannels / BASIC_FACTOR * BASIC_FACTOR;
    // Keep the UB tile if even C=64 cannot fit, or the current C already fits.
    if (cacheChannels < BASIC_FACTOR || cacheChannels >= static_cast<uint64_t>(aFactor)) {
        return;
    }
    // Rank by ceil(blockFactor / Ctile) * ceil(rDim / Rtile), an estimate of the
    // per-core C/R tile count rather than execution time.
    uint64_t currentLoops = RARecomputeLoopCount(blockFactor, rDim, aFactor, rFactor);
    uint64_t bestLoops = UINT64_MAX;
    int64_t bestALoops = INT64_MAX;
    int64_t bestA = aFactor;
    int64_t bestR = rFactor;
    // Try R=64/32/16 under both budgets; each R also changes the reduction-cache depth.
    for (int64_t candidateR : {BASIC_FACTOR, BASIC_FACTOR / TWO, BASIC_FACTOR / (TWO * TWO)}) {
        uint64_t ubChannels = ubSize / RARecomputeBytesPerChannel(rDim, candidateR);
        ubChannels = ubChannels / BASIC_FACTOR * BASIC_FACTOR;
        int64_t candidateA = static_cast<int64_t>(std::min(cacheChannels, ubChannels));
        if (candidateA < BASIC_FACTOR) {
            continue;
        }
        uint64_t loops = RARecomputeLoopCount(blockFactor, rDim, candidateA, candidateR);
        int64_t aLoops = blockFactor / candidateA + (blockFactor % candidateA != 0);
        // Equal tile counts favor fewer C loops to limit parameter reloads and tails.
        if (loops < bestLoops || (loops == bestLoops && aLoops < bestALoops)) {
            bestLoops = loops;
            bestALoops = aLoops;
            bestA = candidateA;
            bestR = candidateR;
        }
    }
    // Apply only a candidate whose total C/R tile count does not exceed the UB fallback.
    if (bestLoops != UINT64_MAX && bestLoops <= currentLoops) {
        aFactor = bestA;
        aFactorAligned = bestA;
        rFactor = bestR;
    }
}

inline int64_t CalcRAFullLoadALoopFactor(int64_t blockFactor, int64_t rDim, int64_t binaryRows, int64_t dtypeSize,
                                         int64_t weightDtypeSize, uint64_t blockSize, uint64_t ubSize,
                                         int64_t& loopFactorAligned, int64_t vectorChannels = 0)
{
    int64_t loopFactor = 0;
    loopFactorAligned = 0;
    for (int64_t factor = 1; factor <= blockFactor; ++factor) {
        int64_t factorAligned = static_cast<int64_t>(
                                    Ops::Base::CeilDiv(static_cast<uint64_t>(factor * dtypeSize), blockSize) *
                                    blockSize) /
                                dtypeSize;
        int64_t totalBytes = rDim * factorAligned * dtypeSize * DOUBLE_BUFFER * THREE +
                             factorAligned * FLOAT_BYTE_NUMBER * DOUBLE_BUFFER * TWO +
                             factorAligned * weightDtypeSize * DOUBLE_BUFFER * THREE +
                             binaryRows * factorAligned * FLOAT_BYTE_NUMBER * TWO;
        if (static_cast<uint64_t>(totalBytes) > ubSize) {
            break;
        }
        loopFactor = factor;
        loopFactorAligned = factorAligned;
    }
    // Prefer full fp32 vectors when a narrower vector-aligned tile needs no
    // extra C loops. This is a performance choice after the exact UB check.
    if (vectorChannels > 0 && loopFactor < blockFactor) {
        int64_t vectorFactor = loopFactor / vectorChannels * vectorChannels;
        if (vectorFactor > 0 &&
            Ops::Base::CeilDiv(blockFactor, vectorFactor) == Ops::Base::CeilDiv(blockFactor, loopFactor)) {
            loopFactor = vectorFactor;
            loopFactorAligned = static_cast<int64_t>(
                                    Ops::Base::CeilDiv(static_cast<uint64_t>(loopFactor * dtypeSize), blockSize) *
                                    blockSize) /
                                dtypeSize;
        }
    }
    return loopFactor;
}
} // namespace

template <typename TilingData>
bool SetRABlockFactorsCommon(TilingData& tilingData, int64_t aDim, int64_t dtypeSize, uint32_t coreNum,
                             uint64_t blockSize)
{
    if (aDim <= 0 || dtypeSize <= 0 || coreNum == 0 || blockSize == 0) {
        return false;
    }
    int64_t aFactor = std::max((aDim + coreNum - 1) / coreNum, static_cast<int64_t>(blockSize / dtypeSize));
    if (aFactor <= 0) {
        return false;
    }
    int64_t numBlocks = (aDim + aFactor - 1) / aFactor;
    if (numBlocks <= 0) {
        return false;
    }
    aFactor = (aDim + numBlocks - 1) / numBlocks;
    if (aFactor <= 0) {
        return false;
    }
    int64_t mainBlockFactor = aFactor;
    int64_t mainBlockCount = aDim / aFactor;
    int64_t tailBlockFactor = aDim - mainBlockCount * mainBlockFactor;
    int64_t tailBlockCount = tailBlockFactor > 0 ? 1 : 0;

    tilingData.set_numBlocks(numBlocks);
    tilingData.set_mainBlockFactor(mainBlockFactor);
    tilingData.set_tailBlockFactor(tailBlockFactor);
    tilingData.set_mainBlockCount(mainBlockCount);
    tilingData.set_tailBlockCount(tailBlockCount);
    return true;
}

static std::vector<std::array<ge::DataType, INPUT_NUM>> validInputDtype = {
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16,
     ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT,
     ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT}};

string BNGTensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
    oss << "(shape:" << Shape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << Shape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToSerialString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";

    return oss.str();
}

string BNGDebugTilingContext(gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << BNGTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << BNGTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

ge::graphStatus BatchNormGradV3TilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormGradV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingStageError(
                    context_->GetNodeName(),
                    "GetPlatformInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);
    blockSize = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32 = static_cast<uint64_t>(compileInfo->vlFp32);
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        coreNum = static_cast<int32_t>(compileInfo->coreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::CheckInputDtypeValid()
{
    bool invalid = true;
    for (auto& dtypeList : validInputDtype) {
        invalid = false;
        for (int i = 0; i < INPUT_NUM; i++) {
            auto inputDesc = context_->GetInputDesc(i);
            OP_CHECK_IF(inputDesc == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            context_->GetNodeName(), ("input[" + std::to_string(i) + "]").c_str(),
                            "the input desc of a required input is missing in the tiling context; check that the graph "
                            "node provides all required inputs"),
                        return ge::GRAPH_FAILED);
            auto dtype = inputDesc->GetDataType();
            if (dtype != dtypeList[i]) {
                invalid = true;
                break;
            }
        }
        if (!invalid) {
            break;
        }
    }
    OP_TILING_CHECK(
        invalid == true,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "inputs",
                                                 "the dtype combination of the inputs is not supported; check the "
                                                 "supported dtype combinations of BatchNormGradV3"),
        return ge::GRAPH_FAILED);

    auto dxDesc = context_->GetOutputDesc(0);
    OP_CHECK_IF(
        dxDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dx",
                                                 "the input desc of required input dx is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType dxDtype = dxDesc->GetDataType();
    auto dweightDesc = context_->GetOutputDesc(1);
    OP_CHECK_IF(
        dweightDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dweight",
                                                 "the input desc of required input dweight is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType dweightDtype = dweightDesc->GetDataType();
    auto dbiasDesc = context_->GetOutputDesc(TWO);
    OP_CHECK_IF(
        dbiasDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dbias",
                                                 "the input desc of required input dbias is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType dbiasDtype = dbiasDesc->GetDataType();
    OP_TILING_CHECK(dxDtype != dyDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "dx, dy",
                        (Ops::Base::ToString(dxDtype) + ", " + Ops::Base::ToString(dyDtype)).c_str(),
                        "The dtypes of dx and dy must be the same."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dweightDtype != weightDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "dweight, weight",
                        (Ops::Base::ToString(dweightDtype) + ", " + Ops::Base::ToString(weightDtype)).c_str(),
                        "The dtypes of dweight and weight must be the same."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dbiasDtype != weightDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "dbias, weight",
                        (Ops::Base::ToString(dbiasDtype) + ", " + Ops::Base::ToString(weightDtype)).c_str(),
                        "The dtypes of dbias and weight must be the same."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::CheckSmallShapesValid()
{
    for (int i = INDEX_2; i <= INDEX_6; i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->GetNodeName(), ("input[" + std::to_string(i) + "]").c_str(),
                        "the input shape of a required input is missing in the tiling context; check that the graph "
                        "node provides all required inputs"),
                    return ge::GRAPH_FAILED);
        auto storageShape = shape->GetStorageShape();
        OP_TILING_CHECK(
            storageShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), ("input[" + std::to_string(i) + "]").c_str(),
                                         std::to_string(storageShape.GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);
        int64_t a = storageShape.GetDim(INDEX_0);
        OP_TILING_CHECK(a != aDim,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), ("input[" + std::to_string(i) + "]").c_str(),
                            Ops::Base::ToString(storageShape).c_str(),
                            ("the shape of the input must have " + std::to_string(aDim) + " elements").c_str()),
                        return ge::GRAPH_FAILED);
    }

    for (int i = INDEX_1; i <= INDEX_2; i++) {
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    ReportTilingStageError(context_->GetNodeName(),
                                           "the output shape is missing in the tiling context; check that InferShape "
                                           "executed and the framework propagated the output shapes"),
                    return ge::GRAPH_FAILED);
        auto storageShape = shape->GetStorageShape();
        OP_TILING_CHECK(
            storageShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), ("output[" + std::to_string(i) + "]").c_str(),
                                         std::to_string(storageShape.GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);
        int64_t a = storageShape.GetDim(INDEX_0);
        OP_TILING_CHECK(a != aDim,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), ("output[" + std::to_string(i) + "]").c_str(),
                            Ops::Base::ToString(storageShape).c_str(),
                            ("the shape of the output must have " + std::to_string(aDim) + " elements").c_str()),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static const gert::Shape g_vec_1_shape = {1};

const gert::Shape& BatchNormGradV3TilingBase::EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus BatchNormGradV3TilingBase::CheckBigShapesValid()
{
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dy",
                                                 "the input desc of required input dy is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_IF(
        dyShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dy",
                                                 "the input shape of required input dy is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    int64_t dyDimNum = dyStorageShape.GetDimNum();

    auto xDesc = context_->GetInputDesc(INDEX_1);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                 "the input desc of required input x is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xShape = context_->GetInputShape(INDEX_1);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t xDimNum = xStorageShape.GetDimNum();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_IF(
        dxDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dx",
                                                 "the output desc of required output dx is missing in the tiling "
                                                 "context; check that the graph node provides all required outputs"),
        return ge::GRAPH_FAILED);
    auto dxFormat = dxDesc->GetFormat().GetStorageFormat();
    auto dxShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_IF(dxShape == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "the output shape of dx is missing in the tiling context; check that InferShape "
                                       "executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto dxStorageShape = EnsureNotScalar(dxShape->GetStorageShape());
    int64_t dxDimNum = dxStorageShape.GetDimNum();

    // 校验dim相等
    OP_TILING_CHECK(
        dyDimNum != xDimNum || dyDimNum != dxDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "dy, x, dx",
            (std::to_string(dyDimNum) + ", " + std::to_string(xDimNum) + ", " + std::to_string(dxDimNum)).c_str(),
            "The dim nums of dy, x and dx must be the same"),
        return ge::GRAPH_FAILED);

    // 校验format
    OP_TILING_CHECK(dyFormat != xFormat || dyFormat != dxFormat,
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "dy, x, dx",
                                                            (ge::TypeUtils::FormatToSerialString(dyFormat) + ", " +
                                                             ge::TypeUtils::FormatToSerialString(xFormat) + ", " +
                                                             ge::TypeUtils::FormatToSerialString(dxFormat))
                                                                .c_str(),
                                                            "The formats of dy, x and dx must be the same"),
                    return ge::GRAPH_FAILED);

    if (dyFormat == ge::FORMAT_NCHW || dyFormat == ge::FORMAT_NHWC) {
        OP_TILING_CHECK(
            dyDimNum != DIM_NUM_4,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy", std::to_string(dyDimNum).c_str(),
                                                     "the dim num of dy must be 4 with NCHW or NHWC format"),
            return ge::GRAPH_FAILED);
    } else if (dyFormat == ge::FORMAT_NCDHW || dyFormat == ge::FORMAT_NDHWC) {
        OP_TILING_CHECK(
            dyDimNum != DIM_NUM_5,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy", std::to_string(dyDimNum).c_str(),
                                                     "the dim num of dy must be 5 with NCDHW or NDHWC format"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "dy", ge::TypeUtils::FormatToSerialString(dyFormat).c_str(),
                                   "NCHW, NHWC, NCDHW or NDHWC");
    }

    // 校验shape的dims相同
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_TILING_CHECK(
            xStorageShape.GetDim(i) != dyStorageShape.GetDim(i) || dxStorageShape.GetDim(i) != dyStorageShape.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "dy, x, dx",
                (Ops::Base::ToString(dyStorageShape) + ", " + Ops::Base::ToString(xStorageShape) + ", " +
                 Ops::Base::ToString(dxStorageShape))
                    .c_str(),
                "The shapes of dy, x and dx must be the same"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(dyStorageShape.GetDim(i) <= 0,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy",
                                                              Ops::Base::ToString(dyStorageShape).c_str(),
                                                              "All axes of dy must be positive numbers"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::GetDtypesAndCheckValid()
{
    auto weightDesc = context_->GetInputDesc(INDEX_2);
    OP_CHECK_IF(
        weightDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "weight",
                                                 "the input desc of required input weight is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    weightDtype = weightDesc->GetDataType();
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dy",
                                                 "the input desc of required input dy is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    dyDtype = dyDesc->GetDataType();

    OP_TILING_CHECK(CheckInputDtypeValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "Dtypes check success. DyDtype: [%s], WeightDtype: [%s].",
            ge::TypeUtils::DataTypeToSerialString(dyDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::GetShapesAndCheckValid()
{
    // 大shape 校验
    OP_TILING_CHECK(CheckBigShapesValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // shape 参数获取
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    auto dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    if (dyFormat == ge::FORMAT_NCHW) {
        r1Dim = dyStorageShape.GetDim(INDEX_0);
        aDim = dyStorageShape.GetDim(INDEX_1);
        r0Dim = dyStorageShape.GetDim(INDEX_2) * dyStorageShape.GetDim(INDEX_3);
    } else if (dyFormat == ge::FORMAT_NCDHW) {
        r1Dim = dyStorageShape.GetDim(INDEX_0);
        aDim = dyStorageShape.GetDim(INDEX_1);
        r0Dim = dyStorageShape.GetDim(INDEX_2) * dyStorageShape.GetDim(INDEX_3) * dyStorageShape.GetDim(INDEX_4);
    } else if (dyFormat == ge::FORMAT_NHWC) {
        r1Dim = dyStorageShape.GetDim(INDEX_0) * dyStorageShape.GetDim(INDEX_1) * dyStorageShape.GetDim(INDEX_2);
        aDim = dyStorageShape.GetDim(INDEX_3);
        r0Dim = 1;
    } else if (dyFormat == ge::FORMAT_NDHWC) {
        r1Dim = dyStorageShape.GetDim(INDEX_0) * dyStorageShape.GetDim(INDEX_1) * dyStorageShape.GetDim(INDEX_2) *
                dyStorageShape.GetDim(INDEX_3);
        aDim = dyStorageShape.GetDim(INDEX_4);
        r0Dim = 1;
    }

    // 小shape 校验
    OP_TILING_CHECK(CheckSmallShapesValid() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input small shapes are invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "TilingContext: %s", optiling::BNGDebugTilingContext(context_).c_str());

    OP_TILING_CHECK(ParseBatchNormGradV3OutputMask(context_, outputMask_) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const bool* isTrainingPtr = attrs->GetBool(0);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    OP_TILING_CHECK(isTraining == false, OP_LOGI(context_->GetNodeName(), "This node only support training."),
                    return ge::GRAPH_PARAM_INVALID);

    OP_TILING_CHECK(GetShapesAndCheckValid() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input shapes are invalid."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(GetDtypesAndCheckValid() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input dtypes are invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void BatchNormGradV3TilingBase::DoBinaryAddTiling(BatchNormGradV3BinaryAddTilingData& tilingData, int64_t quotient)
{
    tilingData.set_binaryAddQuotient(quotient);
    int64_t vcaddNum = quotient / vlFp32;
    if (static_cast<uint64_t>(vcaddNum) <= vlFp32) {
        tilingData.set_binaryAddk(0);
        tilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32;
        tilingData.set_binaryAddk(__builtin_ctzl(binaryAddNum));
        tilingData.set_binaryAddLastNum(vlFp32);
    }
}

ge::graphStatus BatchNormGradV3TilingBase::DoOpTiling()
{
    int64_t minADimPerCore = std::max(1L, Ops::Base::CeilDiv(BNG_PER_CORE_PROCESS_MIN_UB_SIZE, onceProcUbNeed));
    int64_t blockNum = coreNum;
    if (minADimPerCore * coreNum > aDim) {
        // 每个核最少要处理 BNG_PER_CORE_PROCESS_MIN_UB_SIZE, minADimPerCore向上取整不可能是0
        blockNum = aDim / minADimPerCore;
        if (blockNum == 0) {
            blockNum = 1;
        }
    }
    OP_TILING_CHECK(
        blockNum == 0,
        ReportTilingStageError(context_->GetNodeName(), "DoOpTiling: block num is 0 after the platform query (core num "
                                                        "is 0); check the platform info, SoC and OPP installation"),
        return ge::GRAPH_FAILED);
    int64_t tailBlockNum = aDim % blockNum;
    int64_t formerBlockDim = aDim / blockNum;
    int64_t tailBlockDim = formerBlockDim + 1;

    baseTilingData.set_r1Dim(r1Dim);
    baseTilingData.set_aDim(aDim);
    baseTilingData.set_r0Dim(r0Dim);
    baseTilingData.set_rAlign(rAlign);
    baseTilingData.set_blockNum(blockNum);
    baseTilingData.set_tailBlockNum(tailBlockNum);
    baseTilingData.set_formerBlockDim(formerBlockDim);
    baseTilingData.set_tailBlockDim(tailBlockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

bool BatchNormGradV3RARFullLoadTilingBase::IsCapable()
{
    if (r0Dim == 1) {
        return false;
    }
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t weightDtypeSize = weightDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t rAlignTo32B = Ops::Base::CeilDiv(static_cast<uint64_t>(r1Dim * r0Dim * dtypeSize), blockSize) * blockSize;
    rAlign = rAlignTo32B / dtypeSize;
    int64_t onceDyUseUbSize = rAlign * dtypeSize;
    // x dbeta dgamma都按照fp32存储，否则dgamma和dx的精度会有损失
    // 同时为了能原地做fp16->fp32，fp16的数据需要存储在buffer后半段
    // 因此x的buffer需要对齐到2个block
    int64_t onceXUseUbSize = Ops::Base::CeilDiv(rAlign * sizeof(float), 2 * blockSize) * 2 * blockSize;
    int64_t onceMeanUseUbSize = FLOAT_BYTE_NUMBER;
    int64_t onceRstdUseUbSize = FLOAT_BYTE_NUMBER;
    int64_t onceGammaUseUbSize = weightDtypeSize;
    int64_t onceDbetaUseUbSize = sizeof(float);
    int64_t onceDgammaUseUbSize = sizeof(float);
    onceProcUbNeed = onceDyUseUbSize + onceXUseUbSize + onceMeanUseUbSize + onceRstdUseUbSize + onceGammaUseUbSize +
                     onceDbetaUseUbSize + onceDgammaUseUbSize;

    reservUbSizeForAlign = BNG_NEED_RESERVE_UB_PARAM_NUM * blockSize;

    binaryAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim * r0Dim)));
    binaryAddQuotient = binaryAddQuotient == r1Dim * r0Dim ? binaryAddQuotient / TWO : binaryAddQuotient;
    int64_t quotientVcaddNum = binaryAddQuotient / vlFp32;
    binaryAddUbNeed = Ops::Base::CeilDiv(quotientVcaddNum * sizeof(float), blockSize) * blockSize;
    // Kernel keeps one binary-add scratch region per pipeline buffer. Store the total here so both
    // admission and formerUbDim deduct the same allocation as InitBuffer(binaryAddBuf_).
    binaryAddUbNeed *= DOUBLE_BUFFER;
    // 开DB占用UB乘以 2
    if ((onceProcUbNeed + reservUbSizeForAlign) * DOUBLE_BUFFER + binaryAddUbNeed > static_cast<int64_t>(ubSize)) {
        OP_LOGD(context_->GetNodeName(),
                "shape: %ld_%ld_%ld onceUbNeed: %ld bigger than "
                "ub: %ld reserve: %ld binaryAdd %ld core: %d, not all load template",
                r0Dim, aDim, r1Dim, onceProcUbNeed, ubSize, reservUbSizeForAlign, binaryAddUbNeed, coreNum);
        return false;
    }
    OP_LOGD(context_->GetNodeName(),
            "shape: %ld_%ld_%ld onceUbNeed: %ld ub: %ld reserve: %ld binaryAdd: %ld core: %d match all load template",
            r0Dim, aDim, r1Dim, onceProcUbNeed, ubSize, reservUbSizeForAlign, binaryAddUbNeed, coreNum);
    return true;
}

ge::graphStatus BatchNormGradV3RARFullLoadTilingBase::DoOpTiling()
{
    OP_TILING_CHECK(
        onceProcUbNeed == 0,
        ReportTilingStageError(
            context_->GetNodeName(),
            "DoOpTiling: onceProcUbNeed is 0 after the capability computation; internal tiling strategy error"),
        return ge::GRAPH_FAILED);
    auto ret = BatchNormGradV3TilingBase::DoOpTiling();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "failed to do base tiling."),
                    return ge::GRAPH_FAILED);

    // 考虑开DB，ubsize要减半
    OP_TILING_CHECK(
        static_cast<int64_t>((ubSize - binaryAddUbNeed) / DOUBLE_BUFFER) < reservUbSizeForAlign,
        ReportTilingStageError(context_->GetNodeName(), "DoOpTiling: the remaining UB size " +
                                                            std::to_string((ubSize - binaryAddUbNeed) / DOUBLE_BUFFER) +
                                                            " bytes is smaller than the required reserve " +
                                                            std::to_string(reservUbSizeForAlign) + " bytes"),
        return ge::GRAPH_FAILED);
    int64_t formerUbDim = ((ubSize - binaryAddUbNeed) / DOUBLE_BUFFER - reservUbSizeForAlign) / onceProcUbNeed;
    OP_TILING_CHECK(
        formerUbDim <= 0,
        ReportTilingStageError(context_->GetNodeName(), "DoOpTiling: formerUbDim is " + std::to_string(formerUbDim) +
                                                            "; the remaining UB cannot fit one processing tile"),
        return ge::GRAPH_FAILED);

    int64_t formerBlockDim = baseTilingData.get_formerBlockDim();
    int64_t ubLoopOfFormerBlock = Ops::Base::CeilDiv(formerBlockDim, formerUbDim);
    int64_t ubTailOfFormerBlock = (formerBlockDim > 0) ? (formerBlockDim - (formerUbDim * (ubLoopOfFormerBlock - 1))) :
                                                         0;

    int64_t tailBlockDim = baseTilingData.get_tailBlockDim();
    int64_t ubLoopOfTailBlock = Ops::Base::CeilDiv(tailBlockDim, formerUbDim);
    int64_t ubTailOfTailBlock = (tailBlockDim > 0) ? (tailBlockDim - (formerUbDim * (ubLoopOfTailBlock - 1))) : 0;

    BatchNormGradV3TilingBase::DoBinaryAddTiling(tilingData.binaryAddTilingData, binaryAddQuotient);

    tilingData.set_formerUbDim(formerUbDim);
    tilingData.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    tilingData.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    tilingData.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    tilingData.set_ubTailOfTailBlock(ubTailOfTailBlock);
    OP_LOGI(context_->GetNodeName(),
            "BatchNormGradV3 do tiling finish. r1: %ld a: %ld r0: %ld rAlign: %ld "
            "block: %ld tailBlock: %ld formerBlockDim: %ld tailBlockDIm: %ld formerUbDim: %ld "
            "ubLoopOfFormerBlock: %ld ubTailOfFormerBlock: %ld ubLoopOfTailBlock: %ld ubTailOfTailBlock: %ld",
            r1Dim, aDim, r0Dim, rAlign, baseTilingData.get_blockNum(), baseTilingData.get_tailBlockNum(),
            formerBlockDim, tailBlockDim, formerUbDim, ubLoopOfFormerBlock, ubTailOfFormerBlock, ubLoopOfTailBlock,
            ubTailOfTailBlock);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3RARFullLoadTilingBase::GetTilingKey() const
{
    if (dyDtype != ge::DataType::DT_FLOAT && dyDtype != ge::DataType::DT_FLOAT16 && dyDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyDtype).c_str(),
                                              "The dtype of dy must be FLOAT, FLOAT16 or BF16.");
        return BNG_V3_TK_DEFAULT;
    }
    uint64_t tilingKeyValue = BNG_V3_RAR_ALL_LOAD_TK_BASE;
    OP_LOGD(context_->GetNodeName(), "BatchNormGradV3 get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradV3RARFullLoadTilingBase::GetWorkspaceSize()
{
    // workspace size is coreNum * colAlignV * 4 * 2
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3RARFullLoadTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    OP_TILING_CHECK(tilingKey == BNG_V3_TK_DEFAULT,
                    ReportTilingStageError(
                        context_->GetNodeName(),
                        "PostTiling: tiling key is 0 (default); the template failed to produce a valid tiling key"),
                    return ge::GRAPH_FAILED);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "PostTiling: GetWorkspaceSizes(1) returned null; required output-slot count is "
                                       "1. Check the TilingContext construction, not the user device allocation"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData.set_outputMask(outputMask_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradV3RARRecomputeTilingBase::IsCapable()
{
    if (r0Dim == 1) {
        return false;
    }
    // 与 Kernel InitBuffer 的分配保持一致，预留 dy/x 之外的 UB 开销。
    constexpr uint64_t extraUbSize = 8000;
    binaryAddBufSize = (ubSize - extraUbSize) / TWO / DOUBLE_BUFFER;
    // SplitR1 的 dy/x 按两个 block 向上对齐，单块预算需向下对齐以覆盖实际分配。
    binaryAddBufSize = binaryAddBufSize / (TWO * blockSize) * (TWO * blockSize);

    // R0不可全载，核内切R0
    subTilingKey = BNG_V3_RAR_RECOMPUTE_SPLIT_R0;
    if (Ops::Base::CeilDiv(r0Dim * sizeof(float) * TWO, blockSize) * blockSize <= binaryAddBufSize) {
        // R0可全载，核内切R1
        subTilingKey = BNG_V3_RAR_RECOMPUTE_SPLIT_R1;
    }

    OP_LOGD(context_->GetNodeName(), "shape: %ld_%ld_%ld match recompute template", r0Dim, aDim, r1Dim);
    return true;
}

void BatchNormGradV3RARRecomputeTilingBase::DoRecomputeTilingSplitR1()
{
    if (Ops::Base::CeilDiv(r1Dim * r0Dim * sizeof(float), blockSize) * blockSize <= binaryAddBufSize) {
        ubRDimFactor = r1Dim * r0Dim;
        ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
        tilingData.set_ubRDimFactor(ubRDimFactor);
        tilingData.set_ubRDimFactorAlign(ubRDimFactorAlign);
        tilingData.set_ubRDimLoopNum(1);
        tilingData.set_ubRDimTail(0);
        tilingData.set_ubRDimTailFactor(0);
        tilingData.set_ubRDimTailFactorAlign(0);
        tilingData.set_ubRDimTailLoopNum(0);
        tilingData.set_ubRDimTailTail(0);
        tilingData.set_ubRDimTailTailFactor(0);
        tilingData.set_ubRDimTailTailFactorAlign(0);
        tilingData.set_ubRDimTailTailLoopNum(0);
        return;
    }
    r1Factor = (binaryAddBufSize / (r0Dim * sizeof(float))) & (~1L);
    ubRDimLoopNum = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim / r1Factor)));
    ubRDimFactor = r1Factor * r0Dim;
    ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTail = r1Dim * r0Dim - ubRDimFactor * ubRDimLoopNum;
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimFactor / TWO : ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(ubRDimTailFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    SetUbRDimTilingData();
}

void BatchNormGradV3RARRecomputeTilingBase::DoRecomputeTilingSplitR0()
{
    r0Factor = r0Dim * sizeof(float) / binaryAddBufSize;
    ubRDimLoopNum = r0Factor <= 0 ? 1L : std::max(1L, (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r0Factor))));
    ubRDimFactor = std::min(static_cast<uint64_t>(r0Dim), binaryAddBufSize / sizeof(float));
    ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTail = r0Dim - ubRDimFactor * ubRDimLoopNum;
    // ubRDimFactor / 2 * 2 may be less than ubRDimFactor
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? (ubRDimFactor + TWO - 1) / TWO :
                                                                                      ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(ubRDimTailFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    SetUbRDimTilingData();
}

void BatchNormGradV3RARRecomputeTilingBase::SetUbRDimTilingData()
{
    tilingData.set_ubRDimFactor(ubRDimFactor);
    tilingData.set_ubRDimFactorAlign(ubRDimFactorAlign);
    tilingData.set_ubRDimLoopNum(ubRDimLoopNum);
    tilingData.set_ubRDimTail(ubRDimTail);
    tilingData.set_ubRDimTailFactor(ubRDimTailFactor);
    tilingData.set_ubRDimTailFactorAlign(ubRDimTailFactorAlign);
    tilingData.set_ubRDimTailLoopNum(ubRDimTailLoopNum);
    tilingData.set_ubRDimTailTail(ubRDimTailTail);
    tilingData.set_ubRDimTailTailFactor(ubRDimTailTailFactor);
    tilingData.set_ubRDimTailTailFactorAlign(ubRDimTailTailFactorAlign);
    tilingData.set_ubRDimTailTailLoopNum(ubRDimTailTailLoopNum);
}

ge::graphStatus BatchNormGradV3RARRecomputeTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradV3TilingBase::DoOpTiling();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "failed to do base tiling."),
                    return ge::GRAPH_FAILED);

    if (subTilingKey == BNG_V3_RAR_RECOMPUTE_SPLIT_R1) {
        DoRecomputeTilingSplitR1();
    } else {
        DoRecomputeTilingSplitR0();
    }

    int64_t tailBinAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimTailFactor)));
    int64_t generalBinAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimFactor)));

    BatchNormGradV3TilingBase::DoBinaryAddTiling(tilingData.generalBinAddTilingData, generalBinAddQuotient);
    BatchNormGradV3TilingBase::DoBinaryAddTiling(tilingData.tailBinAddTilingData, tailBinAddQuotient);

    OP_LOGI(context_->GetNodeName(),
            "BatchNormGradV3 do tiling finish. r1: %ld a: %ld r0: %ld rAlign: %ld "
            "block: %ld tailBlock: %ld formerBlockDim: %ld tailBlockDIm: %ld "
            "rDimFactor: %ld rDimFactorAlign: %ld rDimLoopNum: %ld "
            "rDimTail: %ld rDimTailFactor: %ld rDimTailFactorAlign: %ld rDimTailLoopNum: %ld "
            "rDimTailTail: %ld rDimTailTailFactor: %ld rDimTailTailFactorAlign: %ld rDimTailTailLoopNum: %ld",
            r1Dim, aDim, r0Dim, rAlign, baseTilingData.get_blockNum(), baseTilingData.get_tailBlockNum(),
            baseTilingData.get_formerBlockDim(), baseTilingData.get_tailBlockDim(), ubRDimFactor, ubRDimFactorAlign,
            ubRDimLoopNum, ubRDimTail, ubRDimTailFactor, ubRDimTailFactorAlign, ubRDimTailLoopNum, ubRDimTailTail,
            ubRDimTailTailFactor, ubRDimTailTailFactorAlign, ubRDimTailTailLoopNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3RARRecomputeTilingBase::GetTilingKey() const
{
    if (dyDtype != ge::DataType::DT_FLOAT && dyDtype != ge::DataType::DT_FLOAT16 && dyDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyDtype).c_str(),
                                              "The dtype of dy must be FLOAT, FLOAT16 or BF16.");
        return BNG_V3_TK_DEFAULT;
    }
    uint64_t tilingKeyValue = BNG_V3_RAR_RECOMPUTE_TK_BASE;
    tilingKeyValue += subTilingKey;
    OP_LOGD(context_->GetNodeName(), "BatchNormGradV3 get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradV3RARRecomputeTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3RARRecomputeTilingBase::PostTiling()
{
    uint64_t curTilingKey = GetTilingKey();
    OP_TILING_CHECK(curTilingKey == BNG_V3_TK_DEFAULT,
                    ReportTilingStageError(
                        context_->GetNodeName(),
                        "PostTiling: tiling key is 0 (default); the template failed to produce a valid tiling key"),
                    return ge::GRAPH_FAILED);
    context_->SetTilingKey(curTilingKey);
    context_->SetBlockDim(baseTilingData.get_blockNum());
    size_t* curWorkspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(curWorkspaces == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "PostTiling: GetWorkspaceSizes(1) returned null; required output-slot count is "
                                       "1. Check the TilingContext construction, not the user device allocation"),
                return ge::GRAPH_FAILED);
    curWorkspaces[0] = workspaceSize_;
    tilingData.set_outputMask(outputMask_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

int64_t BatchNormGradV3TilingBase::CalcRAFullLoadLoopFactor(int64_t blockFactor) const
{
    if (blockFactor <= 0 || r1Dim <= 0 || blockSize == 0 || vlFp32 == 0) {
        return 0;
    }
    // Even the smallest tile needs three double-buffered R-by-block queues.
    // Reject huge R before multiplying it in the detailed capacity formula.
    if (static_cast<uint64_t>(r1Dim) > ubSize / blockSize / (DOUBLE_BUFFER * THREE)) {
        return 0;
    }
    int64_t power2k = r1Dim <= 1 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim - 1)));
    int64_t binaryRows = r1Dim > RA_FULL_LOAD_UNROLLED_ROWS ? power2k / TWO : 0;
    int64_t aligned = 0;
    return CalcRAFullLoadALoopFactor(blockFactor, r1Dim, binaryRows, ge::GetSizeByDataType(dyDtype),
                                     ge::GetSizeByDataType(weightDtype), blockSize, ubSize, aligned, vlFp32);
}

bool BatchNormGradV3TilingBase::PreferVectorRAFullLoad() const
{
    // Wider aligned half-precision rows can use the resident vector layout
    // directly, without the packed channel fold.
    constexpr int64_t quarterVectorDivisor = 4;
    int64_t dtypeSize = ge::GetSizeByDataType(dyDtype);
    if (r0Dim != 1 || aDim <= static_cast<int64_t>(vlFp32) / quarterVectorDivisor || aDim > RA_PACKED_MAX_CHANNELS ||
        dtypeSize != FLOAT16_BYTE_NUMBER || blockSize == 0 || vlFp32 == 0 ||
        static_cast<uint64_t>(aDim * dtypeSize) % blockSize != 0) {
        return false;
    }
    return CalcRAFullLoadLoopFactor(vlFp32) == static_cast<int64_t>(vlFp32);
}

bool BatchNormGradV3TilingBase::CanUsePackedRAFullLoad(int64_t& groups) const
{
    groups = 0;
    if (r0Dim != 1 || aDim < 1 || aDim > RA_PACKED_MAX_CHANNELS || r1Dim < RA_PACKED_MIN_REDUCE ||
        vlFp32 != BASIC_FACTOR || blockSize == 0) {
        return false;
    }
    int64_t rowsPerGroup = 1;
    while (rowsPerGroup * TWO * aDim <= static_cast<int64_t>(vlFp32)) {
        rowsPerGroup *= TWO;
    }
    int64_t packedGroups = Ops::Base::CeilDiv(r1Dim, rowsPerGroup);
    int64_t dtypeSize = ge::GetSizeByDataType(dyDtype);
    // Feature queues alone bound the group count before scratch arithmetic.
    // The exact UB budget below decides the upper limit for every row width.
    if (dtypeSize <= 0 || static_cast<uint64_t>(packedGroups) >
                              ubSize / vlFp32 / static_cast<uint64_t>(dtypeSize) / (DOUBLE_BUFFER * THREE)) {
        return false;
    }
    int64_t power2k = packedGroups <= 1 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(packedGroups - 1)));
    int64_t binaryRows = packedGroups > RA_FULL_LOAD_UNROLLED_ROWS ? power2k / TWO : 0;
    int64_t aligned = 0;
    // Match the packed kernel: one full vector per group, double-buffered
    // feature/parameter queues and two FP32 reduction scratch tensors.
    if (CalcRAFullLoadALoopFactor(vlFp32, packedGroups, binaryRows, ge::GetSizeByDataType(dyDtype),
                                  ge::GetSizeByDataType(weightDtype), blockSize, ubSize,
                                  aligned) != static_cast<int64_t>(vlFp32)) {
        return false;
    }
    groups = packedGroups;
    return true;
}

bool BatchNormGradV3RAFullLoadTilingBase::IsCapable()
{
    if (r0Dim != 1 || aDim <= 0 || r1Dim <= 0 || coreNum == 0 || blockSize == 0 || vlFp32 == 0) {
        return false;
    }
    packed_ = CanUsePackedRAFullLoad(packedGroups_) && !PreferVectorRAFullLoad();
    if (packed_) {
        return true;
    }
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t aBlockFactor = std::max(Ops::Base::CeilDiv(aDim, static_cast<int64_t>(coreNum)),
                                    static_cast<int64_t>(blockSize) / dtypeSize);
    int64_t numBlocks = Ops::Base::CeilDiv(aDim, aBlockFactor);
    aBlockFactor = Ops::Base::CeilDiv(aDim, numBlocks);
    int64_t aLoopFactor = CalcRAFullLoadLoopFactor(aBlockFactor);
    if (aLoopFactor <= 0) {
        return false; // Even the smallest aligned C tile cannot keep all R rows in UB.
    }
    if (aLoopFactor >= aBlockFactor) {
        return true; // The entire per-core C block fits in one FullLoad iteration.
    }
    // Use a coarse vector-width threshold when the core's C block is not resident.
    // Recompute can trade shorter R tiles for wider C tiles below this threshold.
    if (aLoopFactor >= static_cast<int64_t>(vlFp32)) {
        return true;
    }
    return false;
}

ge::graphStatus BatchNormGradV3RAFullLoadTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradV3TilingBase::DoOpTiling();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "failed to do base tiling."),
                    return ge::GRAPH_FAILED);

    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t weightDtypeSize = weightDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    // IsCapable already selected the implementation and its packed group count.
    if (packed_) {
        tilingData.baseTilingData.set_blockNum(1);
        tilingData.baseTilingData.set_formerBlockDim(aDim);
        tilingData.baseTilingData.set_tailBlockNum(0);
        tilingData.baseTilingData.set_tailBlockDim(0);
        tilingData.set_numBlocks(1);
        tilingData.set_mainBlockFactor(aDim);
        tilingData.set_tailBlockFactor(0);
        tilingData.set_mainBlockCount(1);
        tilingData.set_tailBlockCount(0);
        tilingData.set_mainALoopFactor(aDim);
        tilingData.set_mainALoopFactorAligned(vlFp32);
        tilingData.set_tailALoopFactor(aDim);
        tilingData.set_tailALoopFactorAligned(vlFp32);
        int64_t power2k = packedGroups_ <= 1 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(packedGroups_ - 1)));
        SetReduceLoopTimes(power2k, packedGroups_);
        return ge::GRAPH_SUCCESS;
    }
    int64_t rDim_ = tilingData.baseTilingData.get_r1Dim();
    // Match the original full-load power calculation; handle R<=1 to avoid clzl(0).
    int64_t power2k = (rDim_ <= 1) ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(rDim_ - 1)));

    OP_TILING_CHECK(!CalculateBlockFactors(dtypeSize, weightDtypeSize, power2k),
                    ReportTilingStageError(context_->GetNodeName(),
                                           "DoOpTiling: CalculateBlockFactors failed to set RA block factors for the "
                                           "current shape and platform; tiling strategy failed for a valid input"),
                    return ge::GRAPH_FAILED);
    CalculateLoopFactors(dtypeSize, weightDtypeSize, rDim_, power2k);
    SetReduceLoopTimes(power2k, rDim_);

    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradV3RAFullLoadTilingBase::CalculateBlockFactors(int64_t dtypeSize, int64_t weightDtypeSize,
                                                                int64_t power2k)
{
    vectorCorePartition_ = false;
    if (!SetRABlockFactorsCommon(tilingData, aDim, dtypeSize, coreNum, blockSize)) {
        return false;
    }
    if (tilingData.get_mainBlockFactor() >= static_cast<int64_t>(vlFp32)) {
        return true;
    }
    // Long strided copies with an unaligned GM row pitch lose parallelism
    // when merged. Keep their original partition; R<=4 uses the unrolled path.
    if (r1Dim > RA_FULL_LOAD_UNROLLED_ROWS && static_cast<uint64_t>(aDim * dtypeSize) % blockSize != 0) {
        return true;
    }
    // Size the core count for one FP32 vector: 256B for FP32 inputs, 128B for
    // FP16/BF16 inputs. Keep wider existing blocks and retain their parallelism.
    int64_t channels = static_cast<int64_t>(vlFp32);
    int64_t binaryRows = r1Dim > RA_FULL_LOAD_UNROLLED_ROWS ? power2k / TWO : 0;
    int64_t aligned = 0;
    // Merge narrow blocks only when the entire R-by-C tile fits in one UB loop.
    if (CalcRAFullLoadALoopFactor(channels, r1Dim, binaryRows, dtypeSize, weightDtypeSize, blockSize, ubSize,
                                  aligned) != channels) {
        return true;
    }
    int64_t numBlocks = Ops::Base::CeilDiv(aDim, channels);
    if (numBlocks >= tilingData.get_numBlocks()) {
        return true;
    }
    // Balance C without misaligning the GM offsets between cores. An aligned
    // row pitch also keeps the final block at least one data block wide.
    channels = std::min(
        aDim, Ops::Base::CeilAlign(Ops::Base::CeilDiv(aDim, numBlocks), static_cast<int64_t>(blockSize / dtypeSize)));
    int64_t mainBlocks = aDim / channels;
    int64_t tailChannels = aDim % channels;
    int64_t tailBlocks = tailChannels > 0 ? 1 : 0;
    tilingData.set_numBlocks(mainBlocks + tailBlocks);
    tilingData.set_mainBlockFactor(channels);
    tilingData.set_tailBlockFactor(tailChannels);
    tilingData.set_mainBlockCount(mainBlocks);
    tilingData.set_tailBlockCount(tailBlocks);
    vectorCorePartition_ = true;
    return true;
}

void BatchNormGradV3RAFullLoadTilingBase::CalculateLoopFactors(int64_t dtypeSize, int64_t weightDtypeSize,
                                                               int64_t rDim_, int64_t power2k)
{
    int64_t mainALoopFactorAligned = 0;
    int64_t tailALoopFactorAligned = 0;
    int64_t binaryRows = rDim_ > RA_FULL_LOAD_UNROLLED_ROWS ? power2k / TWO : 0;

    int64_t mainALoopFactor = CalcRAFullLoadALoopFactor(tilingData.get_mainBlockFactor(), rDim_, binaryRows, dtypeSize,
                                                        weightDtypeSize, blockSize, ubSize, mainALoopFactorAligned,
                                                        vlFp32);
    int64_t tailALoopFactor = CalcRAFullLoadALoopFactor(tilingData.get_tailBlockFactor(), rDim_, binaryRows, dtypeSize,
                                                        weightDtypeSize, blockSize, ubSize, tailALoopFactorAligned,
                                                        vlFp32);

    if (vectorCorePartition_) {
        // CalculateBlockFactors already checked the full-vector UB budget.
        // Keep the reduction/VF row stride aligned even for partial vectors.
        mainALoopFactorAligned = vlFp32;
        tailALoopFactorAligned = tailALoopFactor > 0 ? vlFp32 : 0;
    }

    tilingData.set_mainALoopFactor(mainALoopFactor);
    tilingData.set_mainALoopFactorAligned(mainALoopFactorAligned);
    tilingData.set_tailALoopFactor(tailALoopFactor);
    tilingData.set_tailALoopFactorAligned(tailALoopFactorAligned);
}

void BatchNormGradV3RAFullLoadTilingBase::SetReduceLoopTimes(int64_t power2k, int64_t rDim_)
{
    int64_t reduceLoopTimes = power2k / TWO;
    int64_t reduceRecursionLoop = reduceLoopTimes > 1 ?
                                      static_cast<int64_t>(ULONG_BIT_LEN - __builtin_clzl(reduceLoopTimes)) - TWO :
                                      0;
    tilingData.set_reduceLoopTimes(reduceLoopTimes);
    tilingData.set_reduceRecursionLoop(reduceRecursionLoop);

    // foldLoop
    int64_t foldLoopStep1 = (rDim_ - power2k) / TWO;
    int64_t foldLoopStep2 = rDim_ > 1 ? rDim_ % TWO : 0;
    int64_t foldLoopStep3 = std::max(reduceLoopTimes - foldLoopStep1 - foldLoopStep2, 0L);
    tilingData.set_foldLoopStep1(foldLoopStep1);
    tilingData.set_foldLoopStep2(foldLoopStep2);
    tilingData.set_foldLoopStep3(foldLoopStep3);

    // foldLoop Offset
    tilingData.set_foldLoopOffset1(power2k);
    tilingData.set_foldLoopOffset2(foldLoopStep1 * TWO);
    tilingData.set_foldLoopOffset3((foldLoopStep1 + foldLoopStep2) * TWO);
}

uint64_t BatchNormGradV3RAFullLoadTilingBase::GetTilingKey() const
{
    if (dyDtype != ge::DataType::DT_FLOAT && dyDtype != ge::DataType::DT_FLOAT16 && dyDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyDtype).c_str(),
                                              "The dtype of dy must be FLOAT, FLOAT16 or BF16.");
        return BNG_V3_TK_DEFAULT;
    }
    uint64_t tilingKeyValue = packed_ ? BNG_V3_RA_PACKED_ALL_LOAD_TK_BASE : BNG_V3_RA_ALL_LOAD_TK_BASE;
    OP_LOGD(context_->GetNodeName(), "BatchNormGradV3 get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradV3RAFullLoadTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3RAFullLoadTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData.get_numBlocks());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "PostTiling: GetWorkspaceSizes(1) returned null; required output-slot count is "
                                       "1. Check the TilingContext construction, not the user device allocation"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData.set_outputMask(outputMask_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradV3RARecomputeTilingBase::IsCapable()
{
    // The recompute cache requires more than one R tile. Short reductions
    // stay on FullLoad, whose in-register path also avoids the cache overhead.
    if (r0Dim == 1 && r1Dim > BASIC_FACTOR) {
        return true;
    }
    return false;
}

ge::graphStatus BatchNormGradV3RARecomputeTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradV3TilingBase::DoOpTiling();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "failed to do base tiling."),
                    return ge::GRAPH_FAILED);
    int64_t aLoopFactor = BASIC_FACTOR;
    int64_t rLoopFactor = BASIC_FACTOR;
    int64_t aLoopFactorAligned = BASIC_FACTOR;

    // block tiling
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t rDim_ = tilingData.baseTilingData.get_r1Dim();
    int64_t aDim_ = tilingData.baseTilingData.get_aDim();
    OP_TILING_CHECK(
        !SetRABlockFactorsCommon(tilingData, aDim_, dtypeSize, coreNum, blockSize),
        ReportTilingStageError(context_->GetNodeName(), "DoOpTiling: SetRABlockFactorsCommon failed for the current "
                                                        "shape and platform; tiling strategy failed for a valid input"),
        return ge::GRAPH_FAILED);

    // Start with the UB-selected tile, then refine it if shared-L2 capacity is known.
    CalcRARecomputeLoopFactors(tilingData.get_mainBlockFactor(), rDim_, ubSize, aLoopFactor, aLoopFactorAligned,
                               rLoopFactor);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        uint64_t l2Size = 0;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
        CalcRARecomputeCacheFactors(tilingData.get_mainBlockFactor(), rDim_, aDim_, dtypeSize,
                                    tilingData.get_numBlocks(), ubSize, l2Size, aLoopFactor, aLoopFactorAligned,
                                    rLoopFactor);
    }
    // UB capacity remains a hard constraint after the cache-based refinement.
    OP_TILING_CHECK(aLoopFactorAligned * RARecomputeBytesPerChannel(rDim_, rLoopFactor) > static_cast<int64_t>(ubSize),
                    ReportTilingStageError(context_->GetNodeName(), "RA recompute UB budget exceeded"),
                    return ge::GRAPH_FAILED);

    // ub tiling
    int64_t rLoopTimes = rDim_ / rLoopFactor;
    int64_t rLoopTail = rDim_ - rLoopTimes * rLoopFactor;
    tilingData.set_aLoopFactor(aLoopFactor);
    tilingData.set_aLoopFactorAligned(aLoopFactorAligned);
    tilingData.set_rLoopFactor(rLoopFactor);
    tilingData.set_rLoopTimes(rLoopTimes);
    tilingData.set_rLoopTail(rLoopTail);

    // ub binary add
    int64_t binaryBlockCount = (rDim_ + rLoopFactor - 1) / rLoopFactor;
    int64_t binaryTailBlock = rDim_ - (rDim_ / rLoopFactor) * rLoopFactor;
    int64_t binaryFoldPoint = (binaryBlockCount <= 1) ?
                                  0 :
                                  1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binaryBlockCount - 1));
    int64_t cacheBufferCount = ULONG_BIT_LEN - __builtin_clzl(binaryFoldPoint);
    tilingData.set_binaryFoldPoint(binaryFoldPoint);
    tilingData.set_binaryBlockCount(binaryBlockCount);
    tilingData.set_binaryTailBlock(binaryTailBlock);
    tilingData.set_cacheBufferCount(cacheBufferCount);

    float reciprocal = 1.0 / rDim_;
    tilingData.set_reciprocal(reciprocal);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3RARecomputeTilingBase::GetTilingKey() const
{
    if (dyDtype != ge::DataType::DT_FLOAT && dyDtype != ge::DataType::DT_FLOAT16 && dyDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyDtype).c_str(),
                                              "The dtype of dy must be FLOAT, FLOAT16 or BF16.");
        return BNG_V3_TK_DEFAULT;
    }
    uint64_t tilingKeyValue = BNG_V3_RA_RECOMPUTE_TK_BASE;
    OP_LOGD(context_->GetNodeName(), "BatchNormGradV3 get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradV3RARecomputeTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3RARecomputeTilingBase::PostTiling()
{
    uint64_t raTilingKey = GetTilingKey();
    context_->SetTilingKey(raTilingKey);
    context_->SetBlockDim(tilingData.get_numBlocks());
    size_t* raWorkspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(raWorkspaces == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "PostTiling: GetWorkspaceSizes(1) returned null; required output-slot count is "
                                       "1. Check the TilingContext construction, not the user device allocation"),
                return ge::GRAPH_FAILED);
    raWorkspaces[0] = workspaceSize_;
    tilingData.set_outputMask(outputMask_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForBatchNormGradV3(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForBatchNormGradV3(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(
        context == nullptr,
        ReportTilingStageError("BatchNormGradV3",
                               "TilingPrepare: TilingParseContext is null; the tiling framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForBatchNormGradV3 enter.");

    auto compileInfo = context->GetCompiledInfo<BatchNormGradV3CompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "TilingPrepare: compile info is null; check the SoC-specific OPP package and the tiling parse context"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: platform info is null; check the platform info query and the OPP/SoC installation"),
                return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vlFp32 = Ops::Base::GetVRegSize(context) / sizeof(float);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        (compileInfo->coreNum <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid core num " + std::to_string(compileInfo->coreNum) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_TILING_CHECK(
        (compileInfo->ubSize <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid UB size " + std::to_string(compileInfo->ubSize) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepareForBatchNormGradV3 exit, ubSize: %ld.", ubSize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3RARFullLoadTilingBase, BNG_V3_RAR_ALL_LOAD_TK_BASE);
REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3RAFullLoadTilingBase, BNG_V3_RA_ALL_LOAD_TK_BASE);
REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3RARRecomputeTilingBase, BNG_V3_RAR_RECOMPUTE_TK_BASE);
REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3RARecomputeTilingBase, BNG_V3_RA_RECOMPUTE_TK_BASE);

IMPL_OP_OPTILING(BatchNormGradV3)
    .Tiling(TilingForBatchNormGradV3)
    .TilingParse<BatchNormGradV3CompileInfo>(TilingPrepareForBatchNormGradV3);
} // namespace optiling
