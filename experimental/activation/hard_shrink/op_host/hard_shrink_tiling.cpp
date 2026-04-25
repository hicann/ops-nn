/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_shrink_tiling.cpp
 * \brief HardShrink 算子 Tiling 实现
 *
 * Tiling 策略：
 *   - 多核切分：totalNum / coreNum
 *   - UB 切分：ubSize / bufferNum / typeSize，256B 对齐
 *   - 双缓冲阈值：totalLength > 1024 → BUFFER_MODE=1
 *   - bf16 处理：TilingKey 用 D_T=float + IS_BF16=1
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <cmath>
#include "../op_kernel/hard_shrink_tiling_data.h"
#include "../op_kernel/hard_shrink_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_RESERVED_BYTE = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape) {
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

struct HardShrinkPlatInfo {
    int64_t coreNum;
    uint64_t ubSize;
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, HardShrinkPlatInfo& info)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    info.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(info.coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    OP_CHECK_IF(info.ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct HardShrinkInputInfo {
    int64_t totalNum;
    ge::DataType dataType;
};

static ge::graphStatus GetInputInfo(gert::TilingContext* context, HardShrinkInputInfo& info)
{
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShape = EnsureNotScalar(inputSelf->GetStorageShape());
    info.totalNum = inputShape.GetShapeSize();
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    info.dataType = inputDesc->GetDataType();
    if (info.dataType != ge::DT_FLOAT16 && info.dataType != ge::DT_FLOAT && info.dataType != ge::DT_BF16) {
        OP_LOGE(context, "HardShrink: unsupported dtype=%d, expected FLOAT16/FLOAT/BF16",
                static_cast<int>(info.dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static float GetLambdAttr(gert::TilingContext* context)
{
    float lambd = 0.5f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* lambdPtr = attrs->GetFloat(0);
        if (lambdPtr != nullptr) {
            lambd = *lambdPtr;
        }
    }
    if (std::isnan(lambd) || std::isinf(lambd)) {
        OP_LOGW(context, "lambd is special value (nan/inf): %f, output may be all zeros", lambd);
    }
    return lambd;
}

static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, ge::DataType dataType)
{
    context->SetBlockDim(1);
    HardShrinkTilingData* tiling = context->GetTilingData<HardShrinkTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardShrinkTilingData), 0, sizeof(HardShrinkTilingData)) != EOK,
        OP_LOGE(context, "memset_s failed"),
        return ge::GRAPH_FAILED);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    uint32_t dType = static_cast<uint32_t>(dataType);
    uint32_t bufferMode = 0;
    uint32_t isBf16 = (dataType == ge::DT_BF16) ? 1 : 0;
    ASCENDC_TPL_SEL_PARAM(context, dType, bufferMode, isBf16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcUbFactor(gert::TilingContext* context, uint64_t ubSize,
                                     bool isBf16, int64_t computeTypeSize,
                                     int64_t BM, int64_t alignElems, int64_t& ubFactor)
{
    int64_t mainBufBytesPerElem;
    if (isBf16) {
        // bf16: inputQueue(bf16)*BM + outputQueue(bf16)*BM + lambdBuf(f32) + negLambdBuf(f32) + tmpBuf(f32) + tmp2Buf(f32)
        mainBufBytesPerElem = 2 * 2 * BM + computeTypeSize * 4;
    } else {
        // fp16/fp32: inputQueue*BM + outputQueue*BM + lambdBuf + negLambdBuf + tmpBuf
        mainBufBytesPerElem = computeTypeSize * 2 * BM + computeTypeSize * 3;
    }
    constexpr uint64_t MAX_UB_SIZE = 512 * 1024 * 1024ULL;
    OP_CHECK_IF(ubSize > MAX_UB_SIZE,
        OP_LOGE(context, "ubSize=%lu exceeds max expected size %lu", ubSize, MAX_UB_SIZE),
        return ge::GRAPH_FAILED);
    int64_t ubFactorRaw = ((static_cast<int64_t>(ubSize) - UB_RESERVED_BYTE) * 8) / (mainBufBytesPerElem * 8 + 1);
    ubFactor = FloorAlign(ubFactorRaw, alignElems);
    OP_CHECK_IF(ubFactor <= 0,
        OP_LOGE(context, "ubFactor is %ld, invalid (ubSize=%lu, UB_RESERVED_BYTE=%ld)",
                ubFactor, ubSize, UB_RESERVED_BYTE),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardShrinkTilingFunc(gert::TilingContext* context)
{
    HardShrinkPlatInfo platInfo;
    auto ret = GetPlatformInfo(context, platInfo);
    if (ret != ge::GRAPH_SUCCESS) { return ret; }

    HardShrinkInputInfo inputInfo;
    ret = GetInputInfo(context, inputInfo);
    if (ret != ge::GRAPH_SUCCESS) { return ret; }

    float lambd = GetLambdAttr(context);

    if (inputInfo.totalNum == 0) {
        return HandleEmptyTensor(context, inputInfo.dataType);
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    bool isBf16 = (inputInfo.dataType == ge::DT_BF16);
    int64_t computeTypeSize = (isBf16 || inputInfo.dataType == ge::DT_FLOAT) ? 4 : 2;

    HardShrinkTilingData* tiling = context->GetTilingData<HardShrinkTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardShrinkTilingData), 0, sizeof(HardShrinkTilingData)) != EOK,
        OP_LOGE(context, "memset_s failed"),
        return ge::GRAPH_FAILED);

    tiling->totalNum = inputInfo.totalNum;
    tiling->blockFactor = CeilDiv(inputInfo.totalNum, platInfo.coreNum);
    int64_t usedCoreNum = CeilDiv(inputInfo.totalNum, tiling->blockFactor);

    uint32_t bufferMode = (inputInfo.totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t BM = bufferMode ? 2 : 1;
    int64_t alignElems = 256 / computeTypeSize;

    int64_t ubFactor;
    ret = CalcUbFactor(context, platInfo.ubSize, isBf16, computeTypeSize, BM, alignElems, ubFactor);
    if (ret != ge::GRAPH_SUCCESS) { return ret; }
    tiling->ubFactor = ubFactor;
    tiling->lambd = lambd;

    context->SetBlockDim(usedCoreNum);

    uint32_t dType = static_cast<uint32_t>(inputInfo.dataType);
    uint32_t isBf16Flag = isBf16 ? 1 : 0;
    ASCENDC_TPL_SEL_PARAM(context, dType, bufferMode, isBf16Flag);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardShrink([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardShrinkCompileInfo {};

IMPL_OP_OPTILING(HardShrink)
    .Tiling(HardShrinkTilingFunc)
    .TilingParse<HardShrinkCompileInfo>(TilingParseForHardShrink);

} // namespace optiling
