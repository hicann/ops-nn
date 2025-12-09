/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file rms_norm_quant_common_tiling.cpp
 * \brief
 */
#include <climits>
#include "op_common/log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "rms_norm_quant_common_tiling.h"

namespace optiling {
// const defination
static constexpr uint32_t BYTE_32 = 32;

static constexpr uint32_t SLICE_SIZE = 8192;          // 用于切分过长的行
static constexpr uint32_t SLICE_SIZE_WITH_BIAS = 8192; // 用于切分过长的行
static constexpr uint32_t FP16_PER_BLOCK = 16;
static constexpr size_t WORKAPCE_RESERVE = 16 * 1024 * 1024;
static constexpr uint32_t BIT_OFFSET_SIX = 6;
// enum defination
enum class PrePostType : uint32_t
{
    Norm = 0,
    PreNorm = 1,
    PostNorm = 2
};

enum class NormType : uint32_t
{ // 没用到
    LayerNorm = 0,
    RmsNorm = 1
};

// function defination
/**
 * Tiling Key Defination
 *
 *
 * |gemma|beta|precision|fast/slice|dtype|
 * |-----|----|---------|----------|-----|
 * |  1  | 1  |    1    |    1     |  6  |
 * - dtype use origin ACL/ge...values.
 */
uint32_t CalcNormTilingKey(NormCommonTilingData1* tilingDataPtr, ge::DataType dtype, bool useBeta)
{
    uint32_t tilingKey = static_cast<uint32_t>(tilingDataPtr->get_gemmaMode() ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useBeta ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(tilingDataPtr->get_highPrecisionMode() ? 1 : 0);
    bool useSlice = tilingDataPtr->get_numCol() > tilingDataPtr->get_sliceSize();
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useSlice ? 1 : 0);
    tilingKey = (tilingKey << BIT_OFFSET_SIX) + static_cast<uint32_t>(dtype);
    return tilingKey;
};

static ge::graphStatus PrepareForRmsNormCommonTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NormCommonCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNumAiv <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm <= 0, OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    compileInfo->ubSize = ubSizePlatForm;

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQuantAttrs([[maybe_unused]]gert::TilingContext* context, NormCommonTilingData1* tilingDataPtr)
{
    const char* quantMinFlagPtr = std::getenv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinFlagPtr != nullptr && strcmp(quantMinFlagPtr, "1") == 0) {
        tilingDataPtr->set_quantMin(-127); // set int8 min to -127
    } else {
        tilingDataPtr->set_quantMin(std::numeric_limits<int8_t>::min());
    }
    tilingDataPtr->set_scale(1.0f);
    tilingDataPtr->set_offset(0);
    return ge::GRAPH_SUCCESS;
}

void LogTilingData(const std::string& opName, NormCommonTilingData1* tilingDataPtr)
{
    std::stringstream tilingDataLogStringSS;
    tilingDataLogStringSS << opName << " Tiling Data: "
                          << " numCore " << tilingDataPtr->get_numCore() << " numCol " << tilingDataPtr->get_numCol()
                          << " numRow " << tilingDataPtr->get_numRow() << " avgFactor "
                          << tilingDataPtr->get_avgFactor() << " epsilon " << tilingDataPtr->get_epsilon()
                          << " sliceSize " << tilingDataPtr->get_sliceSize() << " precisionMode "
                          << tilingDataPtr->get_highPrecisionMode();
    OP_LOGD(opName, "%s", tilingDataLogStringSS.str().c_str());
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return Ops::Base::CeilDiv(a, b) * b;
}

// template defination and implementation
template <bool EN_QUANT>
static uint32_t CalcMaxSliceSize(PrePostType prePostType, bool enBias, const uint32_t ubSize)
{
    if (prePostType == PrePostType::Norm) {
        if (EN_QUANT) {
            return SLICE_SIZE;
        } else {
            return SLICE_SIZE;
        }
    }
    if (prePostType == PrePostType::PreNorm) {
        if (EN_QUANT) {
            return SLICE_SIZE;
        } else {
            return enBias ? SLICE_SIZE_WITH_BIAS : SLICE_SIZE;
        }
    }
    if (prePostType == PrePostType::PostNorm) {
        if (EN_QUANT) {
            static constexpr uint32_t TBUF_NUM = 21;
            return RoundUp(ubSize / TBUF_NUM, BYTE_32);
        } else {
            return enBias ? SLICE_SIZE : SLICE_SIZE_WITH_BIAS;
        }
    }
    return SLICE_SIZE;
}

template <bool EN_QUANT>
static ge::graphStatus NormCommonTiling(gert::TilingContext* context, PrePostType prePostType)
{
    // NormCommonTilingData * tilingDataPtr = context->GetTilingData<NormCommonTilingData>(); // 使用这个API有地址错误
    NormCommonTilingData1 tilingData;

    // prepare common data
    auto compileInfo = static_cast<const NormCommonCompileInfo*>(context->GetCompileInfo());
    uint32_t coreNumAiv = compileInfo->coreNumAiv;
    // get op name
    std::string opName = context->GetNodeName();
    // get x shape
    gert::Shape xShape = context->GetInputShape(static_cast<int>(NormInputIndex::X))->GetOriginShape();

    // calculate numRow
    int64_t numRow = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        uint32_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim <= 0, OP_LOGD(opName, "In tensor X dim %d is invalid", i), return ge::GRAPH_PARAM_OUT_OF_RANGE);
        OP_CHECK_IF(
            numRow > std::numeric_limits<uint32_t>::max() / dim,
            OP_LOGD(opName, "At dim %d, tmpNumRow is overflowed", i), return ge::GRAPH_FAILED);
        numRow *= dim;
    }
    tilingData.set_numRow(static_cast<uint32_t>(numRow)); // 修改
    // check numCol
    uint32_t numCol = xShape.GetDim(xShape.GetDimNum() - 1);
    OP_CHECK_IF(
        numCol > std::numeric_limits<uint32_t>::max(), OP_LOGD(opName, "numCol is overflowed"),
        return ge::GRAPH_PARAM_OUT_OF_RANGE);
    tilingData.set_numCol(numCol);
    // calculate numCore
    tilingData.set_numCore(
        Ops::Base::CeilDiv(tilingData.get_numRow(), Ops::Base::CeilDiv(tilingData.get_numRow(), coreNumAiv)));
    OP_CHECK_IF(
        tilingData.get_numCore() > std::numeric_limits<uint32_t>::max() - tilingData.get_numRow(),
        OP_LOGD(opName, "numRow + numCore is overflowed"), return ge::GRAPH_PARAM_OUT_OF_RANGE);
    // calculate slice size
    auto betaTensorDesc = context->GetInputDesc(static_cast<int>(NormInputIndex::BETA));
    bool enableBeta = betaTensorDesc != nullptr;
    if (enableBeta) {
        enableBeta = context->GetInputShape(static_cast<int>(NormInputIndex::BETA))->GetOriginShape().GetDimNum() != 0;
    }

    const uint32_t maxSliceSize = CalcMaxSliceSize<EN_QUANT>(prePostType, enableBeta, compileInfo->ubSize);
    tilingData.set_sliceSize(maxSliceSize);
    OP_CHECK_IF(
        tilingData.get_numCol() > std::numeric_limits<uint32_t>::max() - tilingData.get_sliceSize(),
        OP_LOGD(opName, "numCol + sliceSize is overflowed"), return ge::GRAPH_PARAM_OUT_OF_RANGE);
    // calculate avgFactor
    OP_CHECK_IF(numCol <= 0, OP_LOGE(opName, "The numCol size is neg: %ud.", numCol), return ge::GRAPH_PARAM_INVALID);
    float avgFactor = static_cast<float>(1.0 / numCol);
    tilingData.set_avgFactor(avgFactor);
    // get attrs
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    // check epsilon
    const float* epsilon = attrs->GetFloat(static_cast<int>(NormAttrIndex::EPSILON));
    OP_CHECK_IF(*epsilon <= 0, OP_LOGD(opName, "epsilon should be positive"), return ge::GRAPH_PARAM_INVALID);
    tilingData.set_epsilon(*epsilon);
    // set high precision mode
    const bool* highPrecisionMode = attrs->GetBool(static_cast<int>(NormAttrIndex::HIGH_PRECISION_MODE));

    tilingData.set_highPrecisionMode(highPrecisionMode && *highPrecisionMode);
    // set gemma mode
    const bool* gemmaMode = attrs->GetBool(static_cast<int>(NormAttrIndex::GEMMA_MODE));
    tilingData.set_gemmaMode(gemmaMode && *gemmaMode);

    if (EN_QUANT) { // if constexpr (EN_QUANT) 会有编译错误
        GetQuantAttrs(context, &tilingData);
    }

    ge::DataType dtype = context->GetInputDesc(static_cast<int>(NormInputIndex::X))->GetDataType();

    if (prePostType != PrePostType::Norm) {
        OP_CHECK_IF(
            tilingData.get_numCol() > std::numeric_limits<uint32_t>::max() - FP16_PER_BLOCK,
            OP_LOGD(opName, "numCol + FP16_PER_BLOCK is overflowed"), return ge::GRAPH_PARAM_OUT_OF_RANGE);
    }

    uint32_t tilingKey = CalcNormTilingKey(&tilingData, dtype, enableBeta);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKAPCE_RESERVE;

    OP_LOGD(opName, "tiling key is %d", tilingKey);
    context->SetBlockDim(tilingData.get_numCore());
    context->SetTilingKey(tilingKey);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    LogTilingData(opName, &tilingData);
    return ge::GRAPH_SUCCESS;
}

template <bool EN_QUANT>
static ge::graphStatus RmsNormTilingFunc(gert::TilingContext* context)
{
    std::string opName = context->GetNodeName();
    auto ret = NormCommonTiling<EN_QUANT>(context, PrePostType::Norm);
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(opName, "RmsNormTilingFunc running error!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <bool EN_QUANT>
static ge::graphStatus PreRmsNormTilingFunc(gert::TilingContext* context)
{
    std::string opName = context->GetNodeName();
    auto ret = NormCommonTiling<EN_QUANT>(context, PrePostType::PreNorm);
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(opName, "PreRmsNormTilingFunc running error!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <bool EN_QUANT>
static ge::graphStatus PostRmsNormTilingFunc(gert::TilingContext* context)
{
    std::string opName = context->GetNodeName();
    auto ret = NormCommonTiling<EN_QUANT>(context, PrePostType::PostNorm);
    OP_CHECK_IF(
        (ret == ge::GRAPH_FAILED), OP_LOGD(opName, "PostRmsNormTilingFunc running error!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormQuant)
    .Tiling(RmsNormTilingFunc<true>) // 需要传入RmsNormTilingFunc<true>
    .TilingParse<NormCommonCompileInfo>(PrepareForRmsNormCommonTiling);
} // namespace optiling