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
 * \file group_norm_silu_tiling.cpp
 * \brief
 */
#include "op_common/op_host/util/op_const_def.h"
#include "group_norm_silu_tiling.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include <string>
#include <vector>
#include "util/shape_util.h"
#include "base/err_msg.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t INDEX_NUM_GROUPS = 0;
static const uint64_t INDEX_EPSILON = 1;
static const uint64_t INDEX_ACTIVATE_SILU = 2;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DEFAULT_PROCESSSIZE = 8192;
static const uint64_t DEFAULT_PROCESSSIZE_KIRINX90 = 5120;
static const uint64_t DEFAULT_NUMGROUPS = 32;
static const uint64_t RESERVED_WORKSPACE_SIZE_910B = static_cast<uint64_t>(16 * 1024 * 1024);
static const uint64_t RESERVED_WORKSPACE_SIZE_310P = static_cast<uint64_t>(2 * 1024 * 1024);
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t FLOAT16_BYTES = 2;
static const uint64_t BLOCK_SIZE = 32;
static const int64_t HW_CAP = 4096;
static const int64_t UPPER_LIMIT_TWO = 4000;
static const int64_t UPPER_LIMIT_ONE = 2700;
static const uint64_t FLOAT_EIGHT = 8;
static const uint64_t FLOAT_DOUBLE_EIGHT = 16;
static const uint64_t GAMMA_BETA_UB_NUM = 6;
static const uint64_t RESERVED_BLOCK_NUM = 2;
static const uint64_t INPUT_OUTPUT_UB_NUM = 20;

inline static ge::graphStatus GroupNormSiluSetTilingData(gert::TilingContext* context,
                                                         GroupNormSiluTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    } else if (value % factor == 0) {
        return value / factor;
    } else {
        return value / factor + 1;
    }
}

inline static int64_t Gcd(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return Gcd(b, a % b);
}

inline static int64_t Lcm(int64_t a, int64_t b) { return a * b / Gcd(a, b); }

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // check x
    auto inputX = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_IF(inputX == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "group_norm_silu" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    auto xDtype = inputX->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF((xDtypeSize <= 0),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                    ("xDtypeSize must be greater than 0, but got " + std::to_string(xDtypeSize)).c_str()),
                return ge::GRAPH_FAILED);
    auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF(xShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "group_norm_silu" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    uint64_t xDims = xShape.GetDimNum();
    OP_CHECK_IF((xDims < X_SHAPE_MIN_LEN),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDims).c_str(),
                                             "greater than or equal to 2"),
                return ge::GRAPH_FAILED);
    uint64_t channel = xShape.GetDim(DIM_1);
    // check gamma and beta
    auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
    if (gammaShapePtr == nullptr) {
        const std::string msgOpName = std::string(
            (context->GetNodeName() == nullptr ? "group_norm_silu" : context->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "The optional gamma/beta metadata is unavailable in this Tiling path; check the context "
                               "and selected implementation for this optional-input combination.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "The optional gamma/beta metadata is unavailable in this Tiling path; check the "
                                      "context and selected implementation for this optional-input combination."}));
        return ge::GRAPH_FAILED;
    };
    auto gammaShape = gammaShapePtr->GetStorageShape();
    uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF(gammaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "gamma",
                                                         std::to_string(gammaShape.GetDimNum()).c_str(),
                                                         "The shape dim of gamma must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaSizes != channel,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "gamma", std::to_string(gammaSizes).c_str(),
                    ("The shape size of gamma must be the same as channel"
                     "(channel is the second dim of x), got gamma size = " +
                     std::to_string(gammaSizes) + ", channel = " + std::to_string(channel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
    if (betaShapePtr == nullptr) {
        const std::string msgOpName = std::string(
            (context->GetNodeName() == nullptr ? "group_norm_silu" : context->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "The optional gamma/beta metadata is unavailable in this Tiling path; check the context "
                               "and selected implementation for this optional-input combination.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "The optional gamma/beta metadata is unavailable in this Tiling path; check the "
                                      "context and selected implementation for this optional-input combination."}));
        return ge::GRAPH_FAILED;
    };
    auto betaShape = betaShapePtr->GetStorageShape();
    uint64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF(betaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "beta",
                                                         std::to_string(betaShape.GetDimNum()).c_str(),
                                                         "The shape dim of beta must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        betaSizes != channel,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "beta", std::to_string(betaSizes).c_str(),
                                                  ("The shape size of beta must be the same as channel"
                                                   "(channel is the second dim of x), got channel = " +
                                                   std::to_string(channel))
                                                      .c_str()),
        return ge::GRAPH_FAILED);
    auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
    if (gammaDtypePtr == nullptr) {
        const std::string msgOpName = std::string(
            (context->GetNodeName() == nullptr ? "group_norm_silu" : context->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "The optional gamma/beta metadata is unavailable in this Tiling path; check the context "
                               "and selected implementation for this optional-input combination.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "The optional gamma/beta metadata is unavailable in this Tiling path; check the "
                                      "context and selected implementation for this optional-input combination."}));
        return ge::GRAPH_FAILED;
    };
    auto gammaDtype = gammaDtypePtr->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
    if (betaDtypePtr == nullptr) {
        const std::string msgOpName = std::string(
            (context->GetNodeName() == nullptr ? "group_norm_silu" : context->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "The optional gamma/beta metadata is unavailable in this Tiling path; check the context "
                               "and selected implementation for this optional-input combination.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "The optional gamma/beta metadata is unavailable in this Tiling path; check the "
                                      "context and selected implementation for this optional-input combination."}));
        return ge::GRAPH_FAILED;
    };
    auto betaDtype = betaDtypePtr->GetDataType();
    uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDtype);
    OP_CHECK_IF((gammaDtypeSize != betaDtypeSize),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "gamma and beta",
                                                       (ge::TypeUtils::DataTypeToSerialString(gammaDtype) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(betaDtype))
                                                           .c_str(),
                                                       "The dtypes of gamma and beta must be the same"),
                return ge::GRAPH_FAILED);
    if (xDtypeSize == FLOAT32_BYTES) {
        OP_CHECK_IF(
            (gammaDtypeSize != xDtypeSize || betaDtypeSize != xDtypeSize),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "gamma and beta",
                                                   (ge::TypeUtils::DataTypeToSerialString(gammaDtype) + " and " +
                                                    ge::TypeUtils::DataTypeToSerialString(betaDtype))
                                                       .c_str(),
                                                   "The dtypes of gamma and beta must be the same as x (float32)"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* tilingContext)
{
    auto xShape = tilingContext->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ? "group_norm_silu" :
                                                                                       (tilingContext)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_IF(
        (*numGroups <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "num_groups",
                                              std::to_string(*numGroups).c_str(), "num_groups must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((channel % *numGroups != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext->GetNodeName(), "num_groups", std::to_string(*numGroups).c_str(),
                    ("channel(channel is the second dim of x) must be integer multiples of num_groups, channel is " +
                     std::to_string(channel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupNormSilu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Start running TilingPrepare4GroupNormSilu.");
    auto compileInfo = context->GetCompiledInfo<GroupNormSiluCompileInfo>();
    if (compileInfo == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object compileInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object compileInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object platformInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object platformInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Get core num for ai_core:%d", compileInfo->totalCoreNum);
    if (ascendcPlatform.GetCurNpuArch() == Ops::Base::DAV_2002) {
        compileInfo->is310P = 1;
        compileInfo->totalCoreNum = compileInfo->totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    if (IsRegbaseSocVersion(context)) {
        compileInfo->isRegbase = 1;
        uint32_t vectorLength = Ops::Base::GetVRegSize(context);
        if ((vectorLength <= FLOAT32_BYTES)) {
            const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                               "group_norm_silu" :
                                               context->GetNodeName());
            const std::string msgReason = (std::string(__func__) + ": " + "vector length is invalid.");
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
            return ge::GRAPH_FAILED;
        };
        compileInfo->vectorLength = vectorLength;

        int32_t blockSize = Ops::Base::GetUbBlockSize(context);
        if ((blockSize <= 0)) {
            const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                               "group_norm_silu" :
                                               context->GetNodeName());
            const std::string msgReason = (std::string(__func__) + ": " + "block size is invalid.");
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
            return ge::GRAPH_FAILED;
        };
        compileInfo->blockSizePlatform = blockSize;
    }
    OP_LOGD(context, "Get total core num:%d", compileInfo->totalCoreNum);
    if ((compileInfo->totalCoreNum <= 0)) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "group_norm_silu" :
                                           context->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "Failed to get core num.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "Failed to get core num."}));
        return ge::GRAPH_FAILED;
    };

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_LOGD(context, "Get total ub size:%lu", compileInfo->ubSizePlatForm);
    if ((compileInfo->ubSizePlatForm <= 0)) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "group_norm_silu" :
                                           context->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "Failed to get ub size.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "Failed to get ub size."}));
        return ge::GRAPH_FAILED;
    };

    OP_LOGD(context, "TilingPrepare4GroupNormSilu ends.");
    return ge::GRAPH_SUCCESS;
}

static void SetAttrParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    const float* epsilon = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    const bool* activateSilu = attrs->GetAttrPointer<bool>(INDEX_ACTIVATE_SILU);
    tilingData.set_numGroups(*numGroups);
    tilingData.set_epsilon(*epsilon);
    tilingData.set_activateSilu(*activateSilu);
}

static void SetTilingParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t hwNum = 1;
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tilingData.set_shapeC(xShape.GetDim(DIM_1));
    tilingData.set_shapeD(tilingData.get_shapeC() / tilingData.get_numGroups());
    tilingData.set_hwNum(hwNum);
    tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t processSize = (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRINX90 ||
                            ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRIN9030) ?
                               DEFAULT_PROCESSSIZE_KIRINX90 :
                               DEFAULT_PROCESSSIZE;
    tilingData.set_processSize(static_cast<int64_t>(processSize));
}

static void SetBlockTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    uint64_t totalGroups = static_cast<uint64_t>(shapeN) * static_cast<uint64_t>(tilingData.get_numGroups());
    tilingData.set_numPerCore(CeilDiv(totalGroups, compileInfo->totalCoreNum));
    tilingData.set_realCoreNum(CeilDiv(totalGroups, tilingData.get_numPerCore()));
    if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize) &&
        (tilingData.get_hwNum() != 1 || tilingData.get_shapeD() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize))) {
        tilingData.set_realCoreNum(1);
    }

    tilingData.set_numLastCore(totalGroups - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    // split coreNum according to N
    if (tilingData.get_hwNum() == 1 && tilingData.get_numGroups() == tilingData.get_shapeC()) {
        if (tilingData.get_shapeC() % (BLOCK_SIZE / xDtypeSize) != 0) {
            tilingData.set_realCoreNum(1);
            tilingData.set_numLastCore(shapeN);
        } else {
            tilingData.set_numPerCore(CeilDiv(shapeN, compileInfo->totalCoreNum));
            tilingData.set_realCoreNum(CeilDiv(shapeN, tilingData.get_numPerCore()));
        }
        tilingData.set_numLastCore(shapeN - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    }

    uint64_t xShapeSize = xShape.GetShapeSize();
    if (xShapeSize == 0) {
        tilingData.set_realCoreNum(-1);
    }
}

static void SetTilingWithSmallHW(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    if (tilingData.get_hwNum() == 1) {
        if (xDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B32));
        } else if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B16));
        }
    } else {
        if (xDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
        } else if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
        }
    }
}

static void SetUbTiling(GroupNormSiluTilingData& tilingData)
{
    tilingData.set_loopNum(CeilDiv(tilingData.get_elemNum(), tilingData.get_processSize()));
    tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * (tilingData.get_loopNum() - 1));
    tilingData.set_innerLoopNum(CeilDiv(tilingData.get_hwNum(), tilingData.get_processSize()));
    tilingData.set_innerLoopTail(tilingData.get_hwNum() -
                                 tilingData.get_processSize() * (tilingData.get_innerLoopNum() - 1));
}

// mixed-dtype(gamma 为 float32)时选 mixKey，否则选 b16Key。抽出以消除 SetTiling 中重复的 gamma dtype 分支。
static int64_t SelectMixOrB16Key(uint64_t gammaDtypeSize, GroupNormSiluTilingKey mixKey, GroupNormSiluTilingKey b16Key)
{
    return static_cast<int64_t>(gammaDtypeSize == FLOAT32_BYTES ? mixKey : b16Key);
}

static void SetTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize)) {
        SetTilingWithSmallHW(context, tilingData);
    } else if (xDtypeSize == FLOAT32_BYTES && tilingData.get_hwNum() % FLOAT_EIGHT == 0 &&
               tilingData.get_hwNum() <= HW_CAP &&
               tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B32));
    } else if (xDtypeSize == FLOAT32_BYTES &&
               tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B32));
    } else if (xDtypeSize == FLOAT16_BYTES && tilingData.get_hwNum() % FLOAT_DOUBLE_EIGHT == 0 &&
               tilingData.get_hwNum() <= HW_CAP &&
               tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
        tilingData.set_tilingKey(SelectMixOrB16Key(gammaDtypeSize,
                                                   GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_MIXTYPE,
                                                   GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B16));
    } else if (xDtypeSize == FLOAT16_BYTES &&
               tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
        tilingData.set_tilingKey(SelectMixOrB16Key(gammaDtypeSize, GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_MIXTYPE,
                                                   GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B16));
    } else if (xDtypeSize == FLOAT32_BYTES) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
    } else {
        tilingData.set_tilingKey(SelectMixOrB16Key(gammaDtypeSize, GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE,
                                                   GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
    }
}

static ge::graphStatus SetProcessSize(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    uint64_t gammaPerCore = (tilingData.get_numPerCore() + 1) * tilingData.get_shapeD();
    OP_CHECK_IF((gammaDtypeSize == 0),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "gamma",
                                                      std::to_string(static_cast<int>(gammaDtype)),
                                                      "Division by zero, gammaDtypeSize is 0!"),
                return ge::GRAPH_FAILED);
    uint64_t gammaPerCoreRoundUp = CeilDiv(gammaPerCore, (BLOCK_SIZE / gammaDtypeSize)) * (BLOCK_SIZE / gammaDtypeSize);
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint64_t remainUbSize = compileInfo->ubSizePlatForm - gammaPerCoreRoundUp * gammaDtypeSize * GAMMA_BETA_UB_NUM -
                            BLOCK_SIZE * RESERVED_BLOCK_NUM;
    if ((remainUbSize == 0)) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "group_norm_silu" :
                                           context->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "RemainUbSize is 0!");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "RemainUbSize is 0!"}));
        return ge::GRAPH_FAILED;
    };
    int64_t maxProcessSize = remainUbSize / INPUT_OUTPUT_UB_NUM;
    if (maxProcessSize < tilingData.get_hwNum()) {
        maxProcessSize = maxProcessSize / BLOCK_SIZE * BLOCK_SIZE;
    } else {
        int64_t lcmNum = Lcm(tilingData.get_hwNum(), (BLOCK_SIZE / gammaDtypeSize));
        OP_CHECK_IF((lcmNum == 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "x",
                        Ops::Base::ToString(context->GetInputShape(INPUT_IDX_X)->GetStorageShape()),
                        "The spatial dimensions must permit a nonzero aligned processing tile."),
                    return ge::GRAPH_FAILED);
        maxProcessSize = maxProcessSize / lcmNum * lcmNum;
    }
    tilingData.set_processSize(maxProcessSize);
    tilingData.set_numGroups(gammaPerCoreRoundUp);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingSD(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    auto xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    uint64_t shapeN = xShape.GetDim(DIM_0);
    if (tilingData.get_realCoreNum() > 0 && tilingData.get_numGroups() == DEFAULT_NUMGROUPS && shapeN == 1 &&
        xDtypeSize == FLOAT16_BYTES && tilingData.get_hwNum() > 1) {
        OP_LOGD(context, "Start running SetTilingSD.");
        auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
        uint64_t numPerCoreRound = tilingData.get_numGroups() / compileInfo->totalCoreNum;
        tilingData.set_numPerCore(numPerCoreRound);
        tilingData.set_realCoreNum(compileInfo->totalCoreNum);
        tilingData.set_numLastCore(tilingData.get_numGroups() -
                                   tilingData.get_numPerCore() * tilingData.get_realCoreNum());
        OP_CHECK_IF((SetProcessSize(context, tilingData) != ge::GRAPH_SUCCESS),
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SetProcessSize failed."), return ge::GRAPH_FAILED);
        tilingData.set_loopNum(tilingData.get_elemNum() / tilingData.get_processSize());
        tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * tilingData.get_loopNum());
        if (tilingData.get_processSize() > tilingData.get_hwNum()) {
            tilingData.set_innerLoopNum(tilingData.get_processSize() / tilingData.get_hwNum());
            tilingData.set_innerLoopTail(tilingData.get_processSize() -
                                         tilingData.get_hwNum() * tilingData.get_innerLoopNum());
        } else {
            tilingData.set_innerLoopNum(tilingData.get_hwNum() / tilingData.get_processSize());
            tilingData.set_innerLoopTail(tilingData.get_hwNum() -
                                         tilingData.get_processSize() * tilingData.get_innerLoopNum());
        }
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SPECIAL_SHAPE_SD));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupNormSilu(gert::TilingContext* context)
{
    OP_LOGD(context, "Start running Tiling4GroupNormSilu.");
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    if (compileInfo->isRegbase == 1) {
        return Tiling4GroupNormSiluRegBase(context);
    }
    // check input && attrs params
    OP_CHECK_IF((CheckInputParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckAttrParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "AttrParams is invalid."), return ge::GRAPH_FAILED);
    GroupNormSiluTilingData tilingData;
    SetAttrParams(context, tilingData);
    SetTilingParams(context, tilingData);
    // block tiling
    SetBlockTiling(context, tilingData);
    // tiling key
    SetTiling(context, tilingData);
    // ub tiling
    SetUbTiling(tilingData);
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
    if (compileInfo->is310P == 1) {
        OP_LOGD(context, "Current chip type is 310P.");
        sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_310P;
        OP_CHECK_IF((SetTilingSD(context, tilingData) != ge::GRAPH_SUCCESS),
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SetTilingSD failed."), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(GroupNormSiluSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GroupNormSiluSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "tilingData is numGroups:%ld, hwNum:%ld, elemNum:%ld, shapeC:%ld, shapeD:%ld, realCoreNum:%ld, \
                numPerCore:%ld, numLastCore:%ld, processSize:%ld, loopNum:%ld, loopTail:%ld, innerLoopNum:%ld, \
                innerLoopTail:%ld, tilingKey:%ld, epsilon:%f, activateSilu:%ld, Tiling4GroupNormSilu ends. ",
            tilingData.get_numGroups(), tilingData.get_hwNum(), tilingData.get_elemNum(), tilingData.get_shapeC(),
            tilingData.get_shapeD(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
            tilingData.get_numLastCore(), tilingData.get_processSize(), tilingData.get_loopNum(),
            tilingData.get_loopTail(), tilingData.get_innerLoopNum(), tilingData.get_innerLoopTail(),
            tilingData.get_tilingKey(), tilingData.get_epsilon(), tilingData.get_activateSilu());

    // block dim, tilingKey
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSilu)
    .Tiling(Tiling4GroupNormSilu)
    .TilingParse<GroupNormSiluCompileInfo>(TilingPrepare4GroupNormSilu);
} // namespace optiling
