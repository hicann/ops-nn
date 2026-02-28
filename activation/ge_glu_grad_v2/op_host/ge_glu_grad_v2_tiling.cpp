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
 * \file ge_glu_grad_v2_tiling.cpp
 * \brief
 */
#include <map>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "ge_glu_grad_v2_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr char NODE_NAME[] = "GeGluGradV2";

constexpr uint32_t DY_INDEX = 0;
constexpr uint32_t X_INDEX = 1;
constexpr uint32_t GELU_INDEX = 2;
constexpr uint32_t DX_INDEX = 0;

constexpr uint32_t DIM_ATTR_INDEX = 0;
constexpr uint32_t APPROXIMATE_ATTR_INDEX = 1;
constexpr uint32_t ACTIVATE_LEFT_ATTR_INDEX = 2;

constexpr uint32_t BATCH_MODE = 1;

/* Tanh */
constexpr int32_t TANH_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t TANH_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t TANH_BUF_CNT_FP32 = 11;

constexpr int32_t TANH_BUF_CNT_FP16_950 = 5 * 2 + 6 * 2;
constexpr int32_t TANH_BUF_CNT_BFP16_950 = 5 * 2 + 2 * 2 * 2 + 4 * 2;
constexpr int32_t TANH_BUF_CNT_FP32_950 = 5 + 6 * 2;

/* Erf */
constexpr int32_t ERF_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t ERF_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t ERF_BUF_CNT_FP32 = 11;

constexpr int32_t ERF_BUF_CNT_FP16_950 = 3 * 2 + 20;  // 转fp32，调用Ascendc erf接口，3*fp32的buf保留
constexpr int32_t ERF_BUF_CNT_BFP16_950 = 3 * 2 + 20; // 转fp32，调用Ascendc erf接口，3*fp32的buf保留
constexpr int32_t ERF_BUF_CNT_FP32_950 = 3 + 16;      // 调用Ascendc erf接口，3*fp32的buf保留

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TRANSPOSE_REPEAT_SIZE = 512;
constexpr int32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t REGBASE_CCEC_RESERVE_SIZE = 8 * 1024;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_HUNDRED = 100;

static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_TANH = {
    {ge::DT_BF16, TANH_BUF_CNT_BFP16}, {ge::DT_FLOAT16, TANH_BUF_CNT_FP16}, {ge::DT_FLOAT, TANH_BUF_CNT_FP32}};
static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_TANH_950 = {
    {ge::DT_BF16, TANH_BUF_CNT_BFP16_950},
    {ge::DT_FLOAT16, TANH_BUF_CNT_FP16_950},
    {ge::DT_FLOAT, TANH_BUF_CNT_FP32_950}};

static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_ERF = {
    {ge::DT_BF16, ERF_BUF_CNT_BFP16}, {ge::DT_FLOAT16, ERF_BUF_CNT_FP16}, {ge::DT_FLOAT, ERF_BUF_CNT_FP32}};
static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_ERF_950 = {
    {ge::DT_BF16, ERF_BUF_CNT_BFP16_950},
    {ge::DT_FLOAT16, ERF_BUF_CNT_FP16_950},
    {ge::DT_FLOAT, ERF_BUF_CNT_FP32_950}};

class GeGluGradV2Tiling {
public:
    explicit GeGluGradV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling4GeGluGradV2();

private:
    ge::graphStatus Init();
    ge::graphStatus CheckParams();
    void FillTilingData();

    template <typename T1, typename T2>
    inline auto AlignA2B(T1 a, T2 b) const -> T1
    {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a / b) * b);
    };

    void CalcValueNM();
    ge::graphStatus CaclMaxProcessCount();
    void ProcessTilingCore();

private:
    GeGluGradV2TilingData tilingData;
    GeGluGradV2TilingKey tilingKey = GeGluGradV2TilingKey::TILING_KEY_TANH_101;
    gert::TilingContext* tilingContext = nullptr;
    const GeGluGradV2CompileInfo* ptrCompileInfo = nullptr;

    // input output infos
    gert::Shape dyShape;
    gert::Shape xShape;
    gert::Shape geluShape;
    gert::Shape dxShape;
    int64_t dimAttr = -1L;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;

    /**
     * The meanings of valueN and valueM are as follows:
     * Shape(A, B, C) of input x, dim=1 ==> valueN=A, valueM=B*C//2
     * Shape(A, B, C) of input x, dim=-1 ==> valueN=A*B, valueM=C//2
     * Shape(A, B, C, D) of input x, dim=2 ==> valueN=A*B, valueM=C*D//2
     */
    int64_t valueN = 1;
    int64_t valueM = 1;
    ge::DataType dyDtype = ge::DT_UNDEFINED;
    int32_t dtypeSize = 0;

    // tiling params
    int64_t maxProcCount = 0;
    int32_t needCoreNum = 0;
    int64_t loopNumPerCore = 0;
    int64_t tailCoreIndex = 0;
    int64_t tailUbLoopNum = 0;
    int64_t groupNum = 0;
    uint64_t ubSizePlatForm_ = 0;
};

ge::graphStatus GeGluGradV2Tiling::RunTiling4GeGluGradV2()
{
    ptrCompileInfo = tilingContext->GetCompileInfo<GeGluGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrCompileInfo);
    ubSizePlatForm_ = ptrCompileInfo->ubSizePlatForm;

    OP_CHECK_IF(Init() != ge::GRAPH_SUCCESS, OP_LOGE(NODE_NAME, "Init failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckParams() != ge::GRAPH_SUCCESS, OP_LOGE(NODE_NAME, "CheckParams failed."), return ge::GRAPH_FAILED);
    CalcValueNM();
    OP_LOGD(
        NODE_NAME, "Platform info, ubSizePlatForm:%lu, totalCoreNum:%d, curSocVersion:%u.", ubSizePlatForm_,
        ptrCompileInfo->totalCoreNum, static_cast<int32_t>(ptrCompileInfo->curSocVersion));
    OP_CHECK_IF(
        CaclMaxProcessCount() != ge::GRAPH_SUCCESS, OP_LOGE(NODE_NAME, "CaclMaxProcessCount failed."),
        return ge::GRAPH_FAILED);

    ProcessTilingCore();

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(static_cast<uint64_t>(tilingKey));
    FillTilingData();
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(WORK_SPACE_SIZE + ptrCompileInfo->totalCoreNum * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeGluGradV2Tiling::Init()
{
    auto inputDy = tilingContext->GetInputTensor(DY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDy);
    dyShape = inputDy->GetStorageShape();
    dyDtype = tilingContext->GetInputDesc(DY_INDEX)->GetDataType();

    auto inputX = tilingContext->GetInputTensor(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX);
    xShape = inputX->GetStorageShape();

    auto inputYgelu = tilingContext->GetInputTensor(GELU_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputYgelu);
    geluShape = inputYgelu->GetStorageShape();

    auto outputDx = tilingContext->GetOutputShape(DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDx);
    dxShape = outputDx->GetStorageShape();

    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const int64_t* ptrDim = attrs->GetAttrPointer<int64_t>(DIM_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrDim);
    dimAttr = *ptrDim;
    const int64_t* ptrApproximate = attrs->GetAttrPointer<int64_t>(APPROXIMATE_ATTR_INDEX);
    // 310P donot support bfloat16 and erf mode
    const bool is310p = ptrCompileInfo->curSocVersion == NpuArch::DAV_2002;
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrApproximate);
    approximateAttr = *ptrApproximate;
    OP_CHECK_IF(
        approximateAttr == 0 && is310p, OP_LOGE(NODE_NAME, "approximate only support 1(Tanh) in 310P."),
        return ge::GRAPH_FAILED);
    const bool* ptrActivateLeft = attrs->GetAttrPointer<bool>(ACTIVATE_LEFT_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrActivateLeft);
    activateLeftAttr = *ptrActivateLeft;

    OP_LOGD(
        NODE_NAME, "Attr info: dimAttr: %ld, approximateAttr: %ld, activateLeftAttr: %s, dyDtype: %d.", dimAttr,
        approximateAttr, activateLeftAttr ? "true" : "false", static_cast<int32_t>(dyDtype));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeGluGradV2Tiling::CheckParams()
{
    OP_CHECK_IF(
        dyDtype != ge::DT_BF16 && dyDtype != ge::DT_FLOAT16 && dyDtype != ge::DT_FLOAT,
        OP_LOGE(NODE_NAME, "Data type support only float16, bfloat16, float32."), return ge::GRAPH_FAILED);

    // 310P donot support bfloat16 and erf mode
    const bool is310p = ptrCompileInfo->curSocVersion == NpuArch::DAV_2002;
    OP_CHECK_IF(
        dyDtype == ge::DT_BF16 && is310p, OP_LOGE(NODE_NAME, "Data type donot support only bfloat16 in 310P."),
        return ge::GRAPH_FAILED);
    dtypeSize = ge::GetSizeByDataType(dyDtype);
    OP_CHECK_IF(dtypeSize <= 0, OP_LOGE(NODE_NAME, "dtypeSize[%d] is invalid.", dtypeSize), return ge::GRAPH_FAILED);

    auto xDtype = tilingContext->GetInputDesc(X_INDEX)->GetDataType();
    auto geluDtype = tilingContext->GetInputDesc(GELU_INDEX)->GetDataType();
    auto dxDtype = tilingContext->GetInputDesc(DX_INDEX)->GetDataType();
    OP_CHECK_IF(
        dyDtype != geluDtype || xDtype != dxDtype || dyDtype != xDtype,
        OP_LOGE(NODE_NAME, "The dtype of input should be same."), return ge::GRAPH_FAILED);

    size_t xDimNum = xShape.GetDimNum();
    dimAttr = dimAttr < 0 ? static_cast<int64_t>(xDimNum) + dimAttr : dimAttr;
    if (dimAttr < 0 || dimAttr >= static_cast<int64_t>(xDimNum)) {
        OP_LOGE(NODE_NAME, "Dim %ld is not in [0, %lu) or input x is a scalar.", dimAttr, xDimNum);
        return ge::GRAPH_FAILED;
    }

    size_t dyDimNum = dyShape.GetDimNum();
    size_t geluDimNum = geluShape.GetDimNum();
    OP_CHECK_IF(
        dyDimNum != xDimNum || geluDimNum != xDimNum, OP_LOGE(NODE_NAME, "The dim num of input should be same."),
        return ge::GRAPH_FAILED);

    int64_t xShapeSize = xShape.GetShapeSize();
    OP_CHECK_IF(
        xShapeSize == 0, OP_LOGE(NODE_NAME, "Can not support input x shape size is zero."), return ge::GRAPH_FAILED);

    gert::Shape tempShape = dyShape;
    tempShape.SetDim(dimAttr, NUM_TWO * dyShape.GetDim(dimAttr));
    if (dyShape != geluShape || xShape != dxShape || tempShape != xShape) {
        OP_LOGE(NODE_NAME, "The input-output shape does not satisfy the operator constraint.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::FillTilingData()
{
    tilingData.set_approximate(static_cast<int32_t>(approximateAttr));
    tilingData.set_activateLeft(static_cast<int32_t>(activateLeftAttr));
    tilingData.set_maxProcCount(maxProcCount);
    tilingData.set_valueN(valueN);
    tilingData.set_valueM(valueM);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_loopNumPerCore(loopNumPerCore);
    tilingData.set_tailCoreIndex(tailCoreIndex);
    tilingData.set_tailUbLoopNum(tailUbLoopNum);
    tilingData.set_groupNum(groupNum);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(
        NODE_NAME,
        "Tiling data is maxProcCount:%ld, valueN:%ld, valueM:%ld, needCoreNum:%ld, loopNumPerCore:%ld, "
        "tailCoreIndex:%ld, tailUbLoopNum:%ld, groupNum:%ld, tilingKey:%lu.",
        tilingData.get_maxProcCount(), tilingData.get_valueN(), tilingData.get_valueM(), tilingData.get_needCoreNum(),
        tilingData.get_loopNumPerCore(), tilingData.get_tailCoreIndex(), tilingData.get_tailUbLoopNum(),
        tilingData.get_groupNum(), static_cast<uint64_t>(tilingKey));
}

void GeGluGradV2Tiling::CalcValueNM()
{
    for (int64_t i = 0; i < dimAttr; ++i) {
        valueN *= dyShape.GetDim(i);
    }
    for (int64_t i = dimAttr; i < int64_t(dyShape.GetDimNum()); ++i) {
        valueM *= dyShape.GetDim(i);
    }
}

ge::graphStatus GeGluGradV2Tiling::CaclMaxProcessCount()
{
    if (approximateAttr == NUM_ONE) {
        const auto iter = ptrCompileInfo->isRegbase ? DTYPE_BUF_CNT_MAP_TANH_950.find(dyDtype) :
                                                           DTYPE_BUF_CNT_MAP_TANH.find(dyDtype);
        maxProcCount = AlignA2B(ubSizePlatForm_ / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_TANH_101;
    } else {
        const auto iter = ptrCompileInfo->isRegbase ? DTYPE_BUF_CNT_MAP_ERF_950.find(dyDtype) :
                                                           DTYPE_BUF_CNT_MAP_ERF.find(dyDtype);
        maxProcCount = AlignA2B(ubSizePlatForm_ / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_ERF_701;
    }

    if (dyDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_HUNDRED);
    } else if (dyDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO * NUM_HUNDRED);
    }

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::ProcessTilingCore()
{
    int64_t ubLoopNum = 0;
    int64_t repeatDataCount = static_cast<int64_t>(TRANSPOSE_REPEAT_SIZE / dtypeSize);
    int64_t maxPerfCount = maxProcCount / repeatDataCount;
    if ((ptrCompileInfo->curSocVersion == NpuArch::DAV_2201 || ptrCompileInfo->isRegbase) &&
        valueM <= maxPerfCount) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO);
        groupNum = AlignA2B(maxProcCount / valueM, repeatDataCount);
        ubLoopNum = Ops::Base::CeilDiv(valueN, groupNum);
        tailUbLoopNum = groupNum == 0 ? valueN : valueN % groupNum;
    } else if (valueM <= maxProcCount) {
        int64_t alignValueM =
            Ops::Base::CeilDiv(valueM, static_cast<int64_t>((BLOCK_SIZE / dtypeSize))) * (BLOCK_SIZE / dtypeSize);
        groupNum = maxProcCount / alignValueM;
        ubLoopNum = Ops::Base::CeilDiv(valueN, groupNum);
        tailUbLoopNum = groupNum == 0 ? valueN : valueN % groupNum;
    } else {
        groupNum = Ops::Base::CeilDiv(valueM, maxProcCount);
        ubLoopNum = valueN * groupNum;
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_ONE);
    }

    needCoreNum = ubLoopNum < ptrCompileInfo->totalCoreNum ? ubLoopNum : ptrCompileInfo->totalCoreNum;
    if (needCoreNum < ptrCompileInfo->totalCoreNum) {
        loopNumPerCore = 0;
        tailCoreIndex = tailUbLoopNum != 0 ? needCoreNum - 1 : needCoreNum;
    } else {
        loopNumPerCore = ubLoopNum / ptrCompileInfo->totalCoreNum;
        int64_t modValue = ubLoopNum % ptrCompileInfo->totalCoreNum;
        if (modValue != 0) {
            tailCoreIndex = tailUbLoopNum != 0 ? modValue - 1 : modValue;
        } else {
            loopNumPerCore -= 1;
            tailCoreIndex = tailUbLoopNum != 0 ? ptrCompileInfo->totalCoreNum - 1 : ptrCompileInfo->totalCoreNum;
        }
    }
}

ge::graphStatus Tiling4GeGluGradV2(gert::TilingContext* context)
{
    OP_LOGD(NODE_NAME, "Tiling4GeGluGradV2 tiling begin.");
    context->SetScheduleMode(BATCH_MODE);
    GeGluGradV2Tiling tilingObject(context);
    OP_CHECK_IF(
        tilingObject.RunTiling4GeGluGradV2() != ge::GRAPH_SUCCESS, OP_LOGE(NODE_NAME, "RunTiling4GeGluGradV2 failed."),
        return ge::GRAPH_FAILED);
    OP_LOGD(NODE_NAME, "Tiling4GeGluGradV2 tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GeGluGradV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GeGluGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(NODE_NAME, "TilingPrepare4GeGluGradV2 get core num failed."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(NODE_NAME, "TilingPrepare4GeGluGradV2 get ub size failed."),
        return ge::GRAPH_FAILED);

    compileInfo->curSocVersion = ascendcPlatform.GetCurNpuArch();
    compileInfo->isRegbase = IsRegbaseSocVersion(context);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GeGluGradV2).Tiling(Tiling4GeGluGradV2).TilingParse<GeGluGradV2CompileInfo>(TilingPrepare4GeGluGradV2);

} // namespace optiling
