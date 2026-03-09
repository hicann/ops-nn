/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file renorm_tiling.cpp
 * \brief
 */

#include "renorm_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "norm/renorm/op_kernel/arch35/renorm_dag.h"
#include "norm/renorm/op_kernel/arch35/renorm_tiling_key.h"

using namespace Ops::Base;

namespace optiling
{
using namespace Renorm;

static const int32_t ATTR_INDEX_P = 0;
static const int32_t ATTR_INDEX_DIM = 1;
static const int32_t ATTR_INDEX_MAXNORM = 2;
static const int32_t ATTR_INDEX_EPSILON = 3;

static const float DEFAULT_EPSILON_FP16 = 1e-7f;
static const float DEFAULT_EPSILON_FP32 = 1e-12f;

static const float P3 = 3.0f;
static const float P2 = 2.0f;
static const float P1 = 1.0f;
static const float P0 = 0.0f;

constexpr float RENORM_FLT_EPSILON = 1e-6f;
static const int64_t EMPTY_SHAPE_DIM_NUM_ONE = 1;

static const int32_t REDUCE_AXIS_ONE = 1;

// 实现与 Python math_isclose_impl 类似功能的函数
static bool Isclose(float a, float b, float rel_tol = RENORM_FLT_EPSILON, float abs_tol = 0.0f)
{
    if (a == b) {
        return true;
    }

    if (std::isinf(a) || std::isinf(b)) {
        return false;
    }

    float diff = std::fabs(b - a);

    auto ret = (((diff <= std::fabs(rel_tol * b)) || (diff <= std::fabs(rel_tol * a))) || (diff <= abs_tol));

    return ret;
}

ge::graphStatus RenormTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");

    tilingData_->epsilon = epsilon_;
    tilingData_->p = p_;
    tilingData_->recp = recp_;
    tilingData_->maxnorm = maxnorm_;

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key_.reduceTiling.patternID, key_.reduceTiling.loopARCount,
                                                  key_.reduceTiling.loopInnerARCount, key_.templateNum);
    OP_LOGI(tilingContext_->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, templateNum is: %u, p: %f, recp: %f, maxnorm: %f",
            key_.reduceTiling.patternID, key_.reduceTiling.loopARCount, key_.reduceTiling.loopInnerARCount, tilingKey,
            key_.templateNum, p_, recp_, maxnorm_);

    tilingContext_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandleP2(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P2;
    if (!dtypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP2Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP2Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP2Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandleP1(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P1;
    if (!dtypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP1Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP1Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP1Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandleP0(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P0;
    if (!dtypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP0Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP0Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP0Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandlePInf(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                           ReduceOpTilingData& reduceTiling)
{
    templateNum_ = p_ < 0 ? TEMPLATE_P_NINF : TEMPLATE_P_INF;
    if (!dtypeXEqualY) {
        if (xDtype_ == ge::DT_FLOAT16) {
            // fp16 inf/-inf场景不需要升精度
            OP_CHECK_IF(
                (Tiling4ReduceOp<Renorm::RenormPInfDag<half, half, float>::OpDag>(
                    tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
                , return ge::GRAPH_FAILED);
        } else if (xDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<Renorm::RenormPInfDag<half, float, float>::OpDag>(
                    tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
                , return ge::GRAPH_FAILED);
        }
    } else if (xDtype_ == ge::DT_FLOAT16) {
        // fp16 inf/-inf场景不需要升精度
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPInfDag<half, half, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPInfDag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPInfDag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandleP3(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    AscendC::GetPowerTmpBufferFactorSize(true, false, false, sizeof(float), opInput.reservedNode, opInput.reservedSize);
    templateNum_ = TEMPLATE_P3;
    recp_ = 1 / p_;
    if (!dtypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP3Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP3Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormP3Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::HandlePOther(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                             ReduceOpTilingData& reduceTiling)
{
    // 高阶api预留8k ub空间
    AscendC::GetPowerTmpBufferFactorSize(true, false, false, sizeof(float), opInput.reservedNode, opInput.reservedSize);
    templateNum_ = TEMPLATE_P_OTHER;
    recp_ = 1 / p_;
    if (!dtypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPOtherDag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPOtherDag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<Renorm::RenormPOtherDag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool RenormTiling::CheckReduceAxisIsOne()
{
    auto xShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());

    for (auto index : reduceAxis_) {
        if (xStorageShape.GetDim(index) != REDUCE_AXIS_ONE) {
            return false;
        }
    }
    return true;
}

ge::graphStatus RenormTiling::TilingReduce(const ReduceOpCompileInfo* compileInfo)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext_, opInput, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext_->GetNodeName(), "ReduceOp get x input failed"),
                    return ge::GRAPH_FAILED);

    opInput.axes = reduceAxis_;

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (Isclose(p_, P0)) {
        ret = HandleP0(compileInfo, opInput, tilingData_->reduceTiling);
    } else if (Isclose(p_, P1) || CheckReduceAxisIsOne()) {
        ret = HandleP1(compileInfo, opInput, tilingData_->reduceTiling);
    } else if (Isclose(p_, P2)) {
        ret = HandleP2(compileInfo, opInput, tilingData_->reduceTiling);
    } else if (Isclose(p_, P3)) {
        ret = HandleP3(compileInfo, opInput, tilingData_->reduceTiling);
    } else if (std::isinf(p_)) {
        ret = HandlePInf(compileInfo, opInput, tilingData_->reduceTiling);
    } else {
        ret = HandlePOther(compileInfo, opInput, tilingData_->reduceTiling);
    }
    OP_CHECK_IF(
        (ret == ge::GRAPH_FAILED),
        OP_LOGE(tilingContext_->GetNodeName(), "Renorm Tiling for p: %f failed", p_),
        return ge::GRAPH_FAILED);

    key_.templateNum = templateNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::GetAndCheckDtypes()
{
    auto xDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto yDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yDtype = yDesc->GetDataType();
    dtypeXEqualY = xDtype_ == yDtype;

    OP_CHECK_IF(xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_BF16,
                    OP_LOGE(tilingContext_->GetNodeName(),
                                                    "input dtype [%s] not support, only support [fp16, fp32, bf16]",
                                                    ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::GetAndCheckReduceAxis()
{
    auto xShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());
    int64_t xShapeDimNum = xStorageShape.GetDimNum();
    xShapeDimNum = xShapeDimNum == 0 ? EMPTY_SHAPE_DIM_NUM_ONE : xShapeDimNum;

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto targetDimPtr = attrs->GetInt(ATTR_INDEX_DIM);
    if (targetDimPtr == nullptr) {
        OP_LOGE("RenormTiling", "dim is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // 边界检查：dim不合法（<0或>=xShapeDimNum）
    int targetDim = *targetDimPtr;
    if (targetDim < 0 || targetDim >= xShapeDimNum) {
        OP_LOGE("RenormTiling", "dim range only support [-%ld, %ld - 1], but now dim is [%ld].", xShapeDimNum, xShapeDimNum, targetDim);
        return ge::GRAPH_FAILED;
    }

    reduceAxis_.clear();
    // 合法dim：预分配+遍历填充
    reduceAxis_.reserve(xShapeDimNum - 1);
    for (int64_t i = 0; i < xShapeDimNum; ++i) {
        if (i != targetDim) {
            reduceAxis_.push_back(i);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::GetAndCheckOtherAttrs(ge::DataType xDtype)
{
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const float* pPtr = attrs->GetFloat(ATTR_INDEX_P);
    p_ = (pPtr == nullptr) ? P2 : *pPtr;

    const float* maxnormPtr = attrs->GetFloat(ATTR_INDEX_MAXNORM);
    if (maxnormPtr == nullptr) {
        OP_LOGE("RenormTiling", "maxnorm_ is nullptr.");
        return ge::GRAPH_FAILED;
    }
    maxnorm_ = *maxnormPtr;

    if (xDtype == ge::DT_FLOAT16) {
        epsilon_ = DEFAULT_EPSILON_FP16;
    } else {  // bf16 or fp32
        epsilon_ = DEFAULT_EPSILON_FP32;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RenormTiling::RunTiling(const ReduceOpCompileInfo* compileInfo)
{
    if (tilingContext_ == nullptr) {
        OP_LOGE("RenormTiling", "tilingContext_ is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingData_ = tilingContext_->GetTilingData<RenormTilingData>();

    OP_CHECK_IF(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, OP_LOGE("RenormTiling", "get and check dtypes failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAndCheckReduceAxis() != ge::GRAPH_SUCCESS,
                    OP_LOGE("RenormTiling", "get and check reduce axis failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckOtherAttrs(xDtype_) != ge::GRAPH_SUCCESS,
                    OP_LOGE("RenormTiling", "get and check attr failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(TilingReduce(compileInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "do tilingData_ failed for reduce"),
                    return ge::GRAPH_FAILED);

    return SetTilingData();
}

ge::graphStatus Tiling4RenormFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Tiling4RenormFunc running begin");

    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context->GetNodeName(), "Tiling4RenormFunc enter Renorm tiling.");
    RenormTiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrepare4Renorm(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ReduceOpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0UL),
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                                        compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= CACHE_BUF_SIZE,
                    OP_LOGE(context->GetNodeName(),
                                                    "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                                                    compileInfo->ubSize, CACHE_BUF_SIZE),
                    return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.",
                                        compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.",
                                        compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.",
                                        compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
            compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
            compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Renorm)
    .Tiling(Tiling4RenormFunc)
    .TilingParse<ReduceOpCompileInfo>(TilingPrepare4Renorm);
}  // namespace optiling