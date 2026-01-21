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
 * \file lp_norm_v2_tiling.cc
 * \brief
 */

#include "lp_norm_v2_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_base.h"
#include "norm/lp_norm_v2/op_kernel/arch35/lp_norm_v2_dag.h"
#include "norm/lp_norm_v2/op_kernel/arch35/lp_norm_v2_tiling_key.h"

using namespace Ops::Base;

namespace optiling
{
using namespace LpNormV2;

static const int32_t ATTR_INDEX_P = 0;
static const int32_t ATTR_INDEX_AXIS = 1;
static const int32_t ATTR_INDEX_EPSILON = 3;

static const float DEFAULT_EPSILON_FP16 = 1e-7f;
static const float DEFAULT_EPSILON_FP32 = 1e-12f;

static const float P3 = 3.0f;
static const float P2 = 2.0f;
static const float P1 = 1.0f;
static const float P0 = 0.0f;

constexpr float FLT_EPSILON = 1e-6f;
static const int64_t EMPTY_SHAPE_DIM_NUM_ONE = 1;

// 实现与 Python math_isclose_impl 类似功能的函数
static bool Isclose(float a, float b, float rel_tol = FLT_EPSILON, float abs_tol = 0.0f)
{
    if (a == b) {
        return true;
    }

    if (std::isinf(a) || std::isinf(b)) {
        return false;
    }

    float diff = std::fabs(b - a);
    float tol_b = std::fabs(rel_tol * b);
    float tol_a = std::fabs(rel_tol * a);
    bool ret = (diff <= tol_b) || (diff <= tol_a) || (diff <= abs_tol);
    return ret;
}

ge::graphStatus LpNormV2Tiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");

    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_p(p_);
    tilingData_.set_recp(recp_);

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);
    OP_CHECK_IF(
        tilingData_.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(tilingContext_->GetNodeName(),
                                        "actual tilingData_ data size %zu > context tilingData_ data size %zu",
                                        tilingData_.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key_.reduceTiling.patternID, key_.reduceTiling.loopARCount,
                                                  key_.reduceTiling.loopInnerARCount, key_.templateNum);
    OP_LOGI(tilingContext_->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, templateNum is: %u, p: %f, recp: %f",
            key_.reduceTiling.patternID, key_.reduceTiling.loopARCount, key_.reduceTiling.loopInnerARCount, tilingKey,
            key_.templateNum, p_, recp_);

    tilingContext_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandleP2(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P2;
    if (!dypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P2Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P2Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P2Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandleP1(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P1;
    if (!dypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P1Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P1Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P1Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandleP0(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    templateNum_ = TEMPLATE_P0;
    if (!dypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P0Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P0Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P0Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandlePInf(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                           ReduceOpTilingData& reduceTiling)
{
    templateNum_ = p_ < 0 ? TEMPLATE_P_NINF : TEMPLATE_P_INF;
    if (!dypeXEqualY) {
        if (xDtype_ == ge::DT_FLOAT16) {
            // fp16 inf/-inf场景不需要升精度
            OP_CHECK_IF(
                (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<half, half, float>::OpDag>(
                    tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
                , return ge::GRAPH_FAILED);
        } else if (xDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<half, float, float>::OpDag>(
                    tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
                , return ge::GRAPH_FAILED);
        } else {
            // never reach here
            OP_CHECK_IF(
                (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<float, float, float>::OpDag>(
                    tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
                , return ge::GRAPH_FAILED);
        }
    } else if (xDtype_ == ge::DT_FLOAT16) {
        // fp16 inf/-inf场景不需要升精度
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<half, half, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        // never reach here
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2PInfDag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandleP3(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                         ReduceOpTilingData& reduceTiling)
{
    AscendC::GetPowerTmpBufferFactorSize(true, false, false, sizeof(float), opInput.reservedNode, opInput.reservedSize);
    templateNum_ = TEMPLATE_P3;
    recp_ = 1 / p_;
    if (!dypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P3Dag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P3Dag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2P3Dag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::HandlePOther(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                             ReduceOpTilingData& reduceTiling)
{
    // 高阶api预留8k ub空间
    AscendC::GetPowerTmpBufferFactorSize(true, false, false, sizeof(float), opInput.reservedNode, opInput.reservedSize);
    templateNum_ = TEMPLATE_P_OTHER;
    recp_ = 1 / p_;
    if (!dypeXEqualY) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2POtherDag<half, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2POtherDag<half, float, half>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (Tiling4ReduceOp<LpNormV2::LpNormV2POtherDag<float, float, float>::OpDag>(
                 tilingContext_, opInput, key_.reduceTiling, compileInfo, &reduceTiling) == ge::GRAPH_FAILED),
            , return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void ConvertReduceOpTilingData(ReduceOpTilingDataV2* dst, const ReduceOpTilingData* src)
{
    dst->set_factorACntPerCore(src->factorACntPerCore);
    dst->set_factorATotalCnt(src->factorATotalCnt);
    dst->set_ubFactorA(src->ubFactorA);
    dst->set_factorRCntPerCore(src->factorRCntPerCore);
    dst->set_factorRTotalCnt(src->factorRTotalCnt);
    dst->set_ubFactorR(src->ubFactorR);
    dst->set_groupR(src->groupR);
    dst->set_outSize(src->outSize);
    dst->set_basicBlock(src->basicBlock);
    dst->set_resultBlock(src->resultBlock);
    dst->set_coreNum(src->coreNum);
    dst->set_useNddma(src->useNddma);
    dst->set_meanVar(src->meanVar);
    uint64_t shape[MAX_DIM] = {0};
    int64_t stride[MAX_DIM] = {0};
    int64_t dstStride[MAX_DIM] = {0};
    uint64_t sliceNum[MAX_DIM] = {0};
    uint64_t sliceShape[MAX_DIM] = {0};
    uint64_t sliceStride[MAX_DIM] = {0};
    for (int32_t i = 0; i < MAX_DIM; i++) {
        shape[i] = src->shape[i];
        stride[i] = src->stride[i];
        dstStride[i] = src->dstStride[i];
        sliceNum[i] = src->sliceNum[i];
        sliceShape[i] = src->sliceShape[i];
        sliceStride[i] = src->sliceStride[i];
    }
    dst->set_shape(shape);
    dst->set_stride(stride);
    dst->set_dstStride(dstStride);
    dst->set_sliceNum(sliceNum);
    dst->set_sliceShape(sliceShape);
    dst->set_sliceStride(sliceStride);
}

ge::graphStatus LpNormV2Tiling::TilingReduce(const ReduceOpCompileInfo* compileInfo)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext_, opInput, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext_->GetNodeName(), "ReduceOp get x input failed"),
                    return ge::GRAPH_FAILED);

    opInput.axes = reduceAxis_;

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ReduceOpTilingData reduceTiling;
    if (Isclose(p_, P0)) {
        ret = HandleP0(compileInfo, opInput, reduceTiling);
    } else if (Isclose(p_, P1)) {
        ret = HandleP1(compileInfo, opInput, reduceTiling);
    } else if (Isclose(p_, P2)) {
        ret = HandleP2(compileInfo, opInput, reduceTiling);
    } else if (Isclose(p_, P3)) {
        ret = HandleP3(compileInfo, opInput, reduceTiling);
    } else if (std::isinf(p_)) {
        ret = HandlePInf(compileInfo, opInput, reduceTiling);
    } else {
        ret = HandlePOther(compileInfo, opInput, reduceTiling);
    }
    ConvertReduceOpTilingData(&tilingData_.reduceTiling, &reduceTiling);
    OP_CHECK_IF(
        (ret == ge::GRAPH_FAILED),
        OP_LOGE(tilingContext_->GetNodeName(), "LpNormV2 Tiling for p: %f failed", p_),
        return ge::GRAPH_FAILED);

    key_.templateNum = templateNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::GetAndCheckDtypes()
{
    auto xDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto yDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yDtype = yDesc->GetDataType();
    dypeXEqualY = xDtype_ == yDtype;

    OP_CHECK_IF(xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_BF16,
                    OP_LOGE(tilingContext_->GetNodeName(),
                                                    "input dtype [%s] not support, only support [fp16, fp32, bf16]",
                                                    ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::GetAndCheckReduceAxis()
{
    auto xShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());
    int64_t xShapeDimNum = xStorageShape.GetDimNum();
    xShapeDimNum = xShapeDimNum == 0 ? EMPTY_SHAPE_DIM_NUM_ONE : xShapeDimNum;

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto axisListPtr = attrs->GetListInt(ATTR_INDEX_AXIS);
    if (axisListPtr == nullptr || axisListPtr->GetSize() == 0) {
        reduceAxis_.resize(xShapeDimNum);
        for (int64_t i = 0; i < xShapeDimNum; i++) {
            reduceAxis_[i] = i;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 支持axis去重
    std::set<int64_t> dimSet;
    for (int i = 0; i < static_cast<int>(axisListPtr->GetSize()); i++) {
        int64_t dim = axisListPtr->GetData()[i];
        OP_CHECK_IF((dim < -xShapeDimNum || dim > xShapeDimNum),
                        OP_LOGE(tilingContext_->GetNodeName(),
                                                        "Dimension is: %ld, out of range [-%ld, %ld]", dim,
                                                        xShapeDimNum, xShapeDimNum - 1),
                        return ge::GRAPH_FAILED);

        dim = (dim < 0) ? (dim + xShapeDimNum) : dim;
        if (dimSet.find(dim) != dimSet.end()) {
            continue;
        }

        dimSet.insert(dim);
        reduceAxis_.push_back(dim);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::GetAndCheckOtherAttrs(ge::DataType xDtype)
{
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const float* pPtr = attrs->GetFloat(ATTR_INDEX_P);
    p_ = (pPtr == nullptr) ? P2 : *pPtr;

    const float* epsilonPtr = attrs->GetFloat(ATTR_INDEX_EPSILON);
    if (epsilonPtr == nullptr) {
        if (xDtype == ge::DT_FLOAT16) {
            epsilon_ = DEFAULT_EPSILON_FP16;
        } else {  // bf16 or fp32
            epsilon_ = DEFAULT_EPSILON_FP32;
        }
    } else {
        if (xDtype == ge::DT_FLOAT16 && *epsilonPtr <= DEFAULT_EPSILON_FP16) {
            epsilon_ = Isclose(*epsilonPtr, 0.0f) ? 0.0f : DEFAULT_EPSILON_FP16;
        } else {
            epsilon_ = *epsilonPtr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpNormV2Tiling::RunTiling(const ReduceOpCompileInfo* compileInfo)
{
    if (tilingContext_ == nullptr) {
        OP_LOGE("LpNormV2Tiling", "tilingContext_ is nullptr.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, OP_LOGE("LpNormV2Tiling", "get and check dtypes failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAndCheckReduceAxis() != ge::GRAPH_SUCCESS,
                    OP_LOGE("LpNormV2Tiling", "get and check reduce axis failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckOtherAttrs(xDtype_) != ge::GRAPH_SUCCESS,
                    OP_LOGE("LpNormV2Tiling", "get and check attr failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(TilingReduce(compileInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "do tilingData_ failed for reduce"),
                    return ge::GRAPH_FAILED);

    return SetTilingData();
}

ge::graphStatus Tiling4LpNormV2Func(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4LpNormV2Func running begin");

    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context->GetNodeName(), "Tiling4LpNormV2Func enter LpNormV2 tiling.");
    LpNormV2Tiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrepare4LpNormV2(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(LpNormV2)
    .Tiling(Tiling4LpNormV2Func)
    .TilingParse<ReduceOpCompileInfo>(TilingPrepare4LpNormV2);
}  // namespace optiling