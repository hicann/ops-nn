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
 * \file sgd_tiling.cpp
 * \brief
 */
#include "sgd_tiling.h"
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace SgdOp;

namespace optiling {
namespace {
constexpr size_t SYS_WORKSPACE = 16777216; // 16M
constexpr size_t MAX_DIM_NUM = 8;          // 对齐 ascend910b：rank 1 ~ 8（canndev kMaxDimNum = 8）
constexpr size_t MIN_DIM_NUM = 1;

constexpr int32_t IDX_PARAMETERS = 0;
constexpr int32_t IDX_GRADIENT = 1;
constexpr int32_t IDX_LEARNING_RATE = 2;
constexpr int32_t IDX_ACCUM = 3;
constexpr int32_t IDX_MOMENTUM = 4;
constexpr int32_t IDX_STAT = 5;

constexpr size_t ATTR_IDX_DAMPENING = 0;
constexpr size_t ATTR_IDX_WEIGHT_DECAY = 1;
constexpr size_t ATTR_IDX_NESTEROV = 2;

// 与 parameters 严格同形同 dtype 的大张量
const std::map<int32_t, std::string> TENSOR_INDEX_LIST = {
    {IDX_GRADIENT, "gradient"}, {IDX_ACCUM, "accum"}, {IDX_STAT, "stat"}};
// shape 必须为 [1]（或 0D 标量）且与 parameters 同 dtype 的标量张量
const std::map<int32_t, std::string> SCALAR_INDEX_LIST = {{IDX_LEARNING_RATE, "learning_rate"},
                                                          {IDX_MOMENTUM, "momentum"}};
} // namespace

ge::graphStatus SgdRegbaseTiling::GetAttr()
{
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const float* dampeningAttr = attrs->GetAttrPointer<float>(ATTR_IDX_DAMPENING);
    dampening_ = (dampeningAttr != nullptr) ? *dampeningAttr : 0.0f;
    const float* weightDecayAttr = attrs->GetAttrPointer<float>(ATTR_IDX_WEIGHT_DECAY);
    weightDecay_ = (weightDecayAttr != nullptr) ? *weightDecayAttr : 0.0f;
    const bool* nesterovAttr = attrs->GetAttrPointer<bool>(ATTR_IDX_NESTEROV);
    nesterov_ = (nesterovAttr != nullptr) ? *nesterovAttr : false;

    // 对齐 ascend910b 的两条属性语义校验（canndev nn_training_ops.cc:1932-1955）。
    // 走结构化上报宏（R6），非裸 OP_LOGE。
    OP_CHECK_IF(nesterov_ && dampening_ != 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "dampening",
                                                      std::to_string(dampening_).c_str(),
                                                      "attr dampening must be 0 when attr nesterov is true"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightDecay_ < 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "weight_decay",
                                                      std::to_string(weightDecay_).c_str(),
                                                      "attr weight_decay must be more than or equal to 0"),
                return ge::GRAPH_FAILED);

    // 属性"是否为 0"固化为编译期分支（spec numerical_stability.skip_zero_branches）：
    // 条件不成立时【真正跳过】该子图，而不是乘 0 —— 0 * inf = NaN 会污染结果。
    useNesterovKey_ = nesterov_ ? 1U : 0U;
    hasWeightDecayKey_ = (weightDecay_ != 0.0f) ? 1U : 0U;
    hasDampeningKey_ = (dampening_ != 0.0f) ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckRank(const gert::Shape& input0Shape)
{
    size_t dimNum = input0Shape.GetDimNum();
    OP_CHECK_IF(dimNum < MIN_DIM_NUM || dimNum > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext_->GetNodeName(), "parameters",
                                             std::to_string(dimNum).c_str(), "1 ~ 8 dims"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckNotEmpty(const gert::Shape& input0Shape)
{
    // 空 Tensor（任意一轴或多轴为 0）按 spec error_codes 归 null_input，拒绝为非法。
    // 本算子无"空进空出"语义 —— accum / stat 的原地回写在 numel == 0 下无定义。
    OP_CHECK_IF(input0Shape.GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext_->GetNodeName(), "parameters",
                                                          std::to_string(input0Shape.GetShapeSize()).c_str(),
                                                          "empty tensor (any axis being 0) is not supported by SGD"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckScalarShape(int32_t inputIdx)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    const std::string& paramName = SCALAR_INDEX_LIST.at(inputIdx);
    OP_CHECK_IF((!storageShape.IsScalar() && storageShape.GetShapeSize() != 1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    tilingContext_->GetNodeName(), paramName.c_str(), Ops::Base::ToString(storageShape).c_str(),
                    (std::string("input ") + paramName + " must be scalar(0D) or have shape size 1").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    // 严格相等，不做广播 —— "可广播但不相等"（如 [2,3] vs [1,3]）同样判非法，
    // 对齐 spec 的 shape_mismatch 口径。
    if (inputShape->GetStorageShape() != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype)
{
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    if (inputDesc->GetDataType() != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::CheckShapeAndType()
{
    auto inputShape = tilingContext_->GetInputShape(IDX_PARAMETERS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputParamDesc = tilingContext_->GetInputDesc(IDX_PARAMETERS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputParamDesc);
    auto inputDtype = inputParamDesc->GetDataType();

    // ascend910b 只校验了 parameters 的 rank，其余 5 个输入的同形 / 同 dtype 完全未校验
    // （canndev CheckSgdDimension 仅看 parameters）。ascend950 侧补齐，见 01 §6.5。
    // 内层 CheckRank 已用 OP_LOGE_FOR_INVALID_SHAPEDIM 结构化上报，此处只作调用链留痕，
    // 不重复打 ERROR（R6 要求的是"校验必须走结构化宏"，不是"每层都再报一次"）。
    OP_CHECK_IF(CheckRank(inputStorageShape) != ge::GRAPH_SUCCESS, OP_LOGD(tilingContext_, "rank check failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckNotEmpty(inputStorageShape) != ge::GRAPH_SUCCESS,
                OP_LOGD(tilingContext_, "empty tensor check failed"), return ge::GRAPH_FAILED);

    for (const auto& pair : SCALAR_INDEX_LIST) {
        OP_CHECK_IF(CheckScalarShape(pair.first) != ge::GRAPH_SUCCESS,
                    OP_LOGD(tilingContext_, "scalar shape check failed for %s", pair.second.c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                tilingContext_->GetNodeName(), (std::string("parameters and ") + pair.second).c_str(),
                (ge::TypeUtils::DataTypeToSerialString(inputDtype) + " and " +
                 ge::TypeUtils::DataTypeToSerialString(tilingContext_->GetInputDesc(pair.first)->GetDataType()))
                    .c_str(),
                (std::string("the dtypes of input ") + pair.second + " and input parameters must be the same").c_str()),
            return ge::GRAPH_FAILED);
    }

    for (const auto& pair : TENSOR_INDEX_LIST) {
        OP_CHECK_IF(
            CheckSameShape(pair.first, inputStorageShape) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                tilingContext_->GetNodeName(), (std::string("parameters and ") + pair.second).c_str(),
                (Ops::Base::ToString(inputStorageShape) + " and " +
                 Ops::Base::ToString(tilingContext_->GetInputShape(pair.first)->GetStorageShape()))
                    .c_str(),
                (std::string("the shapes of input ") + pair.second + " and input parameters must be the same").c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                tilingContext_->GetNodeName(), (std::string("parameters and ") + pair.second).c_str(),
                (ge::TypeUtils::DataTypeToSerialString(inputDtype) + " and " +
                 ge::TypeUtils::DataTypeToSerialString(tilingContext_->GetInputDesc(pair.first)->GetDataType()))
                    .c_str(),
                (std::string("the dtypes of input ") + pair.second + " and input parameters must be the same").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <bool useNesterov, bool hasWeightDecay, bool hasDampening>
ge::graphStatus SgdRegbaseTiling::DoElewiseTilingByDtype(ElewiseBaseTiling& eleBaseTiling, ge::DataType dtype)
{
    // 一律用回写 DAG（doWriteback = true）反解 ubFormer：掩码 DAG 的 BufferNum 严格更小，
    // 按更保守的一套反解不会溢出；且 Host 收不到 Device 张量数据，本来也看不见 momentum 的值。
    if (dtype == ge::DT_FLOAT) {
        return eleBaseTiling
            .DoTiling<typename SgdDag<float, useNesterov, hasWeightDecay, hasDampening, true, float>::OpDag>(
                tiling_->elewiseTiling);
    }
    if (dtype == ge::DT_FLOAT16) {
        return eleBaseTiling
            .DoTiling<typename SgdDag<half, useNesterov, hasWeightDecay, hasDampening, true, float>::OpDag>(
                tiling_->elewiseTiling);
    }
    if (dtype == ge::DT_BF16) {
        return eleBaseTiling
            .DoTiling<typename SgdDag<bfloat16_t, useNesterov, hasWeightDecay, hasDampening, true, float>::OpDag>(
                tiling_->elewiseTiling);
    }
    OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "parameters",
                              ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), "float32, float16 or bfloat16");
    return ge::GRAPH_FAILED;
}

ge::graphStatus SgdRegbaseTiling::DoElewiseTiling()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto paramDesc = tilingContext_->GetInputDesc(IDX_PARAMETERS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, paramDesc);
    ge::DataType dtype = paramDesc->GetDataType();

    OP_LOGI(tilingContext_->GetNodeName(),
            "Do elewise base tiling with nesterov=%d, hasWeightDecay=%d, hasDampening=%d", static_cast<int>(nesterov_),
            static_cast<int>(hasWeightDecayKey_), static_cast<int>(hasDampeningKey_));

    // 6 个合法业务模板（K0~K5）。nesterov == true && dampening != 0 已在 GetAttr 拦下，
    // 故此处 nesterov 分支下 hasDampening 恒为 0，不生成该组合。
    if (useNesterovKey_ == 0) {
        if (hasWeightDecayKey_ == 0 && hasDampeningKey_ == 0) { // K0
            return DoElewiseTilingByDtype<false, false, false>(eleBaseTiling, dtype);
        }
        if (hasWeightDecayKey_ == 0 && hasDampeningKey_ == 1) { // K1
            return DoElewiseTilingByDtype<false, false, true>(eleBaseTiling, dtype);
        }
        if (hasWeightDecayKey_ == 1 && hasDampeningKey_ == 0) { // K2
            return DoElewiseTilingByDtype<false, true, false>(eleBaseTiling, dtype);
        }
        return DoElewiseTilingByDtype<false, true, true>(eleBaseTiling, dtype); // K3
    }
    if (hasWeightDecayKey_ == 1) { // K4
        return DoElewiseTilingByDtype<true, true, false>(eleBaseTiling, dtype);
    }
    return DoElewiseTilingByDtype<true, false, false>(eleBaseTiling, dtype); // K5
}

ge::graphStatus SgdRegbaseTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE; // 算子不申请临时 GM，仅系统保留段

    // 属性值下发给 Kernel，由 sch.SetVar 注入 DAG 的 Placeholder::Var 节点
    tiling_->dampening = dampening_;
    tiling_->weightDecay = weightDecay_;

    tilingKey_ = GET_TPL_TILING_KEY(tiling_->elewiseTiling.scheMode, useNesterovKey_, hasWeightDecayKey_,
                                    hasDampeningKey_);
    OP_LOGI(tilingContext_->GetNodeName(),
            "scheMode=%ld, useNesterov=%ld, hasWeightDecay=%ld, hasDampening=%ld, tilingKey=%lu",
            tiling_->elewiseTiling.scheMode, useNesterovKey_, hasWeightDecayKey_, hasDampeningKey_, tilingKey_);
    tilingContext_->SetTilingKey(tilingKey_);

    uint32_t blockDim = static_cast<uint32_t>(tiling_->elewiseTiling.blockNum);
    OP_CHECK_IF(blockDim <= 0, OP_LOGE(tilingContext_, "Get blockDim failed"), return ge::GRAPH_FAILED);
    tilingContext_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SgdRegbaseTiling::RunTiling()
{
    if (tilingContext_ == nullptr) {
        OP_LOGE("Sgd", "Get nullptr while obtaining tilingContext_.");
        return ge::GRAPH_FAILED;
    }
    // GetAttr / CheckShapeAndType 内部对非法输入已用 OP_LOGE_FOR_INVALID_* 结构化上报，
    // 此处只作调用链留痕、不重复打 ERROR —— 与上面 CheckRank / CheckNotEmpty 的处理一致。
    // R6 要求的是"校验必须走结构化宏"，不是"每层都再报一次"。
    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGD(tilingContext_, "Get attr failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGD(tilingContext_, "Shape and dtype check failed."),
                return ge::GRAPH_FAILED);
    tiling_ = tilingContext_->GetTilingData<SgdRegbaseTilingData>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get SgdRegbaseTilingData from GE context failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "elewiseBaseTiling failed"),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}

ge::graphStatus Tiling4Sgd(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Sgd running begin");
    SgdRegbaseTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForSgd(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSgd running begin");
    return ge::GRAPH_SUCCESS;
}

struct SgdCompileInfo {};

IMPL_OP_OPTILING(SGD).Tiling(Tiling4Sgd).TilingParse<SgdCompileInfo>(TilingPrepareForSgd);

} // namespace optiling
