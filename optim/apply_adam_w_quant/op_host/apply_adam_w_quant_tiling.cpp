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
 * \file apply_adam_w_quant_tiling.cpp
 * \brief
 */
#include <cstdint>
#include <vector>
#include <cmath>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "apply_adam_w_quant_tiling_def.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
static const size_t INDEX_IN_VAR = 0;
static const size_t INDEX_IN_GRAD = 1;
static const size_t INDEX_IN_M = 2;
static const size_t INDEX_IN_V = 3;
static const size_t INDEX_IN_QMAP_M = 4;
static const size_t INDEX_IN_QMAP_V = 5;
static const size_t INDEX_IN_ABSMAX_M = 6;
static const size_t INDEX_IN_ABSMAX_V = 7;
static const size_t INDEX_IN_STEP = 8;

static const size_t INDEX_ATTR_LR = 0;
static const size_t INDEX_ATTR_BETA1 = 1;
static const size_t INDEX_ATTR_BETA2 = 2;
static const size_t INDEX_ATTR_WEIGHT_DECAY = 3;
static const size_t INDEX_ATTR_EPS = 4;
static const size_t INDEX_ATTR_GNORM_SCALE = 5;
static const size_t INDEX_ATTR_QUANT_MODE = 6;
static const size_t INDEX_ATTR_BLOCK_SIZE = 7;

static const size_t TILINGKEY_DATA_VAR_FLOAT = 100;
static const size_t TILINGKEY_DATA_VAR_FLOAT16 = 200;
static const size_t TILINGKEY_DATA_VAR_BFLOAT16 = 300;
static const size_t QMAP_SIZE = 256;
static const uint64_t SIZE_OF_FLOAT = 4;
static const uint64_t SIZE_OF_FLOAT16 = 2;
static const uint64_t PER_BLOCK_OF_MAX_NUM = 1;
static const uint64_t ONE_BLOCK_NEED_BUF = 10;
static const uint64_t NUM_OF_QMAP = 2;
static const float EPS = 0.00001f;
static const int64_t BLOCKSIZE = 256;

inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (b == 0 ? 0 : ((a + b - 1) / b));
}

inline static ge::graphStatus ApplyAdamWQuantSetTilingData(
    gert::TilingContext* context, ApplyAdamWQuantTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(ApplyAdamWQuantTilingData& tilingData)
{
    OP_LOGI("ApplyAdamWQuant", "use_num_core: %ld", tilingData.get_use_num_core());
    OP_LOGI("ApplyAdamWQuant", "last_pre_core_row_work: %ld", tilingData.get_last_pre_core_row_work());
    OP_LOGI("ApplyAdamWQuant", "not_last_core_num: %ld", tilingData.get_not_last_core_num());
    OP_LOGI("ApplyAdamWQuant", "not_last_pre_core_row_work: %ld", tilingData.get_not_last_pre_core_row_work());
    OP_LOGI("ApplyAdamWQuant", "last_core_last_block: %ld", tilingData.get_last_core_last_block());
    OP_LOGI("ApplyAdamWQuant", "lr: %f", tilingData.get_lr());
    OP_LOGI("ApplyAdamWQuant", "beta1: %f", tilingData.get_beta1());
    OP_LOGI("ApplyAdamWQuant", "beta2: %f", tilingData.get_beta2());
    OP_LOGI("ApplyAdamWQuant", "weightDecay: %f", tilingData.get_weight_decay());
    OP_LOGI("ApplyAdamWQuant", "eps: %f", tilingData.get_eps());
    OP_LOGI("ApplyAdamWQuant", "gnorm_scale: %f", tilingData.get_gnorm_scale());
    OP_LOGI("ApplyAdamWQuant", "one_core_do_block_num_per_row: %ld", tilingData.get_one_core_do_block_num_per_row());
    OP_LOGI("ApplyAdamWQuant", "tiling_key: %ld", tilingData.get_tiling_key());
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context, ApplyAdamWQuantTilingParam& tilingParam)
{
    // get attrs of lr, beta1, beta2, weight_decay, eps, gnorm_scale and block_size
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrLr = attrs->GetAttrPointer<float>(INDEX_ATTR_LR);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrLr);
    tilingParam.lr = static_cast<float>(*attrLr);

    auto* attrBeta1 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA1);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBeta1);
    tilingParam.beta1 = static_cast<float>(*attrBeta1);

    auto* attrBeta2 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA2);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBeta2);
    tilingParam.beta2 = static_cast<float>(*attrBeta2);

    auto* attrWeightDecay = attrs->GetAttrPointer<float>(INDEX_ATTR_WEIGHT_DECAY);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrWeightDecay);
    tilingParam.weight_decay = static_cast<float>(*attrWeightDecay);

    auto* attrEps = attrs->GetAttrPointer<float>(INDEX_ATTR_EPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrEps);
    tilingParam.eps = static_cast<float>(*attrEps);

    auto* attrGnormScale = attrs->GetAttrPointer<float>(INDEX_ATTR_GNORM_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrGnormScale);
    tilingParam.gnorm_scale = static_cast<float>(*attrGnormScale);

    auto* attrBlockSize = attrs->GetAttrPointer<int>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.block_size = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(
        tilingParam.block_size != BLOCKSIZE, OP_LOGE(context, "attr block_size should be 256, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsSameShape(const gert::Shape shape1, const gert::Shape shape2)
{
    size_t inputShapeSize = shape1.GetDimNum();
    if (shape2.GetDimNum() != inputShapeSize) {
        return false;
    }
    for (size_t i = 0; i < inputShapeSize; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CheckZero(uint64_t num)
{
    if (num == 0) {
        return true;
    }
    return false;
}

static ge::graphStatus CheckInputShape(const gert::TilingContext* context)
{
    auto varShapePtr = context->GetInputShape(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    auto varShape = varShapePtr->GetStorageShape();

    auto gradShapePtr = context->GetInputShape(INDEX_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
    auto gradShape = gradShapePtr->GetStorageShape();

    auto mShapePtr = context->GetInputShape(INDEX_IN_M);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShapePtr);
    auto mShape = mShapePtr->GetStorageShape();

    auto vShapePtr = context->GetInputShape(INDEX_IN_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShapePtr);
    auto vShape = vShapePtr->GetStorageShape();

    auto qmapMShapePtr = context->GetInputShape(INDEX_IN_QMAP_M);
    OP_CHECK_NULL_WITH_CONTEXT(context, qmapMShapePtr);
    auto qmapMShape = qmapMShapePtr->GetStorageShape();
    auto qmapMSize = qmapMShapePtr->GetStorageShape().GetShapeSize();

    auto qmapVShapePtr = context->GetInputShape(INDEX_IN_QMAP_M);
    OP_CHECK_NULL_WITH_CONTEXT(context, qmapVShapePtr);
    auto qmapVShape = qmapVShapePtr->GetStorageShape();

    auto stepShapePtr = context->GetInputShape(INDEX_IN_STEP);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepShapePtr);
    auto stepShape = stepShapePtr->GetStorageShape();

    bool isDiffShape =
        !IsSameShape(varShape, gradShape) || !IsSameShape(varShape, mShape) || !IsSameShape(varShape, vShape);
    bool isqmapDiffShape = !IsSameShape(qmapMShape, qmapVShape) || qmapMSize != QMAP_SIZE;
    OP_CHECK_IF(
        isDiffShape, OP_LOGE(context, "var,grad,m,v should have same shape, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        isqmapDiffShape, OP_LOGE(context, "qmapM and qmapV should be same shape,shape is [256], please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        stepShape.GetDimNum() != 1 || stepShape.GetDim(0) != 1,
        OP_LOGE(context, "step should have only one element, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(
    const gert::TilingContext* context, ApplyAdamWQuantTilingParam& tilingParam, uint64_t max_ub_size)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    auto shapePtr = context->GetInputShape(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
    uint64_t one_block_size =
        tilingParam.block_size * (SIZE_OF_FLOAT * ONE_BLOCK_NEED_BUF + SIZE_OF_FLOAT16 + SIZE_OF_FLOAT) +
        (PER_BLOCK_OF_MAX_NUM + PER_BLOCK_OF_MAX_NUM) * SIZE_OF_FLOAT;
    bool check_one_block_size_zero = CheckZero(one_block_size);
    OP_CHECK_IF(
        check_one_block_size_zero, OP_LOGE(context, "one core max size can not be 0, please check."),
        return ge::GRAPH_FAILED);
    uint64_t per_core_do_block_num = (max_ub_size - QMAP_SIZE * SIZE_OF_FLOAT * NUM_OF_QMAP) / one_block_size;
    uint64_t totalDataNum = shapePtr->GetStorageShape().GetShapeSize();
    uint64_t block_num = (totalDataNum + tilingParam.block_size - 1) / tilingParam.block_size;
    bool check_per_core_do_block_num = CheckZero(per_core_do_block_num);
    OP_CHECK_IF(
        check_per_core_do_block_num, OP_LOGE(context, "one core do block num can not be 0, please check."),
        return ge::GRAPH_FAILED);
    uint64_t total_use_num_core = (block_num + per_core_do_block_num - 1) / per_core_do_block_num;
    uint64_t last_core_last_block = block_num - (total_use_num_core - 1) * per_core_do_block_num;
    uint64_t use_num_core = CeilDiv(total_use_num_core, CeilDiv(total_use_num_core, aivNum));
    if (use_num_core <= 0) {
        OP_LOGE(context, "use_num_core can't be smaller than 0.");
        return ge::GRAPH_FAILED;
    }

    uint64_t last_pre_core_row_work = total_use_num_core / use_num_core;
    uint64_t not_last_core_num = total_use_num_core - last_pre_core_row_work * use_num_core;
    uint64_t not_last_pre_core_row_work = last_pre_core_row_work + 1;

    tilingParam.use_num_core = use_num_core;
    tilingParam.last_pre_core_row_work = last_pre_core_row_work;
    tilingParam.not_last_core_num = not_last_core_num;
    tilingParam.not_last_pre_core_row_work = not_last_pre_core_row_work;
    tilingParam.last_core_last_block = last_core_last_block;
    tilingParam.per_core_do_block_num = per_core_do_block_num;
    return ge::GRAPH_SUCCESS;
}

static void GetTilingData(ApplyAdamWQuantTilingData& tilingData, const ApplyAdamWQuantTilingParam& tilingParam)
{
    tilingData.set_use_num_core(tilingParam.use_num_core);
    tilingData.set_last_pre_core_row_work(tilingParam.last_pre_core_row_work);
    tilingData.set_not_last_core_num(tilingParam.not_last_core_num);
    tilingData.set_not_last_pre_core_row_work(tilingParam.not_last_pre_core_row_work);
    tilingData.set_last_core_last_block(tilingParam.last_core_last_block);
    tilingData.set_lr(tilingParam.lr);
    tilingData.set_beta1(tilingParam.beta1);
    tilingData.set_beta2(tilingParam.beta2);
    tilingData.set_weight_decay(tilingParam.weight_decay);
    tilingData.set_eps(tilingParam.eps);
    tilingData.set_gnorm_scale(tilingParam.gnorm_scale);
    tilingData.set_block_size(tilingParam.block_size);
    tilingData.set_one_core_do_block_num_per_row(tilingParam.per_core_do_block_num);
}

ge::graphStatus Tiling4ApplyAdamWQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4ApplyAdamWQuant running begin");
    ApplyAdamWQuantTilingParam tilingParam;
    uint64_t max_ub_size;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);

    OP_CHECK_IF(
        GetTilingAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Tiling4ApplyAdamWQuant GetTilingAttr fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckInputShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "input shape check failed."),
        return ge::GRAPH_FAILED);

    auto dtypePtr = context->GetInputDesc(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
    auto dtype = dtypePtr->GetDataType();

    uint64_t tilingKey = 0;
    if (dtype == ge::DataType::DT_FLOAT) {
        tilingKey += TILINGKEY_DATA_VAR_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        tilingKey += TILINGKEY_DATA_VAR_FLOAT16;
    } else if (dtype == ge::DataType::DT_BF16) {
        tilingKey += TILINGKEY_DATA_VAR_BFLOAT16;
    }

    context->SetTilingKey(tilingKey);

    OP_CHECK_IF(
        DoTiling(context, tilingParam, max_ub_size) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Tiling4ApplyAdamWQuant DoTiling fail."), return ge::GRAPH_FAILED);

    ApplyAdamWQuantTilingData tilingData;
    tilingData.set_tiling_key(tilingKey);
    GetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(
        ApplyAdamWQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Tiling4ApplyAdamWQuantSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_use_num_core());

    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ApplyAdamWQuant(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamWQuant)
    .Tiling(Tiling4ApplyAdamWQuant)
    .TilingParse<Tiling4ApplyAdamWQuantCompileInfo>(TilingPrepare4ApplyAdamWQuant);
} // namespace optiling