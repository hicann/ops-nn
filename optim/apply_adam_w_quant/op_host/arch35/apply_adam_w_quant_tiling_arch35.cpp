/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_adam_w_quant_tiling_arch35.cpp
 * \brief ApplyAdamWQuant regbase (Ascend950) tiling implementation.
 *
 * 与 A2 tiling 完全同算法(blockwise-256 分核 + UB 切分),仅出包路径不同:regbase 走
 * RawTilingData 直写裸 buffer。核数/UB 由 PlatformAscendC 运行时取(arch35 自适应)。
 */
#include "apply_adam_w_quant_tiling_arch35.h"

#include <cstdint>
#include <securec.h>
#include <string>
#include "log/log.h"
#include "error_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;

namespace optiling {
namespace {
constexpr size_t INDEX_IN_VAR = 0;
constexpr size_t INDEX_IN_GRAD = 1;
constexpr size_t INDEX_IN_M = 2;
constexpr size_t INDEX_IN_V = 3;
constexpr size_t INDEX_IN_QMAP_M = 4;
constexpr size_t INDEX_IN_QMAP_V = 5;
constexpr size_t INDEX_IN_ABSMAX_M = 6;
constexpr size_t INDEX_IN_ABSMAX_V = 7;
constexpr size_t INDEX_IN_STEP = 8;

constexpr size_t INDEX_ATTR_LR = 0;
constexpr size_t INDEX_ATTR_BETA1 = 1;
constexpr size_t INDEX_ATTR_BETA2 = 2;
constexpr size_t INDEX_ATTR_WEIGHT_DECAY = 3;
constexpr size_t INDEX_ATTR_EPS = 4;
constexpr size_t INDEX_ATTR_GNORM_SCALE = 5;
constexpr size_t INDEX_ATTR_BLOCK_SIZE = 7;

constexpr uint64_t TILINGKEY_DATA_VAR_FLOAT = 100;
constexpr uint64_t TILINGKEY_DATA_VAR_FLOAT16 = 200;
constexpr uint64_t TILINGKEY_DATA_VAR_BFLOAT16 = 300;
constexpr uint64_t QMAP_SIZE = 256;
constexpr uint64_t SIZE_OF_FLOAT = 4;
constexpr uint64_t SIZE_OF_FLOAT16 = 2;
constexpr uint64_t PER_BLOCK_OF_MAX_NUM = 1;
constexpr uint64_t ONE_BLOCK_NEED_BUF = 10;
constexpr uint64_t NUM_OF_QMAP = 2;
constexpr int64_t BLOCKSIZE = 256;

inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return (b == 0 ? 0 : ((a + b - 1) / b)); }

inline bool IsSameShape(const gert::Shape& shape1, const gert::Shape& shape2)
{
    size_t dimNum = shape1.GetDimNum();
    if (shape2.GetDimNum() != dimNum) {
        return false;
    }
    for (size_t i = 0; i < dimNum; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus ApplyAdamWQuantRegbaseTiling::GetAttributes()
{
    auto* attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto* attrLr = attrs->GetAttrPointer<float>(INDEX_ATTR_LR);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrLr);
    lr_ = *attrLr;

    auto* attrBeta1 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrBeta1);
    beta1_ = *attrBeta1;

    auto* attrBeta2 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrBeta2);
    beta2_ = *attrBeta2;

    auto* attrWeightDecay = attrs->GetAttrPointer<float>(INDEX_ATTR_WEIGHT_DECAY);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrWeightDecay);
    weightDecay_ = *attrWeightDecay;

    auto* attrEps = attrs->GetAttrPointer<float>(INDEX_ATTR_EPS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrEps);
    eps_ = *attrEps;

    auto* attrGnormScale = attrs->GetAttrPointer<float>(INDEX_ATTR_GNORM_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrGnormScale);
    gnormScale_ = *attrGnormScale;

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrBlockSize);
    blockSize_ = *attrBlockSize;
    OP_CHECK_IF(blockSize_ != BLOCKSIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "attr block_size should be 256, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWQuantRegbaseTiling::CheckInputShape()
{
    auto varShapePtr = tilingContext_->GetInputShape(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varShapePtr);
    auto gradShapePtr = tilingContext_->GetInputShape(INDEX_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, gradShapePtr);
    auto mShapePtr = tilingContext_->GetInputShape(INDEX_IN_M);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, mShapePtr);
    auto vShapePtr = tilingContext_->GetInputShape(INDEX_IN_V);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, vShapePtr);
    auto qmapMShapePtr = tilingContext_->GetInputShape(INDEX_IN_QMAP_M);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, qmapMShapePtr);
    auto qmapVShapePtr = tilingContext_->GetInputShape(INDEX_IN_QMAP_V);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, qmapVShapePtr);
    auto absmaxMShapePtr = tilingContext_->GetInputShape(INDEX_IN_ABSMAX_M);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, absmaxMShapePtr);
    auto absmaxVShapePtr = tilingContext_->GetInputShape(INDEX_IN_ABSMAX_V);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, absmaxVShapePtr);
    auto stepShapePtr = tilingContext_->GetInputShape(INDEX_IN_STEP);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, stepShapePtr);

    const gert::Shape& varShape = varShapePtr->GetStorageShape();
    const gert::Shape& gradShape = gradShapePtr->GetStorageShape();
    const gert::Shape& mShape = mShapePtr->GetStorageShape();
    const gert::Shape& vShape = vShapePtr->GetStorageShape();
    const gert::Shape& qmapMShape = qmapMShapePtr->GetStorageShape();
    const gert::Shape& qmapVShape = qmapVShapePtr->GetStorageShape();
    const gert::Shape& absmaxMShape = absmaxMShapePtr->GetStorageShape();
    const gert::Shape& absmaxVShape = absmaxVShapePtr->GetStorageShape();
    const gert::Shape& stepShape = stepShapePtr->GetStorageShape();

    bool isDiffShape = !IsSameShape(varShape, gradShape) || !IsSameShape(varShape, mShape) ||
                       !IsSameShape(varShape, vShape);
    bool isQmapDiffShape = !IsSameShape(qmapMShape, qmapVShape) ||
                           static_cast<uint64_t>(qmapMShape.GetShapeSize()) != QMAP_SIZE;
    uint64_t expectedAbsmaxSize = CeilDiv(static_cast<uint64_t>(varShape.GetShapeSize()),
                                          static_cast<uint64_t>(blockSize_));
    bool isAbsmaxDiffShape = static_cast<uint64_t>(absmaxMShape.GetShapeSize()) != expectedAbsmaxSize ||
                             static_cast<uint64_t>(absmaxVShape.GetShapeSize()) != expectedAbsmaxSize;
    OP_CHECK_IF(isDiffShape,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "var,grad,m,v should have same shape, please check."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        isQmapDiffShape,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                              "qmapM and qmapV should be same shape,shape is [256], please check."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(isAbsmaxDiffShape,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext_->GetNodeName(), "parameter", "invalid",
                    "absmaxM and absmaxV should each contain ceil(var size / block_size) elements, please check."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stepShape.GetDimNum() != 1 || stepShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "step should have only one element, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWQuantRegbaseTiling::DetermineTilingKey()
{
    auto dtypePtr = tilingContext_->GetInputDesc(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dtypePtr);
    auto dtype = dtypePtr->GetDataType();
    if (dtype == ge::DataType::DT_FLOAT) {
        tilingKey_ = TILINGKEY_DATA_VAR_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        tilingKey_ = TILINGKEY_DATA_VAR_FLOAT16;
    } else if (dtype == ge::DataType::DT_BF16) {
        tilingKey_ = TILINGKEY_DATA_VAR_BFLOAT16;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                              "var dtype should be float/float16/bfloat16, please check.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWQuantRegbaseTiling::DoTiling()
{
    auto platformInfoPtr = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfoPtr);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "aivNum can not be 0, please check."),
                return ge::GRAPH_FAILED);
    uint64_t maxUbSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    OP_CHECK_IF(maxUbSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "ub size can not be 0, please check."),
                return ge::GRAPH_FAILED);

    auto shapePtr = tilingContext_->GetInputShape(INDEX_IN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapePtr);

    uint64_t oneBlockSize = static_cast<uint64_t>(blockSize_) *
                                (SIZE_OF_FLOAT * ONE_BLOCK_NEED_BUF + SIZE_OF_FLOAT16 + SIZE_OF_FLOAT) +
                            (PER_BLOCK_OF_MAX_NUM + PER_BLOCK_OF_MAX_NUM) * SIZE_OF_FLOAT;
    OP_CHECK_IF(oneBlockSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "one core max size can not be 0, please check."),
                return ge::GRAPH_FAILED);
    uint64_t reservedQmap = QMAP_SIZE * SIZE_OF_FLOAT * NUM_OF_QMAP;
    OP_CHECK_IF(maxUbSize <= reservedQmap,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "ub size too small for qmap, please check."),
                return ge::GRAPH_FAILED);
    perCoreDoBlockNum_ = (maxUbSize - reservedQmap) / oneBlockSize;
    OP_CHECK_IF(perCoreDoBlockNum_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "one core do block num can not be 0, please check."),
                return ge::GRAPH_FAILED);

    uint64_t totalDataNum = static_cast<uint64_t>(shapePtr->GetStorageShape().GetShapeSize());
    uint64_t blockNum = CeilDiv(totalDataNum, static_cast<uint64_t>(blockSize_));
    uint64_t totalUseNumCore = CeilDiv(blockNum, perCoreDoBlockNum_);
    OP_CHECK_IF(totalUseNumCore == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "total use num core can not be 0, please check."),
                return ge::GRAPH_FAILED);
    lastCoreLastBlock_ = blockNum - (totalUseNumCore - 1) * perCoreDoBlockNum_;
    lastBlockSize_ = totalDataNum - (blockNum - 1) * static_cast<uint64_t>(blockSize_);
    useNumCore_ = CeilDiv(totalUseNumCore, CeilDiv(totalUseNumCore, aivNum));
    OP_CHECK_IF(useNumCore_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "use_num_core can't be 0."),
                return ge::GRAPH_FAILED);

    lastPreCoreRowWork_ = totalUseNumCore / useNumCore_;
    notLastCoreNum_ = totalUseNumCore - lastPreCoreRowWork_ * useNumCore_;
    notLastPreCoreRowWork_ = lastPreCoreRowWork_ + 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWQuantRegbaseTiling::SetTilingData()
{
    auto* rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData->GetData());
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(ApplyAdamWQuantRegbaseTilingData),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext_->GetNodeName(), "parameter", "invalid",
                    "tiling data capacity is less than ApplyAdamWQuantRegbaseTilingData size."),
                return ge::GRAPH_FAILED);

    ApplyAdamWQuantRegbaseTilingData tiling;
    tiling.use_num_core = useNumCore_;
    tiling.last_pre_core_row_work = lastPreCoreRowWork_;
    tiling.not_last_core_num = notLastCoreNum_;
    tiling.not_last_pre_core_row_work = notLastPreCoreRowWork_;
    tiling.last_core_last_block = lastCoreLastBlock_;
    tiling.lr = lr_;
    tiling.beta1 = beta1_;
    tiling.beta2 = beta2_;
    tiling.weight_decay = weightDecay_;
    tiling.eps = eps_;
    tiling.gnorm_scale = gnormScale_;
    tiling.block_size = blockSize_;
    tiling.one_core_do_block_num_per_row = perCoreDoBlockNum_;
    tiling.tiling_key = tilingKey_;
    tiling.last_block_size = lastBlockSize_;

    auto ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tiling,
                        sizeof(ApplyAdamWQuantRegbaseTilingData));
    OP_CHECK_IF(ret != EOK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "memcpy_s",
                                                      std::to_string(ret).c_str(), "copy tiling data failed"),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(ApplyAdamWQuantRegbaseTilingData));

    tilingContext_->SetBlockDim(static_cast<uint32_t>(useNumCore_));
    tilingContext_->SetTilingKey(tilingKey_);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWQuantRegbaseTiling::RunTiling()
{
    OP_CHECK_IF(
        tilingContext_ == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ApplyAdamWQuant", "parameter", "invalid", "tiling context is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttributes() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "GetAttributes failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "input shape check failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DetermineTilingKey() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "determine tiling key failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "parameter", "invalid",
                                                      "DoTiling failed."),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}

ge::graphStatus Tiling4ApplyAdamWQuant(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Tiling4ApplyAdamWQuant", "parameter", "invalid",
                                              "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling4ApplyAdamWQuant running begin");
    ApplyAdamWQuantRegbaseTiling regbaseTiling(context);
    return regbaseTiling.RunTiling();
}

static ge::graphStatus TilingPrepare4ApplyAdamWQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4ApplyAdamWQuant enter.");
    auto compileInfo = context->GetCompiledInfo<ApplyAdamWQuantRegbaseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "TilingPrepare4ApplyAdamWQuant exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamWQuant)
    .Tiling(Tiling4ApplyAdamWQuant)
    .TilingParse<ApplyAdamWQuantRegbaseCompileInfo>(TilingPrepare4ApplyAdamWQuant);
} // namespace optiling
