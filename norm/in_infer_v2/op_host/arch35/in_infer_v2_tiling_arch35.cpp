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
 * \file in_infer_v2_tiling_arch35.cpp
 * \brief INInferV2 arch35 tiling（ND-only；GE 图模式可能下发 NCHW 标签，一并接受）
 *        统一切分模型：N=d0, C=d1, inner=R=prod(d2:)，units=N*C 个连续 plane；
 *        统计量恒为 [N,C] 逻辑布局，元素数=N*C 校验；无 workspace。
 */

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "in_infer_v2_tiling_arch35.h"
#include "log/log.h"

using namespace optiling;

// asc_opc 编译需要注册 tiling 结构体（字段与 op_kernel 侧 plain struct 二进制一致）
BEGIN_TILING_DATA_DEF(INInferV2TilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, numN);
TILING_DATA_FIELD_DEF(int64_t, numC);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, units);
TILING_DATA_FIELD_DEF(int64_t, unitCores);
TILING_DATA_FIELD_DEF(int64_t, formerCoreNum);
TILING_DATA_FIELD_DEF(int64_t, formerUnits);
TILING_DATA_FIELD_DEF(int64_t, latterUnits);
TILING_DATA_FIELD_DEF(int64_t, innerCores);
TILING_DATA_FIELD_DEF(int64_t, innerPerCore);
TILING_DATA_FIELD_DEF(int64_t, ubTileSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, hasGammaBeta);
TILING_DATA_FIELD_DEF(int64_t, hasBatchMean);
TILING_DATA_FIELD_DEF(int64_t, hasBatchVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(INInferV2, INInferV2TilingDataDef)

using namespace ge;

namespace optiling {

static constexpr int64_t VL = 64;           // fp32 向量寄存器宽度
static constexpr int64_t CHUNK_PLANES = 64; // 统计量 staging 粒度（4 队列 × 64 × 4B = 1KB）
static constexpr int64_t FLOAT_BYTES = 4;
// TPipe 元数据 + 统计量 staging + scale 预计算 TBuf + batch 小队列余量
static constexpr int64_t RESERVED_UB = 8512 + 5 * CHUNK_PLANES * FLOAT_BYTES + 2048;

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_GAMMA_INDEX = 1;
static constexpr size_t INPUT_BETA_INDEX = 2;
static constexpr size_t INPUT_MEAN_INDEX = 3;
static constexpr size_t INPUT_VAR_INDEX = 4;
static constexpr size_t OUTPUT_BATCH_MEAN_INDEX = 1;
static constexpr size_t OUTPUT_BATCH_VAR_INDEX = 2;
static constexpr size_t ATTR_EPSILON_INDEX = 0;

ge::graphStatus INInferV2Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const INInferV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    }
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "invalid coreNum %ld", coreNum_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_, "invalid ubSize %ld", ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INInferV2Tiling::GetShapeAndDtype()
{
    // attr epsilon（可选，缺省 1e-5，与 proto 一致）
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(ATTR_EPSILON_INDEX);
    epsilon_ = (epsilonPtr == nullptr) ? 1e-5f : *epsilonPtr;

    // x desc：仅支持 ND/NCHW（二者同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化轴 R；
    // GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受）
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xFormat = xDesc->GetOriginFormat();
    xDtypeSize_ = (xDesc->GetDataType() == ge::DT_FLOAT16) ? 2 : 4;

    // x shape（optional 输入缺席时框架压实存储，必须用 def 声明序访问器）
    auto xShape = context_->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    size_t dimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF(xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCHW,
                OP_LOGE(context_, "x origin format %d not supported (only ND/NCHW)", static_cast<int>(xFormat)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context_, "x dim num %zu < 2, not supported", dimNum), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(xStorageShape.GetDim(i) < 0, OP_LOGE(context_, "not supported dynamic shape info"),
                    return ge::GRAPH_FAILED);
    }

    numN_ = xStorageShape.GetDim(0);
    numC_ = xStorageShape.GetDim(1);
    innerSize_ = 1;
    for (size_t i = 2; i < dimNum; i++) {
        innerSize_ *= xStorageShape.GetDim(i);
    }
    units_ = numN_ * numC_;

    // 统计量（mean/variance）：proto/def 层 optional，实际必须提供——null 检查即拦截点；
    // 恒为 [N,C] 逻辑布局（fractal 组合下 [N,C1,1,1,C0]，内存等价），元素数必须等于 N*C
    int64_t ncPlanes = numN_ * numC_;
    auto meanShape = context_->GetOptionalInputShape(INPUT_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanShape);
    OP_CHECK_IF(
        meanShape->GetStorageShape().GetShapeSize() != ncPlanes,
        OP_LOGE(context_, "mean elements %ld != N*C %ld", meanShape->GetStorageShape().GetShapeSize(), ncPlanes),
        return ge::GRAPH_FAILED);
    auto varShape = context_->GetOptionalInputShape(INPUT_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    OP_CHECK_IF(
        varShape->GetStorageShape().GetShapeSize() != ncPlanes,
        OP_LOGE(context_, "variance elements %ld != N*C %ld", varShape->GetStorageShape().GetShapeSize(), ncPlanes),
        return ge::GRAPH_FAILED);

    // gamma/beta：optional，必须同有同无（910b TBE 语义）；存在时元素数同样须等于 N*C
    const gert::StorageShape* gammaShape = context_->GetOptionalInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(INPUT_BETA_INDEX);
    OP_CHECK_IF((gammaShape == nullptr) != (betaShape == nullptr),
                OP_LOGE(context_, "gamma and beta must both exist or both be absent"), return ge::GRAPH_FAILED);
    if (gammaShape != nullptr) {
        OP_CHECK_IF(
            gammaShape->GetStorageShape().GetShapeSize() != ncPlanes,
            OP_LOGE(context_, "gamma elements %ld != N*C %ld", gammaShape->GetStorageShape().GetShapeSize(), ncPlanes),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            betaShape->GetStorageShape().GetShapeSize() != ncPlanes,
            OP_LOGE(context_, "beta elements %ld != N*C %ld", betaShape->GetStorageShape().GetShapeSize(), ncPlanes),
            return ge::GRAPH_FAILED);
        hasGammaBeta_ = 1;
    }

    // batch_mean/batch_variance：optional 输出，防御式检测
    hasBatchMean_ = (context_->GetOutputShape(OUTPUT_BATCH_MEAN_INDEX) != nullptr) ? 1 : 0;
    hasBatchVar_ = (context_->GetOutputShape(OUTPUT_BATCH_VAR_INDEX) != nullptr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INInferV2Tiling::CalcCoreSplit()
{
    // 单元维主切分
    unitCores_ = units_ < coreNum_ ? units_ : coreNum_;
    if (unitCores_ < 1) {
        unitCores_ = 1; // 空 tensor 防御：单核空跑
    }

    // 单元不够分且 inner 维够大时，inner 维再切（每份至少一个完整向量）
    innerCores_ = 1;
    if (unitCores_ < coreNum_ && innerSize_ >= VL) {
        int64_t maxByInner = innerSize_ / VL;
        int64_t maxByCore = coreNum_ / unitCores_;
        innerCores_ = maxByCore < maxByInner ? maxByCore : maxByInner;
        if (innerCores_ < 1) {
            innerCores_ = 1;
        }
    }
    innerPerCore_ = (innerSize_ + innerCores_ - 1) / innerCores_;

    // 单元前多后少均分：前 formerCoreNum 核每核 formerUnits 个，其余每核 latterUnits 个
    int64_t base = units_ / unitCores_;
    int64_t rem = units_ % unitCores_;
    formerCoreNum_ = rem;
    formerUnits_ = (rem > 0) ? (base + 1) : base;
    latterUnits_ = base;

    // UB tile：x/y 双缓冲 fp32 16B/elem、fp16 8B/elem（regbase 在 reg 内解包/打包，无 fp32 中间 buffer）
    int64_t perElemBytes = (xDtypeSize_ == 2) ? (2 * 2 + 2 * 2) : (2 * 4 + 2 * 4);
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    ubTileSize_ = (ubAvail / perElemBytes) / VL * VL;
    // fp16 时批量透传 bulkChunk = ubTileSize/2 须 ≥VL（否则退化为 0 死循环），故 fp16 最小 2*VL
    int64_t minTile = (xDtypeSize_ == 2) ? (2 * VL) : VL;
    OP_CHECK_IF(ubTileSize_ < minTile,
                OP_LOGE(context_, "ub too small, ubTileSize=%ld (min %ld)", ubTileSize_, minTile),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INInferV2Tiling::FillTilingData()
{
    auto* tilingData = reinterpret_cast<INInferV2TilingData*>(context_->GetRawTilingData()->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->numN = numN_;
    tilingData->numC = numC_;
    tilingData->innerSize = innerSize_;
    tilingData->units = units_;
    tilingData->unitCores = unitCores_;
    tilingData->formerCoreNum = formerCoreNum_;
    tilingData->formerUnits = formerUnits_;
    tilingData->latterUnits = latterUnits_;
    tilingData->innerCores = innerCores_;
    tilingData->innerPerCore = innerPerCore_;
    tilingData->ubTileSize = ubTileSize_;
    tilingData->epsilon = epsilon_;
    tilingData->hasGammaBeta = hasGammaBeta_;
    tilingData->hasBatchMean = hasBatchMean_;
    tilingData->hasBatchVar = hasBatchVar_;
    context_->GetRawTilingData()->SetDataSize(sizeof(INInferV2TilingData));

    int64_t usedCores = unitCores_ * innerCores_;
    context_->SetBlockDim(usedCores);
    context_->SetTilingKey(0); // key 恒为 0（ND 单路径）；dtype 编译期双二进制，hasGammaBeta 运行时分发
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 无 workspace

    OP_LOGI(context_,
            "INInferV2 tiling: N=%ld, C=%ld, inner=%ld, units=%ld, unitCores=%ld, innerCores=%ld, "
            "usedCores=%ld, ubTile=%ld, hasGammaBeta=%ld, hasBatchMean=%ld, hasBatchVar=%ld, eps=%f",
            numN_, numC_, innerSize_, units_, unitCores_, innerCores_, usedCores, ubTileSize_, hasGammaBeta_,
            hasBatchMean_, hasBatchVar_, epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INInferV2Tiling::DoTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAndDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CalcCoreSplit() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FillTilingData();
}

static ge::graphStatus TilingForINInferV2(gert::TilingContext* context)
{
    INInferV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForINInferV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<INInferV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INInferV2).Tiling(TilingForINInferV2).TilingParse<INInferV2CompileInfo>(TilingPrepareForINInferV2);

} // namespace optiling
