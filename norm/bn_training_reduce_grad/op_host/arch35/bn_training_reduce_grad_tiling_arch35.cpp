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
 * \file bn_training_reduce_grad_tiling_arch35.cpp
 * \brief BNTrainingReduceGrad arch35 tiling（ND-only；GE 图模式可能下发 NCHW 标签，一并接受）
 *        统一切分模型：N=d0, C=d1, inner=R=prod(d2:)，units=N*C 个连续 plane；
 *        统计量恒为 [C] 逻辑布局，元素数=C 校验；grads 与 x 同 dtype 同 shape 校验；
 *        numRecip=fp32(1/(N*R))，negNumRecip=-numRecip；无 workspace。
 */

#include "register/op_impl_registry.h"
#include "bn_training_reduce_grad_tiling_arch35.h"
#include "log/log.h"

using namespace optiling;
using namespace ge;

namespace optiling {

static constexpr int64_t VL = 64;                             // fp32 向量寄存器宽度
static constexpr int64_t CHUNK_CHANNELS = 64;                 // 统计量 staging 粒度（每份 64 × 4B = 256B）
static constexpr int64_t MERGE_CHANNELS = 4 * CHUNK_CHANNELS; // R==1 合并 tile 系数缓冲槽位（256 项）
static constexpr int64_t FLOAT_BYTES = 4;
static constexpr float DEFAULT_EPSILON = 0.0001f; // 对齐 A2 proto .ATTR(epsilon, Float, 0.0001)

// UB 预留分量（不可用于 grads/x/y tile 的部分），RESERVED_UB 为各项之和
static constexpr int64_t PIPE_META_RESERVE = 8512; // TPipe/TQue 元数据与事件表
static constexpr int64_t STAT_STAGING_BYTES = 5 * CHUNK_CHANNELS *
                                              FLOAT_BYTES; // diff_scale/diff_offset/scale/batch_mean/batch_variance
                                                           // staging（5 队列）
static constexpr int64_t COEFF_PRECOMPUTE_BYTES = 3 * MERGE_CHANNELS *
                                                  FLOAT_BYTES; // multiplier/addend/mulScale 预计算 TBuf（256 项槽位）
static constexpr int64_t MISC_RESERVE = 2048;                  // 对齐余量
static constexpr int64_t RESERVED_UB = PIPE_META_RESERVE + STAT_STAGING_BYTES + COEFF_PRECOMPUTE_BYTES + MISC_RESERVE;

static constexpr size_t INPUT_GRADS_INDEX = 0;
static constexpr size_t INPUT_X_INDEX = 1;
static constexpr size_t INPUT_DIFF_SCALE_INDEX = 2;
static constexpr size_t INPUT_DIFF_OFFSET_INDEX = 3;
static constexpr size_t INPUT_SCALE_INDEX = 4;
static constexpr size_t INPUT_BATCH_MEAN_INDEX = 5;
static constexpr size_t INPUT_BATCH_VAR_INDEX = 6;
static constexpr size_t ATTR_EPSILON_INDEX = 0;

ge::graphStatus BNTrainingReduceGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const BNTrainingReduceGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "compileInfo", "null",
                                                          "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    }
    OP_CHECK_IF(coreNum_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "coreNum",
                                                      std::to_string(coreNum_).c_str(), "coreNum must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
                                                      std::to_string(ubSize_).c_str(), "ubSize must be positive"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::GetShapeAndDtype()
{
    // attr epsilon（OPTIONAL，proto 缺省 0.0001；缺失时取缺省值）
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(ATTR_EPSILON_INDEX);
    epsilon_ = (epsilonPtr != nullptr) ? *epsilonPtr : DEFAULT_EPSILON;

    if (CheckGradsDescAndShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStatInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // numRecip = 1/(N*R)：fp64 计算后舍入 fp32，与 A2 TBE tvm.const(1.0/num, float32) 语义一致；
    // negNumRecip 为 fp32 取负（IEEE 取负精确，对齐 A2 tvm.const(-num_bw, float32)）
    numRecip_ = static_cast<float>(1.0 / static_cast<double>(reduceNum_));
    negNumRecip_ = -numRecip_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::CheckGradsShapeValidity(const gert::Shape& gradsStorageShape)
{
    // rank >= 2；逐维拒绝动态维（-1 下发到 tiling 即拒）与空维（空 tensor 不支持，A2 proto
    // 明示；num=N*R 作分母，任一维为 0 即除零）——均结构化报错返回 GRAPH_FAILED
    size_t dimNum = gradsStorageShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grads",
                                                      Ops::Base::ToString(gradsStorageShape).c_str(),
                                                      "dim num must be no less than 2"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(gradsStorageShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grads",
                                                             std::to_string(gradsStorageShape.GetDim(i)).c_str(),
                                                             "dynamic shape dim is not supported in tiling"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradsStorageShape.GetDim(i) == 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grads",
                                                             std::to_string(gradsStorageShape.GetDim(i)).c_str(),
                                                             "empty tensor is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::CheckGradsDescAndShape()
{
    // grads desc：仅支持 ND/NCHW（二者同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化轴 R；
    // GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受）
    auto gradsDesc = context_->GetInputDesc(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsDesc);
    auto gradsFormat = gradsDesc->GetOriginFormat();
    auto gradsDtype = gradsDesc->GetDataType();
    OP_CHECK_IF(gradsDtype != ge::DT_FLOAT16 && gradsDtype != ge::DT_FLOAT && gradsDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grads", Ops::Base::ToString(gradsDtype).c_str(),
                                          "float16/float32/bfloat16"),
                return ge::GRAPH_FAILED);
    xDtypeSize_ = (gradsDtype == ge::DT_FLOAT) ? 4 : 2;

    // x 与 grads 同 dtype（def 位置配对语义；kernel 单模板 T 直通）
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(
        xDesc->GetDataType() != gradsDtype,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                  "must equal dtype of grads"),
        return ge::GRAPH_FAILED);

    auto gradsShape = context_->GetRequiredInputShape(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsShape);
    auto gradsStorageShape = gradsShape->GetStorageShape();
    OP_CHECK_IF(gradsFormat != ge::FORMAT_ND && gradsFormat != ge::FORMAT_NCHW,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "grads", Ops::Base::ToString(gradsFormat).c_str(),
                                           "ND or NCHW"),
                return ge::GRAPH_FAILED);
    if (CheckGradsShapeValidity(gradsStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED; // 具体原因已在 CheckGradsShapeValidity 内 OP_LOGE
    }

    // x 与 grads 同 shape（逐元素对应，A2 ELEWISE_WITH_BROADCAST 契约）
    auto xShape = context_->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OP_CHECK_IF(xShape->GetStorageShape() != gradsStorageShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      Ops::Base::ToString(xShape->GetStorageShape()).c_str(),
                                                      "shape must equal shape of grads"),
                return ge::GRAPH_FAILED);

    numN_ = gradsStorageShape.GetDim(0);
    numC_ = gradsStorageShape.GetDim(1);
    size_t dimNum = gradsStorageShape.GetDimNum(); // 已过 CheckGradsShapeValidity，恒 >= 2
    innerSize_ = 1;
    for (size_t i = 2; i < dimNum; i++) {
        innerSize_ *= gradsStorageShape.GetDim(i);
    }
    units_ = numN_ * numC_;
    reduceNum_ = numN_ * innerSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::CheckStatInputs()
{
    // 统计量（diff_scale/diff_offset/scale/batch_mean/batch_variance）：REQUIRED 输入，恒为 [C] 逻辑布局，
    // 元素数必须等于 C
    std::string cReason = "elements must equal C (" + std::to_string(numC_) + ")";
    const size_t statIndexes[5] = {INPUT_DIFF_SCALE_INDEX, INPUT_DIFF_OFFSET_INDEX, INPUT_SCALE_INDEX,
                                   INPUT_BATCH_MEAN_INDEX, INPUT_BATCH_VAR_INDEX};
    const char* statNames[5] = {"diff_scale", "diff_offset", "scale", "batch_mean", "batch_variance"};
    for (size_t i = 0; i < 5; i++) {
        auto statShape = context_->GetRequiredInputShape(statIndexes[i]);
        OP_CHECK_NULL_WITH_CONTEXT(context_, statShape);
        OP_CHECK_IF(statShape->GetStorageShape().GetShapeSize() != numC_,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), statNames[i],
                        std::to_string(statShape->GetStorageShape().GetShapeSize()).c_str(), cReason.c_str()),
                    return ge::GRAPH_FAILED);
        auto statDesc = context_->GetInputDesc(statIndexes[i]);
        OP_CHECK_NULL_WITH_CONTEXT(context_, statDesc);
        OP_CHECK_IF(statDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), statNames[i],
                                              Ops::Base::ToString(statDesc->GetDataType()).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::CalcCoreSplit()
{
    // plane 维主切分
    unitCores_ = units_ < coreNum_ ? units_ : coreNum_;
    if (unitCores_ < 1) {
        unitCores_ = 1; // 防御：单核空跑
    }

    // plane 不够分且 inner 维够大时，inner 维再切（每份至少一个完整向量）
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

    // plane 前多后少均分：前 formerCoreNum 核每核 formerUnits 个，其余每核 latterUnits 个
    int64_t base = units_ / unitCores_;
    int64_t rem = units_ % unitCores_;
    formerCoreNum_ = rem;
    formerUnits_ = (rem > 0) ? (base + 1) : base;
    latterUnits_ = base;

    // UB tile：grads/x/y 三路双缓冲 fp32 24B/elem、fp16/bf16 12B/elem（regbase 在 reg 内解包/打包，
    // 无 fp32 中间 buffer）
    int64_t perElemBytes = (xDtypeSize_ == 2) ? (3 * 2 + 3 * 2) : (3 * 4 + 3 * 4);
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    ubTileSize_ = (ubAvail / perElemBytes) / VL * VL;
    // 16bit 时尾块掩码最小单位为 VL，批量最小 2*VL（否则退化为 0 死循环）
    int64_t minTile = (xDtypeSize_ == 2) ? (2 * VL) : VL;
    OP_CHECK_IF(ubTileSize_ < minTile,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubTileSize",
                                                      std::to_string(ubTileSize_).c_str(),
                                                      ("ub too small, min tile is " + std::to_string(minTile)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::FillTilingData()
{
    auto* tilingData = context_->GetTilingData<BNTrainingReduceGradTilingData>(); // 含容量检查与 SetDataSize
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
    tilingData->numRecip = numRecip_;
    tilingData->negNumRecip = negNumRecip_;
    tilingData->reserved = 0.0f;

    int64_t usedCores = unitCores_ * innerCores_;
    context_->SetBlockDim(usedCores);
    context_->SetTilingKey(0); // key 恒为 0（ND 单路径）；dtype 编译期三二进制
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 无 workspace

    OP_LOGI(context_,
            "BNTrainingReduceGrad tiling: N=%ld, C=%ld, inner=%ld, units=%ld, unitCores=%ld, innerCores=%ld, "
            "usedCores=%ld, ubTile=%ld, eps=%f, reduceNum=%ld, numRecip=%e, negNumRecip=%e",
            numN_, numC_, innerSize_, units_, unitCores_, innerCores_, usedCores, ubTileSize_, epsilon_, reduceNum_,
            numRecip_, negNumRecip_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingReduceGradTiling::DoTiling()
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

static ge::graphStatus TilingForBNTrainingReduceGrad(gert::TilingContext* context)
{
    BNTrainingReduceGradTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBNTrainingReduceGrad(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<BNTrainingReduceGradCompileInfo>();
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

IMPL_OP_OPTILING(BNTrainingReduceGrad)
    .Tiling(TilingForBNTrainingReduceGrad)
    .TilingParse<BNTrainingReduceGradCompileInfo>(TilingPrepareForBNTrainingReduceGrad);

} // namespace optiling
