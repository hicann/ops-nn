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
 * \file bn_training_update_v3_tiling_arch35.cpp
 * \brief BNTrainingUpdateV3 arch35 tiling（ND-only；GE 图模式可能下发 NCHW 标签，一并接受）
 *        统一切分模型：N=d0, C=d1, inner=R=prod(d2:)，units=N*C 个连续 plane；
 *        统计量恒为 [C] 逻辑布局，元素数=C 校验；numRecip=fp32(1/(N*R))；
 *        batchVarScaler=fp32(num/(num-1))（num==1 时 0.0）；无 workspace。
 */

#include "register/op_impl_registry.h"
#include "bn_training_update_v3_tiling_arch35.h"
#include "log/log.h"

using namespace optiling;
using namespace ge;

namespace optiling {

static constexpr int64_t VL = 64;                             // fp32 向量寄存器宽度
static constexpr int64_t CHUNK_CHANNELS = 64;                 // 统计量 staging 粒度（每份 64 × 4B = 256B）
static constexpr int64_t MERGE_CHANNELS = 4 * CHUNK_CHANNELS; // R==1 合并 tile 系数缓冲槽位（256 项）
static constexpr int64_t FLOAT_BYTES = 4;

// UB 预留分量（不可用于 x/y tile 的部分），RESERVED_UB 为各项之和
static constexpr int64_t PIPE_META_RESERVE = 8512; // TPipe/TQue 元数据与事件表
static constexpr int64_t STAT_STAGING_BYTES = 4 * CHUNK_CHANNELS *
                                              FLOAT_BYTES; // sum/square_sum/scale/offset staging（4 队列）
static constexpr int64_t AFFINE_PRECOMPUTE_BYTES = 2 * MERGE_CHANNELS *
                                                   FLOAT_BYTES; // multiplier/addend 预计算 TBuf（256 项槽位）
static constexpr int64_t MISC_RESERVE = 2048;                   // batch 统计量写出小队列与对齐余量
static constexpr int64_t RESERVED_UB = PIPE_META_RESERVE + STAT_STAGING_BYTES + AFFINE_PRECOMPUTE_BYTES + MISC_RESERVE;

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_SUM_INDEX = 1;
static constexpr size_t INPUT_SQUARE_SUM_INDEX = 2;
static constexpr size_t INPUT_SCALE_INDEX = 3;
static constexpr size_t INPUT_OFFSET_INDEX = 4;
static constexpr size_t ATTR_EPSILON_INDEX = 0;

ge::graphStatus BNTrainingUpdateV3Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const BNTrainingUpdateV3CompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus BNTrainingUpdateV3Tiling::GetShapeAndDtype()
{
    // attr epsilon（REQUIRED，proto/def 层强制；缺省即非法）
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(ATTR_EPSILON_INDEX);
    OP_CHECK_IF(epsilonPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "epsilon", "absent",
                                                      "epsilon is a required attr"),
                return ge::GRAPH_FAILED);
    epsilon_ = *epsilonPtr;

    if (CheckXDescAndShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStatInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // numRecip = 1/(N*R)：fp64 计算后舍入 fp32，与 A2 TBE tvm.const(1.0/num, float32) 语义一致
    numRecip_ = static_cast<float>(1.0 / static_cast<double>(numN_ * innerSize_));
    // batchVarScaler = num/(num-1)（无偏方差修正因子）：fp64 计算后舍入 fp32，与 A2 TBE
    // float(num)/(num-1) 的 python float 语义一致；num==1 时为 0.0（A2 TBE 同口径特判）
    int64_t num = numN_ * innerSize_;
    batchVarScaler_ = (num == 1) ? 0.0f : static_cast<float>(static_cast<double>(num) / static_cast<double>(num - 1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::CheckXDescAndShape()
{
    // x desc：仅支持 ND/NCHW（二者同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化轴 R；
    // GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受）
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xFormat = xDesc->GetOriginFormat();
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                          "float16/float32/bfloat16"),
                return ge::GRAPH_FAILED);
    xDtypeSize_ = (xDtype == ge::DT_FLOAT) ? 4 : 2;

    auto xShape = context_->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    size_t dimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF(
        xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCHW,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(), "ND or NCHW"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dimNum < 2,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xStorageShape).c_str(),
                                              "dim num must be no less than 2"),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(xStorageShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                             std::to_string(xStorageShape.GetDim(i)).c_str(),
                                                             "dynamic shape dim is not supported in tiling"),
                    return ge::GRAPH_FAILED);
        // 空 tensor 不支持（A2 proto 明示；num=N*R 作分母，任一维为 0 即除零）——结构化拒绝
        OP_CHECK_IF(xStorageShape.GetDim(i) == 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                             std::to_string(xStorageShape.GetDim(i)).c_str(),
                                                             "empty tensor is not supported"),
                    return ge::GRAPH_FAILED);
    }

    numN_ = xStorageShape.GetDim(0);
    numC_ = xStorageShape.GetDim(1);
    innerSize_ = 1;
    for (size_t i = 2; i < dimNum; i++) {
        innerSize_ *= xStorageShape.GetDim(i);
    }
    units_ = numN_ * numC_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::CheckStatInputs()
{
    // 统计量（sum/square_sum/scale/offset）：REQUIRED 输入，恒为 [C] 逻辑布局，元素数必须等于 C
    std::string cReason = "elements must equal C (" + std::to_string(numC_) + ")";
    const size_t statIndexes[4] = {INPUT_SUM_INDEX, INPUT_SQUARE_SUM_INDEX, INPUT_SCALE_INDEX, INPUT_OFFSET_INDEX};
    const char* statNames[4] = {"sum", "square_sum", "scale", "offset"};
    for (size_t i = 0; i < 4; i++) {
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

ge::graphStatus BNTrainingUpdateV3Tiling::CalcCoreSplit()
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

    // UB tile：x/y 双缓冲 fp32 16B/elem、fp16/bf16 8B/elem（regbase 在 reg 内解包/打包，无 fp32 中间 buffer）
    int64_t perElemBytes = (xDtypeSize_ == 2) ? (2 * 2 + 2 * 2) : (2 * 4 + 2 * 4);
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    ubTileSize_ = (ubAvail / perElemBytes) / VL * VL;
    // 16bit 时批量写出 bulkChunk = ubTileSize/2 须 ≥VL（否则退化为 0 死循环），故 16bit 最小 2*VL
    int64_t minTile = (xDtypeSize_ == 2) ? (2 * VL) : VL;
    OP_CHECK_IF(ubTileSize_ < minTile,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubTileSize",
                                                      std::to_string(ubTileSize_).c_str(),
                                                      ("ub too small, min tile is " + std::to_string(minTile)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::FillTilingData()
{
    auto* tilingData = context_->GetTilingData<BNTrainingUpdateV3TilingData>(); // 含容量检查与 SetDataSize
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
    tilingData->batchVarScaler = batchVarScaler_;
    tilingData->reserved = 0.0f;

    int64_t usedCores = unitCores_ * innerCores_;
    context_->SetBlockDim(usedCores);
    context_->SetTilingKey(0); // key 恒为 0（ND 单路径）；dtype 编译期三二进制
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 无 workspace

    OP_LOGI(context_,
            "BNTrainingUpdateV3 tiling: N=%ld, C=%ld, inner=%ld, units=%ld, unitCores=%ld, innerCores=%ld, "
            "usedCores=%ld, ubTile=%ld, eps=%f, numRecip=%e, batchVarScaler=%e",
            numN_, numC_, innerSize_, units_, unitCores_, innerCores_, usedCores, ubTileSize_, epsilon_, numRecip_,
            batchVarScaler_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::DoTiling()
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

static ge::graphStatus TilingForBNTrainingUpdateV3(gert::TilingContext* context)
{
    BNTrainingUpdateV3Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBNTrainingUpdateV3(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<BNTrainingUpdateV3CompileInfo>();
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

IMPL_OP_OPTILING(BNTrainingUpdateV3)
    .Tiling(TilingForBNTrainingUpdateV3)
    .TilingParse<BNTrainingUpdateV3CompileInfo>(TilingPrepareForBNTrainingUpdateV3);

} // namespace optiling
