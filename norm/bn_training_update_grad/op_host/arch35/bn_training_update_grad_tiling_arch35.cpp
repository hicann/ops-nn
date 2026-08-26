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
 * \file bn_training_update_grad_tiling_arch35.cpp
 * \brief BNTrainingUpdateGrad arch35 tiling（ND-only；GE 图模式可能下发 NCHW 标签，一并接受）
 *        统一切分模型：N=d0, C=d1, inner=R=prod(d2:)；channel 主切分（前多后少，
 *        每 channel 的完整归约由唯一归属核完成，零核间通信、无 workspace）；
 *        核内 channel chunk(cLenCap) × R 分片(sliceR) × N 行 tile(rowsPerTile)。
 *        统计量 batch_mean/batch_variance 恒为 [C] 逻辑布局，元素数=C 校验。
 */

#include "register/op_impl_registry.h"
#include "bn_training_update_grad_tiling_arch35.h"
#include "log/log.h"

using namespace optiling;
using namespace ge;

namespace optiling {

static constexpr int64_t FLOAT_BYTES = 4;
static constexpr int64_t VL = 64; // fp32 向量寄存器宽度（各缓冲 +VL 槽位用）

// UB 预留分量（不可用于 tile 的部分），RESERVED_UB 为各项之和
static constexpr int64_t PIPE_META_RESERVE = 8512;  // TPipe/TQue 元数据与事件表
static constexpr int64_t REDUCE_TMP_RESERVE = 1024; // ReduceSum sharedTmpBuffer（reuse-source 路径实际不用）
static constexpr int64_t MISC_RESERVE = 2048;       // 对齐余量
static constexpr int64_t RESERVED_UB = PIPE_META_RESERVE + REDUCE_TMP_RESERVE + MISC_RESERVE;

static constexpr int64_t Q_CAP_MAX = 4096;     // channel chunk 元素（cLen*sliceR）上限
static constexpr int64_t C_LEN_CAP_MAX = 4096; // channel chunk 长度上限

static constexpr size_t INPUT_GRADS_INDEX = 0;
static constexpr size_t INPUT_X_INDEX = 1;
static constexpr size_t INPUT_BATCH_MEAN_INDEX = 2;
static constexpr size_t INPUT_BATCH_VAR_INDEX = 3;
static constexpr size_t ATTR_EPSILON_INDEX = 0;

static constexpr float DEFAULT_EPSILON = 0.0001f; // 对齐 A2 proto .ATTR(epsilon, Float, 0.0001)

ge::graphStatus BNTrainingUpdateGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const BNTrainingUpdateGradCompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus BNTrainingUpdateGradTiling::GetShapeAndDtype()
{
    // attr epsilon（OPTIONAL，缺省 0.0001，对齐 A2 proto）
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(ATTR_EPSILON_INDEX);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    if (CheckGradsXDescAndShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckStatInputs();
}

ge::graphStatus BNTrainingUpdateGradTiling::CheckGradsXDescAndShape()
{
    // grads/x desc：仅支持 ND/NCHW（二者同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化轴 R；
    // GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受）；dtype 必须同型
    auto gradsDesc = context_->GetInputDesc(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsDesc);
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto gradsFormat = gradsDesc->GetOriginFormat();
    auto xFormat = xDesc->GetOriginFormat();
    auto gradsDtype = gradsDesc->GetDataType();
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(gradsDtype != ge::DT_FLOAT16 && gradsDtype != ge::DT_FLOAT && gradsDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grads", Ops::Base::ToString(gradsDtype).c_str(),
                                          "float16/float32/bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDtype != gradsDtype,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                          "must be the same as grads"),
                return ge::GRAPH_FAILED);
    xDtypeSize_ = (gradsDtype == ge::DT_FLOAT) ? 4 : 2;
    OP_CHECK_IF(gradsFormat != ge::FORMAT_ND && gradsFormat != ge::FORMAT_NCHW,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "grads", Ops::Base::ToString(gradsFormat).c_str(),
                                           "ND or NCHW"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCHW,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(), "ND or NCHW"),
        return ge::GRAPH_FAILED);

    auto gradsShape = context_->GetRequiredInputShape(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsShape);
    auto xShape = context_->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto gradsStorageShape = gradsShape->GetStorageShape();
    auto xStorageShape = xShape->GetStorageShape();
    size_t dimNum = gradsStorageShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grads",
                                                      Ops::Base::ToString(gradsStorageShape).c_str(),
                                                      "dim num must be no less than 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xStorageShape.GetDimNum() != dimNum,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xStorageShape).c_str(),
                                              "dim num must equal grads"),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(gradsStorageShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grads",
                                                             std::to_string(gradsStorageShape.GetDim(i)).c_str(),
                                                             "dynamic shape dim is not supported in tiling"),
                    return ge::GRAPH_FAILED);
        // 空 tensor 不支持（归约语义，空轴无和；A2 proto 同约束）——结构化拒绝
        OP_CHECK_IF(gradsStorageShape.GetDim(i) == 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grads",
                                                             std::to_string(gradsStorageShape.GetDim(i)).c_str(),
                                                             "empty tensor is not supported"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xStorageShape.GetDim(i) != gradsStorageShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                             std::to_string(xStorageShape.GetDim(i)).c_str(),
                                                             "shape must equal grads"),
                    return ge::GRAPH_FAILED);
    }

    numN_ = gradsStorageShape.GetDim(0);
    numC_ = gradsStorageShape.GetDim(1);
    innerSize_ = 1;
    for (size_t i = 2; i < dimNum; i++) {
        innerSize_ *= gradsStorageShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateGradTiling::CheckStatInputs()
{
    // 统计量（batch_mean/batch_variance）：REQUIRED 输入，恒为 [C] 逻辑布局，元素数必须等于 C 且恒 fp32
    std::string cReason = "elements must equal C (" + std::to_string(numC_) + ")";
    const size_t statIndexes[2] = {INPUT_BATCH_MEAN_INDEX, INPUT_BATCH_VAR_INDEX};
    const char* statNames[2] = {"batch_mean", "batch_variance"};
    for (size_t i = 0; i < 2; i++) {
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

ge::graphStatus BNTrainingUpdateGradTiling::CalcCoreSplit()
{
    // channel 维主切分：前 cFormerCoreNum 核每核 cFormerLen 个 channel，其余 cLatterLen 个
    channelCores_ = numC_ < coreNum_ ? numC_ : coreNum_;
    if (channelCores_ < 1) {
        channelCores_ = 1; // 防御：单核空跑
    }
    int64_t base = numC_ / channelCores_;
    int64_t rem = numC_ % channelCores_;
    cFormerCoreNum_ = rem;
    cFormerLen_ = (rem > 0) ? (base + 1) : base;
    cLatterLen_ = base;
    int64_t cRangeMax = cFormerLen_ > cLatterLen_ ? cFormerLen_ : cLatterLen_;

    // UB 反推：cLenCap/sliceR 决定后按"每行字节数"反推 rowsPerTile，与 kernel InitBuffer 严格一致：
    //   每行开销 = grads/x 双缓冲队列 2*2*pitchElems*dsize（行距 32B 对齐）
    //   固定开销 = 展开系数 2*qCap*4B + 二维累加器 accRow 2*qCap*4B + 归约 dst 2*cLenPad*4B
    //              + 每 channel 固定 6*cLenCap*4B（stat 2 路/rstd/acc 2 路/outQue）
    //              + 各缓冲 +VL 槽位 + 碎片余量
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    int64_t budgetQcap = ubAvail / 64; // 保守上界（每行开销 + 固定开销 + slack）
    int64_t qCap = budgetQcap < Q_CAP_MAX ? budgetQcap : Q_CAP_MAX;
    cLenCap_ = qCap / innerSize_;
    if (cLenCap_ < 1) {
        cLenCap_ = 1;
    }
    if (cLenCap_ > cRangeMax) {
        cLenCap_ = cRangeMax;
    }
    if (cLenCap_ > C_LEN_CAP_MAX) {
        cLenCap_ = C_LEN_CAP_MAX;
    }
    sliceR_ = qCap / cLenCap_;
    if (sliceR_ > innerSize_) {
        sliceR_ = innerSize_;
    }
    if (sliceR_ < 1) {
        sliceR_ = 1;
    }
    int64_t qCapActual = cLenCap_ * sliceR_;
    int64_t pitchElems = (qCapActual * xDtypeSize_ + 31) / 32 * 32 / xDtypeSize_;
    int64_t cLenPad = (cLenCap_ + 7) / 8 * 8;
    if (cLenCap_ == 1) {
        // 快路(单 channel chunk):每 (n,c) 段连续 1D,sliceR 改作 1D chunk 元素数;
        // buffer = g/x 双缓冲队列 + 两路 fp32 scratch + 小固定项(无 qCap 级展开系数/二维累加器)
        //   小固定 = stat 2 路 + rstd + acc 2 路 + outQue(各 1+VL)+ dst 2 路(8+VL) + reduceTmp + 尾槽
        int64_t fastFixed = (4 * (1 + VL) + (1 + VL) + 2 * (8 + VL)) * FLOAT_BYTES + REDUCE_TMP_RESERVE +
                            2 * VL * FLOAT_BYTES /* 尾部 64 槽 */ + 4 * VL * FLOAT_BYTES /* Kahan 补偿 2 块 */ + 256;
        // 精确预算:4 个队列缓冲(双缓冲 g/x,各 chunk+VL)+ 2 路 fp32 scratch(各 chunk+2VL)
        int64_t fastAvail = ubAvail - fastFixed - 4 * VL * xDtypeSize_ - 4 * VL * FLOAT_BYTES;
        int64_t chunk = fastAvail / (2 * 2 * xDtypeSize_ + 2 * FLOAT_BYTES); // 双缓冲队列 + 两路 fp32 scratch
        chunk = chunk / 64 * 64; // 64 对齐:多数 chunk 免尾部标量路径
        // 不按 innerSize 钳制:sliceR 同时是 C==1 摊平段/R==1 整平面/小 R 批处理的缓冲
        // 容量依据(实测 256x3 例钳到 8 后标量路径失效);chunk 由 UB 精确预算,大了不越界
        OP_CHECK_IF(chunk < 8,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "chunk",
                                                          std::to_string(chunk).c_str(), "ub too small for fast path"),
                    return ge::GRAPH_FAILED);
        sliceR_ = chunk;
        rowsPerTile_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    int64_t perRowBytes = 2 * 2 * pitchElems * xDtypeSize_;
    int64_t vlSlackBytes = VL * (4 * xDtypeSize_ + 11 * FLOAT_BYTES); // 各缓冲 +VL 槽位
    int64_t fixedBytes = 4 * qCapActual * FLOAT_BYTES + 2 * cLenPad * FLOAT_BYTES + 6 * cLenCap_ * FLOAT_BYTES +
                         vlSlackBytes + 256;
    rowsPerTile_ = (ubAvail - fixedBytes) / perRowBytes;
    if (rowsPerTile_ > numN_) {
        rowsPerTile_ = numN_;
    }
    if (rowsPerTile_ > 65535) { // DataCopyPad blockCount 上限
        rowsPerTile_ = 65535;
    }
    OP_CHECK_IF(
        rowsPerTile_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "rowsPerTile",
                                              std::to_string(rowsPerTile_).c_str(), "ub too small for one tile row"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateGradTiling::FillTilingData()
{
    auto* tilingData = context_->GetTilingData<BNTrainingUpdateGradTilingData>(); // 含容量检查与 SetDataSize
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->numN = numN_;
    tilingData->numC = numC_;
    tilingData->innerSize = innerSize_;
    tilingData->channelCores = channelCores_;
    tilingData->cFormerCoreNum = cFormerCoreNum_;
    tilingData->cFormerLen = cFormerLen_;
    tilingData->cLatterLen = cLatterLen_;
    tilingData->cLenCap = cLenCap_;
    tilingData->sliceR = sliceR_;
    tilingData->rowsPerTile = rowsPerTile_;
    tilingData->epsilon = epsilon_;
    tilingData->reserved = 0.0f;

    context_->SetBlockDim(channelCores_);
    context_->SetTilingKey(0); // key 恒为 0（ND 单路径）；dtype 编译期三二进制
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 无 workspace（channel 归属唯一，零核间通信）

    OP_LOGI(context_,
            "BNTrainingUpdateGrad tiling: N=%ld, C=%ld, R=%ld, channelCores=%ld, cLenCap=%ld, sliceR=%ld, "
            "rowsPerTile=%ld, eps=%f",
            numN_, numC_, innerSize_, channelCores_, cLenCap_, sliceR_, rowsPerTile_, epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateGradTiling::DoTiling()
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

static ge::graphStatus TilingForBNTrainingUpdateGrad(gert::TilingContext* context)
{
    BNTrainingUpdateGradTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBNTrainingUpdateGrad(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<BNTrainingUpdateGradCompileInfo>();
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

IMPL_OP_OPTILING(BNTrainingUpdateGrad)
    .Tiling(TilingForBNTrainingUpdateGrad)
    .TilingParse<BNTrainingUpdateGradCompileInfo>(TilingPrepareForBNTrainingUpdateGrad);

} // namespace optiling
