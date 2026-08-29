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
 * \brief BNTrainingUpdateGrad arch35 tiling（ND/NCHW + NHWC 双路径；GE 图模式可能下发
 *        NCHW/NHWC 标签，一并接受）
 *        ND 路径：N=d0, C=d1, inner=R=prod(d2:)；channel 主切分（前多后少，
 *        每 channel 的完整归约由唯一归属核完成，零核间通信、无 workspace）；
 *        核内 channel chunk(cLenCap) × R 分片(sliceR) × N 行 tile(rowsPerTile)。
 *        NHWC 路径（含 ND C==1 大规模 / ND R==1 巨 C 的 reroute，布局同构）：
 *        [rows, C] 行主序（C=最后一维）；channelSplit（C 大，零通信零 ws）或
 *        rowSplit（C 小，原子加直写输出 GM，零 ws 零 SyncAll，并行度恒=核数）。
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

// NHWC 路径常量
static constexpr int64_t NHWC_TILE_MAX_ROWS = 65535;     // DataCopyPad blockCount 上限（行 tile 行数）
static constexpr int64_t ND_REROUTE_MIN_ELEMS = 1 << 20; // ND C==1 reroute 阈值（≥1M 元素才值得多核行切）
static constexpr int64_t NHWC_R1_REROUTE_MIN_C = 2048; // ND R==1 reroute 阈值（C≤2K 时 ND 慢路自身放行，无需改道）

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
    // grads/x desc：支持 ND/NCHW（二者同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化轴 R；
    // GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受）与 NHWC（C=最后一维，A2 同款支持面）；
    // grads/x 必须同布局（ND/NCHW 同组或同为 NHWC），dtype 必须同型
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
    auto isPlaneFormat = [](ge::Format fmt) { return fmt == ge::FORMAT_ND || fmt == ge::FORMAT_NCHW; };
    OP_CHECK_IF(!isPlaneFormat(gradsFormat) && gradsFormat != ge::FORMAT_NHWC,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "grads", Ops::Base::ToString(gradsFormat).c_str(),
                                           "ND, NCHW or NHWC"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isPlaneFormat(xFormat) && xFormat != ge::FORMAT_NHWC,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(),
                                           "ND, NCHW or NHWC"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((gradsFormat == ge::FORMAT_NHWC) != (xFormat == ge::FORMAT_NHWC),
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "grads",
                                                        Ops::Base::ToString(gradsFormat).c_str(),
                                                        "grads and x must use the same layout family"),
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

    if (gradsFormat == ge::FORMAT_NHWC) {
        // NHWC：C=最后一维（逻辑序）。origin 与 storage shape 不同源时内存实际序不可推断，宁拒不猜
        auto gradsOriginShape = gradsShape->GetOriginShape();
        OP_CHECK_IF(gradsOriginShape.GetDimNum() != dimNum ||
                        gradsOriginShape.GetShapeSize() != gradsStorageShape.GetShapeSize(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grads",
                                                          Ops::Base::ToString(gradsOriginShape).c_str(),
                                                          "NHWC requires storage shape equal to origin shape"),
                    return ge::GRAPH_FAILED);
        return ParseNhwcShape(gradsStorageShape, dimNum);
    }
    numN_ = gradsStorageShape.GetDim(0);
    numC_ = gradsStorageShape.GetDim(1);
    innerSize_ = 1;
    for (size_t i = 2; i < dimNum; i++) {
        innerSize_ *= gradsStorageShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateGradTiling::ParseNhwcShape(const gert::Shape& gradsStorageShape, size_t dimNum)
{
    // NHWC：[rows, C] 行主序，C=最后一维（内存最内），rows=numel/C（=N·H·W…，归约轴展平）
    numC_ = gradsStorageShape.GetDim(dimNum - 1);
    int64_t numel = gradsStorageShape.GetShapeSize();
    rows_ = numel / numC_;
    numN_ = rows_; // 语义 N*H*W（仅日志/核对口径，kernel 按行切分消费 rows）
    innerSize_ = 1;
    isNhwc_ = true;
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
    if (isNhwc_) {
        return CalcNhwcSplit();
    }
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

int64_t BNTrainingUpdateGradTiling::SolveNhwcWindowMax() const
{
    // NHWC UB 预算（与 kernel InitBuffer 严格一致）：fp32 常驻 8 条 W+VL 缓冲
    // （statMean/statVar/rstd/accOff/accScale/kahanOff/kahanScale/out，共 8*(W+VL)*4B）。
    //   Wmax = floor64((ub − 保留项 − 8*VL*4) / (32 + 4*dsize))
    // 每元素 32B = 常驻 fp32 8 字节 + g/x 双队列×DOUBLE_BUFFER 4 份 × dsize
    int64_t fixed = RESERVED_UB + 8 * VL * FLOAT_BYTES;
    int64_t wAfford = (ubSize_ - fixed) / (8 * FLOAT_BYTES + 4 * xDtypeSize_);
    return wAfford / 64 * 64; // 64 对齐：UB 行基址 32B 对齐、行尾 64 元向量无掩码读不越 pitch
}

int64_t BNTrainingUpdateGradTiling::SolveNhwcTileRows(int64_t window) const
{
    // 行 tile：g/x 双队列 × DOUBLE_BUFFER(2) × tileRows × W × dsize；常驻项随 W 结算
    int64_t fixed = RESERVED_UB + 8 * (window + VL) * FLOAT_BYTES;
    int64_t tileRows = (ubSize_ - fixed) / (4 * window * xDtypeSize_);
    if (tileRows > NHWC_TILE_MAX_ROWS) {
        tileRows = NHWC_TILE_MAX_ROWS; // DataCopyPad blockCount 上限
    }
    return tileRows;
}

void BNTrainingUpdateGradTiling::SplitRangeAcrossCores(int64_t total)
{
    // 切分维（通道段/行段共用）：前多后少均分，复用 cFormer* 字段
    int64_t base = total / channelCores_;
    int64_t rem = total % channelCores_;
    cFormerCoreNum_ = rem;
    cFormerLen_ = (rem > 0) ? (base + 1) : base;
    cLatterLen_ = base;
}

ge::graphStatus BNTrainingUpdateGradTiling::CalcNhwcSplit()
{
    // rowSplit 谓词：rows ≥ coreNum（满核占用）且 W=ceil64(C) 在 UB 预算内（C 上限由
    // 窗预算决定，~4.7K@fp32）；否则 channelSplit（零通信，继承 ND 哲学）。rowSplit 的
    // 跨核合并走原子加直写输出（零 workspace；ws 段写入实测 VECTOR_CORE_EXCEPTION）。
    int64_t pitchC = (numC_ + VL - 1) / VL * VL;
    bool rowSplitOk = (rows_ >= coreNum_) && (pitchC <= SolveNhwcWindowMax());
    if (rowSplitOk) {
        nhwcSplitMode_ = 2;
        channelCores_ = coreNum_;
        SplitRangeAcrossCores(rows_); // cFormer* 语义=行段
        cLenCap_ = pitchC;            // 单窗全 C（W 预算内 ⇒ 必单窗成立）
        if (numC_ == 1) {
            // C==1 reroute：kernel 走整段 1D（ND 快路 AccumChunk 形态），sliceR 语义复用为
            // chunk 元素数（快路公式：4 队列双缓冲 + 两路 fp32 scratch + Kahan/尾槽）
            int64_t ubAvail = ubSize_ - RESERVED_UB;
            int64_t fastFixed = (4 * (1 + VL) + (1 + VL) + 2 * (8 + VL)) * FLOAT_BYTES + REDUCE_TMP_RESERVE +
                                2 * VL * FLOAT_BYTES + 4 * VL * FLOAT_BYTES + 256;
            int64_t fastAvail = ubAvail - fastFixed - 4 * VL * xDtypeSize_ - 4 * VL * FLOAT_BYTES;
            int64_t chunk = fastAvail / (2 * 2 * xDtypeSize_ + 2 * FLOAT_BYTES);
            if (chunk > rows_) {
                chunk = rows_; // 段不超过行段总量（单段兜底）
            }
            // 不做 64 取整：AccumChunk 尾路径本就支持 <64 元素段（标量尾缓冲 + 补零归约），
            // 兼容 rows_<64 的退化形状（核数<64 的 SKU 上原生 NHWC [rows,1] 小行数场景）
            sliceR_ = chunk;
            OP_CHECK_IF(
                sliceR_ < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "nhwcC1Chunk",
                                                      std::to_string(sliceR_).c_str(), "ub too small for C==1 segment"),
                return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
    } else {
        nhwcSplitMode_ = 1;
        channelCores_ = numC_ < coreNum_ ? numC_ : coreNum_;
        if (channelCores_ < 1) {
            channelCores_ = 1; // 防御：单核空跑
        }
        SplitRangeAcrossCores(numC_); // cFormer* 语义=通道段
        wsBytes_ = 0;
        // 窗口宽 W 恒 64 对齐（向量块/UB 行距约束），按 C 全量而非段长取——段长 < W 的
        // 尾窗 cnt<W 由 kernel 死 lane 语义覆盖（acc 死 lane 恒 0，最终只写 [cnt] 元素）
        int64_t wMax = SolveNhwcWindowMax();
        int64_t window = pitchC < wMax ? pitchC : wMax;
        window = window / 64 * 64;
        OP_CHECK_IF(
            window < 64,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "nhwcWindow", std::to_string(window).c_str(),
                                                  "ub too small for NHWC window"),
            return ge::GRAPH_FAILED);
        cLenCap_ = window;
    }
    sliceR_ = SolveNhwcTileRows(cLenCap_);
    OP_CHECK_IF(sliceR_ < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "nhwcTileRows",
                                                      std::to_string(sliceR_).c_str(), "ub too small for NHWC tile"),
                return ge::GRAPH_FAILED);
    rowsPerTile_ = 1; // NHWC 不消费（行 tile 已由 sliceR=tileRows 承载）
    return ge::GRAPH_SUCCESS;
}

void BNTrainingUpdateGradTiling::ConvertNdToNhwcLayout()
{
    // ND→NHWC reroute（布局同构）：C==1 大规模（G22 单核 DMA-bound）或 R==1（含 rank2）
    // 巨 C（ND 慢路 rowsPerTile<1 拒收）。内存序不变，仅切换切分视角。
    isNhwc_ = true;
    rows_ = (numC_ == 1) ? numN_ * innerSize_ : numN_;
    numN_ = rows_;
    innerSize_ = 1;
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
    tilingData->isNhwc = isNhwc_ ? 1 : 0;
    tilingData->nhwcSplitMode = nhwcSplitMode_;
    tilingData->epsilon = epsilon_;
    tilingData->reserved = 0.0f;

    context_->SetBlockDim(channelCores_);
    context_->SetTilingKey(0); // key 恒为 0；ND/NHWC 运行时按 isNhwc 分发，dtype 编译期三二进制
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = static_cast<size_t>(wsBytes_); // rowSplit 部分和；ND/channelSplit 恒 0

    if (isNhwc_) {
        OP_LOGI(context_,
                "BNTrainingUpdateGrad tiling(NHWC): rows=%ld, C=%ld, mode=%ld, cores=%ld, window=%ld, tileRows=%ld, "
                "wsBytes=%ld, eps=%f",
                rows_, numC_, nhwcSplitMode_, channelCores_, cLenCap_, sliceR_, wsBytes_, epsilon_);
    } else {
        OP_LOGI(context_,
                "BNTrainingUpdateGrad tiling: N=%ld, C=%ld, R=%ld, channelCores=%ld, cLenCap=%ld, sliceR=%ld, "
                "rowsPerTile=%ld, eps=%f",
                numN_, numC_, innerSize_, channelCores_, cLenCap_, sliceR_, rowsPerTile_, epsilon_);
    }
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
    // ND→NHWC reroute（先于 CalcCoreSplit，不触发 ND 慢路拒收报错）：
    // ① C==1 大规模（≥1M 元素）：单核 DMA-bound 场景切到 NHWC rowSplit 多核（G22）；
    // ② R==1（含 rank2）：ND 慢路在巨 C 下 rowsPerTile<1 拒收（C ≳ 3.8K 起），布局与
    //    NHWC [rows,C] 同构，按 NHWC 重切消除 UB 容量类拒收点（G27 登记项）
    if (!isNhwc_ && numC_ == 1 && numN_ * innerSize_ >= ND_REROUTE_MIN_ELEMS) {
        ConvertNdToNhwcLayout();
    } else if (!isNhwc_ && innerSize_ == 1 && numC_ > NHWC_R1_REROUTE_MIN_C) {
        ConvertNdToNhwcLayout();
    }
    return CalcCoreSplit() == ge::GRAPH_SUCCESS ? FillTilingData() : ge::GRAPH_FAILED;
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
