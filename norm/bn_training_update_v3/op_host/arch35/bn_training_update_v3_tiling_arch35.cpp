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
 * \brief BNTrainingUpdateV3 arch35 tiling（ND/NHWC 双路径；GE 图模式可能下发 NCHW 标签，布局同 ND 一并接受）
 *        ND：N=d0, C=d1, inner=R=prod(d2:)，units=N*C 个连续 plane；
 *        NHWC：C=最后一维，rows=numel/C=num（numRecip 分母，与 ND 的 N*R 同为 numel/C），
 *        三路径分派（向量访存 32B 对齐约束 ⇒ 系数切片仅 64 对齐整块）：
 *        Flat（C%64==0 且 C≤12288，静态 pattern=coeff，plane=64 元向量块）/
 *        Stream（C%64==0 且 C>12288，驻留 12288 chunk 环）/
 *        Rows（C%64!=0 任意 C，plane=一行，行距 pitch 逐行 DataCopyPad；行预算超 UB 拒收）。
 *        统计量恒为 [C] 逻辑布局，元素数=C 校验；numRecip=fp32(1/num)；
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

// NHWC 三路径分派与 pattern 预算
static constexpr int64_t NHWC_PATTERN_LIMIT = 12288;  // Flat/Stream 分界（C%64==0 且 ≤此值 → Flat）
static constexpr int64_t NHWC_RESIDENT_LIMIT = 20480; // pattern 全驻留元素上限（双 pattern 2*C*4B ≤ 160KB；
                                                      // 超出才启用滑动环——避免大 C 反复重载系数）
static constexpr int64_t NHWC_STAT_BULK_ELEMS = 1024; // Flat/Stream 统计量 bulk staging 粒度（4 队列 × 4KB）
static constexpr int64_t NHWC_ROWS_SEG_ELEMS = 2048; // RowsBlocked 分段长度（x/y 段 tile 与系数窗均与 C 无关，
                                                     // odd-C 支持面无上限）
static constexpr int64_t NHWC_PATH_FLAT = 1;         // C%64==0 且 C≤12288：静态 pattern（=coeff 本身）
static constexpr int64_t NHWC_PATH_STREAM = 2;       // C%64==0 且 C>12288：驻留 12288 chunk 环
static constexpr int64_t NHWC_PATH_ROWS = 3;         // C%64!=0 且整行预算内：行距 pitch 逐行搬运
static constexpr int64_t NHWC_PATH_ROWS_BLOCKED = 4; // C%64!=0 且整行预算外：分段流式（odd-C 无上限）

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
    // numRecip 分母 num：ND 为 N*R，NHWC 为 rows，数值上同为 numel/C
    int64_t num = isNhwc_ ? rows_ : (numN_ * innerSize_);
    // numRecip = 1/num：fp64 计算后舍入 fp32，与 A2 TBE tvm.const(1.0/num, float32) 语义一致
    numRecip_ = static_cast<float>(1.0 / static_cast<double>(num));
    // batchVarScaler = num/(num-1)（无偏方差修正因子）：fp64 计算后舍入 fp32，与 A2 TBE
    // float(num)/(num-1) 的 python float 语义一致；num==1 时为 0.0（A2 TBE 同口径特判）
    batchVarScaler_ = (num == 1) ? 0.0f : static_cast<float>(static_cast<double>(num) / static_cast<double>(num - 1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::CheckXDescAndShape()
{
    // x desc：支持 ND/NCHW/NHWC（ND 与 NCHW 同为 plane 连续语义：dim0=N、dim1=C、后导维为归一化
    // 轴 R，GE 图模式可能把 ND 归一化成 NCHW 标签下发，须一并接受；NHWC 为 C=最后一维的倒置布局）
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
    OP_CHECK_IF(xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(),
                                           "ND, NCHW or NHWC"),
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

    if (xFormat == ge::FORMAT_NHWC) {
        return ParseNhwcShape(xStorageShape, dimNum);
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

// NHWC：C=最后一维（内存最内），rows=前导维乘积=num（numRecip/batchVarScaler 分母）；
// units/innerSize 由 CalcNhwcSplit 按路径填（Flat/Stream=向量块数/64，Rows=rows/1）
ge::graphStatus BNTrainingUpdateV3Tiling::ParseNhwcShape(const gert::Shape& xStorageShape, size_t dimNum)
{
    isNhwc_ = true;
    numC_ = xStorageShape.GetDim(dimNum - 1);
    int64_t numel = 1;
    for (size_t i = 0; i < dimNum; i++) {
        numel *= xStorageShape.GetDim(i);
    }
    rows_ = numel / numC_;
    numN_ = rows_; // 语义 N*H*W（host 日志与调试核对口径）
    return SelectNhwcPath();
}

ge::graphStatus BNTrainingUpdateV3Tiling::SelectNhwcPath()
{
    // 三路径分派（对齐约束：MicroAPI 向量 load/store 32B 对齐 ⇒ 系数切片只能取 64 对齐整块）：
    //   Flat（1）  C%64==0 且 C≤12288：静态 pattern（pattern[j]=coeff[j%C]，周期恰为 C，
    //              chunk staging 落址恒 64 对齐），向量 v 取 pattern 向量 v mod (C/64)
    //   Stream（2）C%64==0 且 C>12288：pattern 按预算驻留 min(C,RESIDENT_LIMIT) 元（C≤RESIDENT_LIMIT
    //              即全驻留一次装载；更大才滑动重载），周期仍为 C
    //   Rows（3）  C%64!=0（任意 C）：无 64 对齐的无旋转切片，改行距 pitch 逐行 DataCopyPad，
    //              行内按 64-chunk 取 coeff 连续段（无旋转，天然对齐）
    if (numC_ % VL == 0) {
        nhwcPath_ = (numC_ <= NHWC_PATTERN_LIMIT) ? NHWC_PATH_FLAT : NHWC_PATH_STREAM;
    } else {
        nhwcPath_ = NHWC_PATH_ROWS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNTrainingUpdateV3Tiling::CalcNhwcSplit()
{
    if (nhwcPath_ == NHWC_PATH_ROWS) {
        // Rows：plane=一行（C 元素，逐行 1D DataCopyPad），units=rows，inner 恒 1
        units_ = rows_;
        innerSize_ = 1;
        innerCores_ = 1;
        innerPerCore_ = 1;
        SplitPlanesAcrossCores();
        return CalcNhwcRowsUbTile();
    }

    // Flat/Stream：plane=一个 64 元向量块，units=ceil(numel/64)（尾块 masked）
    int64_t numel = rows_ * numC_;
    units_ = (numel + VL - 1) / VL;
    innerSize_ = VL;
    innerCores_ = 1; // innerSize==VL，inner 维无再切空间
    innerPerCore_ = VL;
    SplitPlanesAcrossCores();
    return CalcNhwcPatternUbTile();
}

// plane 前多后少均分（ND/NHWC 共用模型）：unitCores=min(units,coreNum)（防御下限 1），
// 前 formerCoreNum 核每核 formerUnits 个 plane，其余每核 latterUnits 个
void BNTrainingUpdateV3Tiling::SplitPlanesAcrossCores()
{
    unitCores_ = units_ < coreNum_ ? units_ : coreNum_;
    if (unitCores_ < 1) {
        unitCores_ = 1; // 防御：单核空跑
    }
    int64_t base = units_ / unitCores_;
    int64_t rem = units_ % unitCores_;
    formerCoreNum_ = rem;
    formerUnits_ = (rem > 0) ? (base + 1) : base;
    latterUnits_ = base;
}

// Rows 行距 pitch 的 UB tile：pitch 64 元素对齐（kernel 行尾 64 元向量无掩码读不越界，兼保证
// 32B 行基址对齐）。系数 buffer 驻留 2×ceil64(C)×4B（一次构建，替代 RESERVED_UB 内 256 项槽位）；
// x/y 双队列各 DOUBLE_BUFFER=2 份行 tile，共 4×tileRows×rowBytes；ubTileSize 复用为 tileRows。
// 整行预算放不下（odd-C 超大）时切 RowsWindowed（nhwcPath=4）：c 窗口外层 × 行内层——系数窗
// W 元从任意通道偏移按 64 对齐直算重建（无拷贝拼接，规避 VEC 340），每窗流式处理本核全部行的
// 对应段；全部 UB 占用（4W×dtype + 2W×4 + bulk staging）与 C 无关 → odd-C 支持面无上限，且
// 每核系数计算总量仍为 C/64（每通道恰好一次，与快路径同量）
ge::graphStatus BNTrainingUpdateV3Tiling::CalcNhwcRowsUbTile()
{
    int64_t rowBytes = ((numC_ + VL - 1) / VL * VL) * xDtypeSize_;
    int64_t coeffBytes = 2 * ((numC_ + VL - 1) / VL * VL) * FLOAT_BYTES - AFFINE_PRECOMPUTE_BYTES;
    int64_t statBulkBytes = 4 * NHWC_STAT_BULK_ELEMS * FLOAT_BYTES - STAT_STAGING_BYTES;
    int64_t ubAvail = ubSize_ - RESERVED_UB - coeffBytes - statBulkBytes;
    ubTileSize_ = ubAvail / (4 * rowBytes);
    if (ubTileSize_ >= 1) {
        return ge::GRAPH_SUCCESS; // 快路径：整行 tile + 系数一次驻留
    }
    // 窗口流式：W = min(ceil64(C), UB 可容纳)，64 对齐；ubTileSize 复用为 W（c 窗口宽度）。
    // 预算：RESERVED_UB + (2W×4 − AFFINE_PRE) + statBulk + 4W×dtypeSize ≤ ub
    nhwcPath_ = NHWC_PATH_ROWS_BLOCKED;
    int64_t wAfford = (ubSize_ - RESERVED_UB - statBulkBytes + AFFINE_PRECOMPUTE_BYTES) /
                      (2 * FLOAT_BYTES + 4 * xDtypeSize_);
    int64_t wMax = (wAfford / VL) * VL;
    OP_CHECK_IF(wMax < VL,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "ubTileSize", "0",
                    ("NHWC rows path: C=" + std::to_string(numC_) + " window budget exceeds UB").c_str()),
                return ge::GRAPH_FAILED);
    int64_t pitch = (numC_ + VL - 1) / VL * VL;
    ubTileSize_ = (pitch < wMax) ? pitch : wMax;
    return ge::GRAPH_SUCCESS;
}

// Flat/Stream 的 UB tile：扣除 pattern 驻留（min(C,RESIDENT_LIMIT) 元素 ×2）与统计量 bulk
// staging（4 队列 × 1024 × 4B，替代 RESERVED_UB 内的 64 粒度 STAT_STAGING_BYTES）后按 x/y 双缓冲均分
ge::graphStatus BNTrainingUpdateV3Tiling::CalcNhwcPatternUbTile()
{
    int64_t patternVecs = (numC_ < NHWC_RESIDENT_LIMIT) ? numC_ : NHWC_RESIDENT_LIMIT;
    int64_t patternBytes = 2 * patternVecs * FLOAT_BYTES;
    int64_t statBulkBytes = 4 * NHWC_STAT_BULK_ELEMS * FLOAT_BYTES - STAT_STAGING_BYTES;
    int64_t ubAvail = ubSize_ - RESERVED_UB - patternBytes - statBulkBytes;
    OP_CHECK_IF(
        ubAvail <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubAvail", std::to_string(ubAvail).c_str(),
                                              "ub too small after NHWC pattern budget"),
        return ge::GRAPH_FAILED);

    // UB tile：x/y 双缓冲 fp32 16B/elem、fp16/bf16 8B/elem（regbase 在 reg 内解包/打包）
    int64_t perElemBytes = (xDtypeSize_ == 2) ? (2 * 2 + 2 * 2) : (2 * 4 + 2 * 4);
    ubTileSize_ = (ubAvail / perElemBytes) / VL * VL;
    int64_t minTile = (xDtypeSize_ == 2) ? (2 * VL) : VL;
    OP_CHECK_IF(ubTileSize_ < minTile,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubTileSize",
                                                      std::to_string(ubTileSize_).c_str(),
                                                      ("ub too small, min tile is " + std::to_string(minTile)).c_str()),
                return ge::GRAPH_FAILED);
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
    if (isNhwc_) {
        return CalcNhwcSplit();
    }

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
    SplitPlanesAcrossCores();

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
    tilingData->isNhwc = isNhwc_ ? 1 : 0;
    tilingData->nhwcPath = nhwcPath_;
    tilingData->epsilon = epsilon_;
    tilingData->numRecip = numRecip_;
    tilingData->batchVarScaler = batchVarScaler_;
    tilingData->reserved = 0.0f;

    int64_t usedCores = unitCores_ * innerCores_;
    context_->SetBlockDim(usedCores);
    context_->SetTilingKey(0); // key 恒为 0（ND/NHWC 单 key，kernel 运行时按 isNhwc 分发）；dtype 编译期三二进制
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 无 workspace

    OP_LOGI(context_,
            "BNTrainingUpdateV3 tiling: N=%ld, C=%ld, inner=%ld, units=%ld, unitCores=%ld, innerCores=%ld, "
            "usedCores=%ld, ubTile=%ld, eps=%f, numRecip=%e, batchVarScaler=%e, isNhwc=%d, nhwcPath=%ld, rows=%ld",
            numN_, numC_, innerSize_, units_, unitCores_, innerCores_, usedCores, ubTileSize_, epsilon_, numRecip_,
            batchVarScaler_, tilingData->isNhwc, tilingData->nhwcPath, rows_);
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
