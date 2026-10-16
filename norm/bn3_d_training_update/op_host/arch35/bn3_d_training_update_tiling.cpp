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
 * \file bn3_d_training_update_tiling.cpp
 * \brief BN3DTrainingUpdate arch35 tiling 入口:校验 → 几何 → UB 反解 → SetTilingKey → 填 TilingData。
 */

#include "bn3_d_training_update_tiling.h"

#include <cmath>
#include <cinttypes>
#include <string>
#include "../../op_kernel/arch35/bn3_d_training_update_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

using namespace optiling;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelFirstTilingData;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelLastTilingData;

// ---------------------------- TilingData 注册(宿主侧与内核侧 struct 字段/次序逐条一致)-----------
BEGIN_TILING_DATA_DEF(Bn3dTrainingUpdateChannelFirstTilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, spatial);
TILING_DATA_FIELD_DEF(int64_t, spatialTile);
TILING_DATA_FIELD_DEF(int64_t, spatialTiles);
TILING_DATA_FIELD_DEF(int64_t, channelWindow);
TILING_DATA_FIELD_DEF(int64_t, baseTilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, extraCoreCount);
TILING_DATA_FIELD_DEF(int64_t, statUnitCount);
TILING_DATA_FIELD_DEF(int64_t, statChunk);
TILING_DATA_FIELD_DEF(int64_t, baseStatPerCore);
TILING_DATA_FIELD_DEF(int64_t, extraStatCount);
TILING_DATA_FIELD_DEF(float, numRec);
TILING_DATA_FIELD_DEF(float, besselScaler);
TILING_DATA_FIELD_DEF(float, factor);
TILING_DATA_FIELD_DEF(float, oneMinusFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(Bn3dTrainingUpdateChannelLastTilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, num);
TILING_DATA_FIELD_DEF(int64_t, channelBlock);
TILING_DATA_FIELD_DEF(int64_t, rowsPerTile);
TILING_DATA_FIELD_DEF(int64_t, tilesPerBlock);
TILING_DATA_FIELD_DEF(int64_t, baseTilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, extraCoreCount);
TILING_DATA_FIELD_DEF(int64_t, statUnitCount);
TILING_DATA_FIELD_DEF(int64_t, statChunk);
TILING_DATA_FIELD_DEF(int64_t, baseStatPerCore);
TILING_DATA_FIELD_DEF(int64_t, extraStatCount);
TILING_DATA_FIELD_DEF(int64_t, coeffCap);
TILING_DATA_FIELD_DEF(float, numRec);
TILING_DATA_FIELD_DEF(float, besselScaler);
TILING_DATA_FIELD_DEF(float, factor);
TILING_DATA_FIELD_DEF(float, oneMinusFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BN3DTrainingUpdate_2000, Bn3dTrainingUpdateChannelFirstTilingDataDef);
REGISTER_TILING_DATA_CLASS(BN3DTrainingUpdate_3000, Bn3dTrainingUpdateChannelLastTilingDataDef);

namespace {
// 算子操作数次序(与 def.cpp / proto.h 一致,供校验收敛取用):
//   输入 0=x, 1=sum, 2=square_sum, 3=scale, 4=offset, 5=mean, 6=variance
//   输出 0=y, 1=mean, 2=variance, 3=batch_mean, 4=batch_variance
constexpr int64_t INPUT_X = 0;
constexpr int64_t INPUT_SUM = 1;
constexpr int64_t INPUT_NUM = 7;
constexpr int64_t OUTPUT_Y = 0;
constexpr int64_t OUTPUT_MEAN = 1;
constexpr int64_t OUTPUT_NUM = 5;
constexpr int64_t ATTR_FACTOR = 0;
constexpr int64_t ATTR_EPSILON = 1;

// x 的通道轴(format 唯一决定;NCHW/NCDHW ⇒ 1,NHWC/NDHWC ⇒ 末轴)。
inline bool IsChannelLastFormat(ge::Format format) { return format == ge::FORMAT_NHWC || format == ge::FORMAT_NDHWC; }

inline int64_t ChannelAxis(ge::Format format, int64_t rank) { return IsChannelLastFormat(format) ? (rank - 1) : 1; }
} // namespace

ge::graphStatus Bn3dTrainingUpdateTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        plat_.coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        plat_.ubSize = static_cast<int64_t>(ubSize);
        plat_.vlFp32 = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)) / static_cast<int64_t>(sizeof(float));
        plat_.blockBytes = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
        plat_.libApiWorkspaceBytes = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    } else {
        auto compileInfo = reinterpret_cast<const Bn3dTrainingUpdateCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        plat_.coreNum = compileInfo->coreNum;
        plat_.ubSize = compileInfo->ubSize;
        plat_.vlFp32 = compileInfo->vectorLength / static_cast<int64_t>(sizeof(float));
        plat_.blockBytes = compileInfo->blockSize;
        plat_.libApiWorkspaceBytes = compileInfo->libApiWorkspaceBytes;
    }
    OP_CHECK_IF(plat_.coreNum <= 0 || plat_.vlFp32 <= 0 || plat_.blockBytes <= 0 ||
                    plat_.ubSize <= kPipeMetaReserveBytes + StatPhaseBytes(plat_),
                OP_LOGE(context_, "BN3DTrainingUpdate: invalid platform values core=%ld ub=%ld vl=%ld block=%ld",
                        plat_.coreNum, plat_.ubSize, plat_.vlFp32, plat_.blockBytes),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::CheckAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* factor = attrs->GetAttrPointer<float>(ATTR_FACTOR);
    const float* epsilon = attrs->GetAttrPointer<float>(ATTR_EPSILON);
    OP_CHECK_IF(factor == nullptr || epsilon == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "factor/epsilon", "nullptr",
                                                      "both attributes are REQUIRED"),
                return ge::GRAPH_FAILED);
    // factor:契约(def.cpp / proto.h)未声明范围约束,A2 亦不校验 ⇒ 不按取值范围拒绝,只保留有限性门
    // (NaN/Inf 会让 EMA 结果整体污染,属"无意义输入"而非"范围越界")。factor == 0 是 PyTorch momentum
    // 的合法用法(running 统计量冻结,mean_out == mean),硬拒会造成相对 A2 的支持面回退。
    OP_CHECK_IF(std::isnan(*factor) || std::isinf(*factor),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "factor",
                                                      std::to_string(*factor).c_str(), "factor must be a finite float"),
                return ge::GRAPH_FAILED);
    // epsilon:归一化分母为 sqrt(saveVar + epsilon),epsilon <= 0 时在常量特征(saveVar == 0)上退化为
    // 1/sqrt(0) = inf/NaN,产物无意义 ⇒ 保留 > 0 的拒绝(epsilon 无 0 值合法语义)。
    OP_CHECK_IF(
        std::isnan(*epsilon) || std::isinf(*epsilon) || !(*epsilon > 0.0f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "epsilon", std::to_string(*epsilon).c_str(),
                                              "epsilon must be a finite float > 0"),
        return ge::GRAPH_FAILED);
    factor_ = *factor;
    epsilon_ = *epsilon;
    oneMinusFactor_ = static_cast<float>(1.0 - static_cast<double>(factor_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::CheckInputs()
{
    // ---- x:rank / format / dtype ----
    const auto* xDesc = context_->GetInputDesc(INPUT_X);
    const gert::StorageShape* xShape = context_->GetRequiredInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    const int64_t rank = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    const ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    const bool knownFormat = xFormat == ge::FORMAT_NCHW || xFormat == ge::FORMAT_NCDHW || xFormat == ge::FORMAT_NHWC ||
                             xFormat == ge::FORMAT_NDHWC;
    if (!knownFormat) {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ge::TypeUtils::FormatToSerialString(xFormat).c_str(),
                                   "NCHW, NCDHW, NHWC or NDHWC");
        return ge::GRAPH_FAILED;
    }
    const bool rankFitsFormat = (rank == 4 && (xFormat == ge::FORMAT_NCHW || xFormat == ge::FORMAT_NHWC)) ||
                                (rank == 5 && (xFormat == ge::FORMAT_NCDHW || xFormat == ge::FORMAT_NDHWC));
    if (!rankFitsFormat) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                 "NCHW/NHWC require rank 4 and NCDHW/NDHWC require rank 5");
        return ge::GRAPH_FAILED;
    }
    xDtype_ = xDesc->GetDataType();
    const bool knownDtype = xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_FLOAT || xDtype_ == ge::DT_BF16;
    if (!knownDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                  "float16, float32 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    elemBytes_ = (xDtype_ == ge::DT_FLOAT) ? 4 : 2;
    channelLast_ = IsChannelLastFormat(xFormat);

    // ---- 逐轴:非负(N == 0 / 空间轴 == 0 属空 tensor,支持)----
    const gert::Shape& shape = xShape->GetStorageShape();
    for (int64_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) < 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(shape).c_str(),
                                                  "runtime dimensions must be non-negative");
            return ge::GRAPH_FAILED;
        }
    }

    // ---- 6 路统计量输入:shape 全等且 rank == 1;C = sum.shape[0](C == 0 属空张量,不拒绝)----
    const gert::Shape* statShape = nullptr;
    for (int64_t i = INPUT_SUM; i < INPUT_NUM; ++i) {
        const auto* desc = context_->GetInputDesc(i);
        const gert::StorageShape* stat = context_->GetRequiredInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_NULL_WITH_CONTEXT(context_, stat);
        if (desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), ("input" + std::to_string(i)).c_str(),
                                      ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), "float32");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& cur = stat->GetStorageShape();
        if (cur.GetDimNum() != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), ("input" + std::to_string(i)).c_str(),
                                                     std::to_string(cur.GetDimNum()).c_str(),
                                                     "every (C,) statistic input must be rank 1");
            return ge::GRAPH_FAILED;
        }
        if (statShape != nullptr && cur != *statShape) {
            const std::string shapes = Ops::Base::ToString(*statShape) + " and " + Ops::Base::ToString(cur);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "sum..variance", shapes.c_str(),
                                                   "all six (C,) statistic inputs must share one shape");
            return ge::GRAPH_FAILED;
        }
        statShape = &cur;
    }
    channel_ = statShape->GetDim(0);
    // C == 0 不拒绝:x 与所有 (C,) 统计量此时均为空张量,无相 X / 相 S 单元 ⇒ 产出"全空"tiling
    // (xTileCount = statUnitCount = 0),kernel 空跑即正确(输出为空,无内容需写)。

    // ---- 通道维与 C 对齐 ----
    const int64_t channelAxis = ChannelAxis(xFormat, rank);
    if (shape.GetDim(channelAxis) != channel_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(shape).c_str(),
                                              "the channel dimension of x must equal sum.shape[0]");
        return ge::GRAPH_FAILED;
    }

    // ---- 5 路输出:dtype 与 def.cpp 一致;y 与 x 同形 ----
    const auto* yDesc = context_->GetOutputDesc(OUTPUT_Y);
    const gert::StorageShape* yShape = context_->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    if (yDesc->GetDataType() != xDtype_) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str());
        return ge::GRAPH_FAILED;
    }
    if (yShape->GetStorageShape() != shape) {
        const std::string shapes = Ops::Base::ToString(shape) + " and " +
                                   Ops::Base::ToString(yShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapes.c_str(),
                                               "y shape must equal x shape");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = OUTPUT_MEAN; i < OUTPUT_NUM; ++i) {
        const auto* desc = context_->GetOutputDesc(i);
        const gert::StorageShape* outStat = context_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outStat);
        if (desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), ("output" + std::to_string(i)).c_str(),
                                      ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), "float32");
            return ge::GRAPH_FAILED;
        }

        const gert::Shape& outShape = outStat->GetStorageShape();
        if (outShape != *statShape) {
            const std::string shapes = Ops::Base::ToString(*statShape) + " and " + Ops::Base::ToString(outShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), ("output" + std::to_string(i)).c_str(),
                                                   shapes.c_str(),
                                                   "statistic outputs must share the statistic inputs' (C,) shape");
            return ge::GRAPH_FAILED;
        }
    }

    // ---- 几何量:num = 通道轴之外的元素个数 = N * S(空 batch ⇒ 0,走零统计契约)----
    batch_ = shape.GetDim(0);
    int64_t num = 1;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != channelAxis) {
            num *= shape.GetDim(i);
        }
    }
    num_ = num;
    spatial_ = (batch_ > 0) ? (num_ / batch_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::TilingChannelFirst()
{
    // ① Ss_ub:S 上"总占用 <= ubSize"的最大块长(总占用对 Ss 单调不减 ⇒ 二分)。K=1 起解(S == 0 时取 1,
    //    spatialTiles 自然为 0 ⇒ 无 X 单元,只有相 S 单元工作)。
    int64_t lo = 1;
    int64_t hi = spatial_ > 0 ? spatial_ : 1;
    int64_t ssUb = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (CfKernelBytes(mid, 1, elemBytes_, plat_) <= plat_.ubSize) {
            ssUb = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (ssUb < 1) {
        OP_LOGE(context_, "BN3DTrainingUpdate: cannot fit one channel-first tile in UB (ub=%ld stat=%ld)", plat_.ubSize,
                StatPhaseBytes(plat_));
        return ge::GRAPH_FAILED;
    }

    // ② K_ub:该 Ss 下"总占用 <= ubSize"的最大通道窗宽,再 clamp 到 ≤ C 且 ≤ kParamChunkLen
    //    (折叠一次处理完整个窗)。
    const int64_t kChunk = kParamChunkLen(plat_.vlFp32);
    int64_t kHi = channel_ < kChunk ? channel_ : kChunk;
    lo = 1;
    hi = kHi < 1 ? 1 : kHi;
    int64_t kUb = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (CfKernelBytes(ssUb, mid, elemBytes_, plat_) <= plat_.ubSize) {
            kUb = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (kUb < 1) {
        OP_LOGE(context_, "BN3DTrainingUpdate: cannot fit one channel window in UB (ub=%ld Ss=%" PRId64 ")",
                plat_.ubSize, ssUb);
        return ge::GRAPH_FAILED;
    }

    // ③ K = min(K_ub, Kcap_units):Kcap_units = floor(C·N·spatialTiles / unitsTarget) 保单元数达标。
    //    与 Cb = min(C, CbCap) 同形 —— **取小,不是阈值路由**。
    const int64_t unitsTarget = plat_.coreNum * kUnitsPerCoreTarget;
    const int64_t spatialTilesAtUb = spatial_ > 0 ? ((spatial_ + ssUb - 1) / ssUb) : 0;
    int64_t kcapUnits = 0;
    if (spatialTilesAtUb > 0 && batch_ > 0 && channel_ > 0) {
        kcapUnits = (channel_ * batch_ * spatialTilesAtUb) / unitsTarget; // 除法取整,无字面量阈值
    }
    channelWindow_ = kUb;
    if (kcapUnits >= 1 && kcapUnits < channelWindow_) {
        channelWindow_ = kcapUnits;
    }
    if (channelWindow_ < 1) {
        channelWindow_ = 1;
    }

    // ④ Ss = min(Ss_ub, max(SsFloor, ceil(S / ceil(unitsTarget / (cWindows·N))))):单元数不足时再缩空间块长。
    cWindows_ = (channel_ + channelWindow_ - 1) / channelWindow_;
    const int64_t parallelUnits = cWindows_ * batch_;
    const int64_t ssFloor = (channelWindow_ >= plat_.vlFp32) ? 1 :
                                                               ((plat_.vlFp32 + channelWindow_ - 1) / channelWindow_);
    spatialTile_ = TargetBlockLen(spatial_, ssUb, ssFloor, parallelUnits, unitsTarget);
    spatialTiles_ = (spatial_ + spatialTile_ - 1) / spatialTile_;
    xTileCount_ = cWindows_ * batch_ * spatialTiles_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::TilingChannelLast()
{
    const int64_t statPhaseBytes = StatPhaseBytes(plat_);
    const int64_t budget = plat_.ubSize - kPipeMetaReserveBytes - statPhaseBytes;
    // ① 通道块 Cb:最小"rows=1 且 VT=Cb 的占用 <= 预算"的 Cb(占用对 Cb 单调不减 ⇒ 二分)。
    //    C == 0 时无通道块(channelBlocks 自然为 0):上界取 1 只为解出 Cb=1 的合法缓冲形状,不产生 X 单元。
    int64_t lo = 1;
    int64_t hi = channel_ < 1 ? 1 : channel_;
    int64_t solvedBlock = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        const int64_t coeffCap = CoeffTileCap(mid, plat_.vlFp32);
        if (ClPhaseXBytes(1, mid, coeffCap, elemBytes_, plat_) <= budget) {
            solvedBlock = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (solvedBlock < 1) {
        OP_LOGE(context_, "BN3DTrainingUpdate: cannot fit one channel block in UB (ub=%ld)", plat_.ubSize);
        return ge::GRAPH_FAILED;
    }
    channelBlock_ = solvedBlock;
    coeffCap_ = CoeffTileCap(channelBlock_, plat_.vlFp32);

    // ② 每瓦片行数 rows:最大"占用 <= 预算"的行数;num == 0(空 batch)时取 1,tilesPerBlock 自然为 0。
    // 上界同时受 VF 循环归纳变量(uint16_t)约束:rows * channelBlock <= kMaxTileElementsPerVf。
    lo = 1;
    int64_t hiRows = num_ > 0 ? num_ : 1;
    // VF 循环计数器只能装 uint16_t(见 kMaxTileElementsPerVf 的依据),rows * channelBlock 必须 ≤ 该上界。
    // 若通道块本身就超过上界,反解无解 —— 显式报错,不做"静默夹到 rows=1"的兜底(那种分支仍违反被守的
    // 不变量却继续出包,与同文件其它反解失败路径"不留兜底"的口径不一致)。
    const int64_t rowsByVfBound = kMaxTileElementsPerVf / channelBlock_;
    if (rowsByVfBound < 1) {
        OP_LOGE(context_,
                "BN3DTrainingUpdate: channel block %ld exceeds the VF loop bound %" PRId64 " (rows cannot be resolved)",
                channelBlock_, kMaxTileElementsPerVf);
        return ge::GRAPH_FAILED;
    }
    if (hiRows > rowsByVfBound) {
        hiRows = rowsByVfBound;
    }
    hi = hiRows;
    int64_t solvedRows = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ClPhaseXBytes(mid, channelBlock_, coeffCap_, elemBytes_, plat_) <= budget) {
            solvedRows = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (solvedRows < 1) {
        OP_LOGE(context_, "BN3DTrainingUpdate: cannot fit one channel-last tile row in UB (ub=%ld)", plat_.ubSize);
        return ge::GRAPH_FAILED;
    }
    // ④ rows = min(rows_ub, max(rowsFloor, ceil(num / ceil(unitsTarget / channelBlocks)))):并行度目标
    //    rowsFloor 保证每单元(rows × Cb)至少一个整 VL 块。**kernel 零改动**。
    channelBlocks_ = (channel_ + channelBlock_ - 1) / channelBlock_;
    const int64_t unitsTarget = plat_.coreNum * kUnitsPerCoreTarget;
    const int64_t rowsFloor = (channelBlock_ >= plat_.vlFp32) ? 1 :
                                                                ((plat_.vlFp32 + channelBlock_ - 1) / channelBlock_);
    rowsPerTile_ = TargetBlockLen(num_, solvedRows, rowsFloor, channelBlocks_, unitsTarget);
    tilesPerBlock_ = (num_ + rowsPerTile_ - 1) / rowsPerTile_;
    xTileCount_ = channelBlocks_ * tilesPerBlock_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::FillChannelFirst()
{
    auto raw = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, raw);
    OP_CHECK_IF(raw->GetData() == nullptr ||
                    raw->GetCapacity() < Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES,
                OP_LOGE(context_, "BN3DTrainingUpdate: raw tiling capacity %zu is too small", raw->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto* data = reinterpret_cast<Bn3dTrainingUpdateChannelFirstTilingData*>(raw->GetData());
    data->c = channel_;
    data->spatial = spatial_;
    data->spatialTile = spatialTile_;
    data->spatialTiles = spatialTiles_;
    data->channelWindow = channelWindow_;
    data->baseTilesPerCore = baseTilesPerCore_;
    data->extraCoreCount = extraCoreCount_;
    data->statUnitCount = statUnitCount_;
    data->statChunk = statChunk_;
    data->baseStatPerCore = baseStatPerCore_;
    data->extraStatCount = extraStatCount_;
    data->numRec = numRec_;
    data->besselScaler = besselScaler_;
    data->factor = factor_;
    data->oneMinusFactor = oneMinusFactor_;
    data->epsilon = epsilon_;
    data->reserved = 0;
    raw->SetDataSize(Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    OP_CHECK_IF(context_->SetTilingKey(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "BN3DTrainingUpdate: failed to set the channel-first tiling key"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::FillChannelLast()
{
    auto raw = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, raw);
    OP_CHECK_IF(raw->GetData() == nullptr ||
                    raw->GetCapacity() < Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES,
                OP_LOGE(context_, "BN3DTrainingUpdate: raw tiling capacity %zu is too small", raw->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto* data = reinterpret_cast<Bn3dTrainingUpdateChannelLastTilingData*>(raw->GetData());
    data->c = channel_;
    data->num = num_;
    data->channelBlock = channelBlock_;
    data->rowsPerTile = rowsPerTile_;
    data->tilesPerBlock = tilesPerBlock_;
    data->baseTilesPerCore = baseTilesPerCore_;
    data->extraCoreCount = extraCoreCount_;
    data->statUnitCount = statUnitCount_;
    data->statChunk = statChunk_;
    data->baseStatPerCore = baseStatPerCore_;
    data->extraStatCount = extraStatCount_;
    data->coeffCap = coeffCap_;
    data->numRec = numRec_;
    data->besselScaler = besselScaler_;
    data->factor = factor_;
    data->oneMinusFactor = oneMinusFactor_;
    data->epsilon = epsilon_;
    data->reserved = 0;
    raw->SetDataSize(Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    OP_CHECK_IF(context_->SetTilingKey(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "BN3DTrainingUpdate: failed to set the channel-last tiling key"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Bn3dTrainingUpdateTiling::DoTiling()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("BN3DTrainingUpdate", "Tiling context is null"), return ge::GRAPH_FAILED);
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || CheckInputs() != ge::GRAPH_SUCCESS ||
        CheckAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // numRec 先求倒数(num == 0(含 C == 0 下 num 只由 x 的非通道轴定,可能仍 > 0)时 numRec = 0)。
    numRec_ = (num_ > 0) ? static_cast<float>(1.0 / static_cast<double>(num_)) : 0.0f;
    // Bessel:仅用于 running variance 的 EMA;num <= 1 一律 0(避免除零)。
    besselScaler_ = (num_ > 1) ? static_cast<float>(static_cast<double>(num_) / (static_cast<double>(num_) - 1.0)) :
                                 0.0f;

    if (channelLast_) {
        if (TilingChannelLast() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (TilingChannelFirst() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    statChunk_ = SolveStatChunk(channel_, plat_);
    statUnitCount_ = (channel_ + statChunk_ - 1) / statChunk_;
    usedCoreNum_ = plat_.coreNum;
    const int64_t workUnits = xTileCount_ > statUnitCount_ ? xTileCount_ : statUnitCount_;
    if (workUnits < usedCoreNum_) {
        usedCoreNum_ = workUnits < 1 ? 1 : workUnits;
    }
    baseTilesPerCore_ = xTileCount_ / usedCoreNum_;
    extraCoreCount_ = xTileCount_ % usedCoreNum_;
    baseStatPerCore_ = statUnitCount_ / usedCoreNum_;
    extraStatCount_ = statUnitCount_ % usedCoreNum_;

    const ge::graphStatus fill = channelLast_ ? FillChannelLast() : FillChannelFirst();
    if (fill != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "BN3DTrainingUpdate: failed to set block dim %ld", usedCoreNum_),
                return ge::GRAPH_FAILED);
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    // 本算子无跨核归约、无中间落盘,但框架自用部分仍按平台值预留(不写死常量)。
    workspace[0] = static_cast<size_t>(plat_.libApiWorkspaceBytes);
    OP_LOGI(context_,
            "BN3DTrainingUpdate tiling channelLast=%d C=%ld num=%ld xTiles=%ld cores=%ld statUnits=%ld statChunk=%ld "
            "spatialTile=%ld spatialTiles=%ld K=%ld rows=%ld channelBlock=%ld coeffCap=%ld eps=%f factor=%f",
            channelLast_ ? 1 : 0, channel_, num_, xTileCount_, usedCoreNum_, statUnitCount_, statChunk_, spatialTile_,
            spatialTiles_, channelWindow_, rowsPerTile_, channelBlock_, coeffCap_, epsilon_, factor_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForBN3DTrainingUpdate(gert::TilingContext* context)
{
    Bn3dTrainingUpdateTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBN3DTrainingUpdate(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BN3DTrainingUpdate", "Tiling parse context is null"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<Bn3dTrainingUpdateCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->vectorLength = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    compileInfo->blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    compileInfo->libApiWorkspaceBytes = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BN3DTrainingUpdate)
    .Tiling(TilingForBN3DTrainingUpdate)
    .TilingParse<Bn3dTrainingUpdateCompileInfo>(TilingPrepareForBN3DTrainingUpdate);
