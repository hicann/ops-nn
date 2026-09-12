/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_hash_table_apply_adam_w_tiling_arch35.cpp
 * \brief
 */

#include "embedding_hash_table_apply_adam_w_tiling_arch35.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t INPUT_KEYS_IDX = 1;
constexpr uint32_t INPUT_VALUES_IDX = 2;
constexpr uint32_t ATTR_EMBEDDINGDIM_IDX = 0;
constexpr uint32_t ATTR_TABLE_SIZE_IDX = 1;
constexpr uint32_t MAX_UINT32 = 4294967295;
// SIMT 访存合并（fp32 生效）分派条件。生效窗口经 950PR 全 dim 扫描标定
// （N=1M B=2M，2026-09-11，数据见 e2e_driver/perf_adamw_dim_sweep*.csv）：
//   - 主窗口 dim 6~128 偶数（除 26/62）：合并 1.04~1.72x。本 kernel 小 dim 时
//     probe/key 并行度主导，blockX 封顶 4（dim>=88 为 8），bx>=16 全面劣化；
//     dim=26/62 实测 0.975~0.978x（5 次复测稳定，非噪声），回退 legacy；
//   - dim 130~256：元素 MLP 主导，旧大 blockX 单 pass 流式布局已近最优，合并
//     实测 0.88~0.96x 赢不了，回退 legacy（dim=256 是旧公式单 pass 最优点 0.88x）；
//   - dim>256：旧公式 blockNum=N*ceil(dim/256)，超出 keyNum 的 block 全部空转
//     （kernel 内 idx=block_idx 起即 >=keyNum），probe 摊销全灭。此区间改用
//     probe 摊销布局（legacy kernel 内 xLoop 一次 probe 走全程，
//     xLoopSize=ceil(dim/blockX) 由 kernel 推导，全 dtype/奇偶生效且与旧公式
//     位级一致）：fp32 bx=32/by=16 实测 257~736 档 1.06~1.49x、奇数档 1.13~1.16x；
//     fp16 bx=64/by=8（2026-09-12 v7 全档复测标定，见下）257~1024 全档 1.02~2.98x；
//     fp32 偶数 dim>736 元素吞吐主导，合并访存 VF 反超
//     （768:1.08x/1024:1.22x），仅该段走合并 VF（blockX 封顶 32）；
//     合并 VF 在 257~736 实测 0.90~1.18x 不敌纯布局，不启用。
//   - 奇数 dim / dim<6：本就 legacy。
// kernel 侧 Process() 分派条件必须与 IsMergeFp32Dim 逐字一致（tiling/kernel 布局契约）。
constexpr uint32_t MERGE_FP32_DIM_MIN = 6;
constexpr uint32_t MERGE_FP32_DIM_MAX = 128;
constexpr uint32_t MERGE_FP32_XLOOP_DIM = 256; // 旧公式 dim>256 切 xLoop 的起点
constexpr uint32_t MERGE_VF_HIGH_DIM = 736;    // 合并 VF 高窗口起点（736 处与纯布局打平）
constexpr uint32_t MERGE_BLOCK_X_CAP = 4;
constexpr uint32_t MERGE_BLOCK_X_CAP_LARGE_DIM = 8;
constexpr uint32_t MERGE_BLOCK_X_CAP_XLOOP = 32;
constexpr uint32_t MERGE_LARGE_DIM_THRESHOLD = 88;
constexpr uint32_t XLOOP_LAYOUT_BLOCK_X = 32; // dim>256 probe 摊销布局的 blockX（fp32）
// （fp32 侧 bx=64 实测在 dim≡32(mod64) 档存在跨进程双峰塌缩 0.85x~1.2x，bx=32 全档稳定 >=1.06x）
constexpr uint32_t XLOOP_LAYOUT_BLOCK_X_FP16 = 64; // fp16/bf16 专用（2026-09-12 v7 标定）：
// fp16 bx=32 在 even 344~352 / 奇数 641~767 实测 0.93~0.99 稳定回退（多进程复测非噪声），
// bx=64 后 257~1024 全 768 档 1.017~2.976x（含全部 ≡32(mod64) 档），多上下文复测最差 0.997；
// fp16 bx=16 全面崩（0.64~0.86）。两 dtype 布局分叉是实测结果，勿合并。

namespace {
inline bool IsMergeFp32Dim(uint32_t bitWidth, uint32_t embeddingDim)
{
    if (bitWidth != 4 || embeddingDim % 2 != 0) {
        return false;
    }
    if (embeddingDim > MERGE_VF_HIGH_DIM) {
        return true; // 高窗口 fp32 偶数 dim>736：合并 VF 反超纯布局（768:1.08x/1024:1.22x）
    }
    // 主窗口 [6,128]：26/62 实测回退除外；130~256 实测不敌旧布局
    return embeddingDim >= MERGE_FP32_DIM_MIN && embeddingDim <= MERGE_FP32_DIM_MAX && embeddingDim != 26 &&
           embeddingDim != 62;
}

inline uint32_t nextPowerOf2(uint32_t x)
{
    if (x == 0) {
        return 1;
    }
    x--;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}
} // namespace

// values support datatype
static const std::unordered_map<ge::DataType, uint64_t> VALUES_DATA_TYPE_TO_INT{{ge::DataType::DT_FLOAT16, 2},
                                                                                {ge::DataType::DT_BF16, 2},
                                                                                {ge::DataType::DT_FLOAT, 4},
                                                                                {ge::DataType::DT_INT32, 4},
                                                                                {ge::DataType::DT_INT64, 8}};

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(opName, "EmbeddingHashTableApplyAdamWTiling fail to get coreNum."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetShapeAttrsInfo()
{
    auto const keyShape = context_->GetInputShape(INPUT_KEYS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    auto const keyShapeVal = keyShape->GetStorageShape();
    int64_t keyShapeSize = keyShapeVal.GetShapeSize();
    OP_CHECK_IF((keyShapeSize < 0) || (keyShapeSize > MAX_UINT32),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName, "keys", std::to_string(keyShapeSize).c_str(),
                    "The shape size of keys must be in the representable range of the uint32_t type"),
                return ge::GRAPH_FAILED);
    keyNum_ = static_cast<uint32_t>(keyShapeSize);

    auto values = context_->GetInputDesc(INPUT_VALUES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, values);
    auto valuesDtype = values->GetDataType();

    auto iter = VALUES_DATA_TYPE_TO_INT.find(valuesDtype);
    if (iter != VALUES_DATA_TYPE_TO_INT.end()) {
        bitWidth_ = iter->second;
    } else {
        OP_LOGD(context_->GetNodeName(), "valuesDtype = %u not supported. please check.", valuesDtype);
        return ge::GRAPH_FAILED;
    }

    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const uint32_t* tableSize = attrs->GetAttrPointer<uint32_t>(ATTR_TABLE_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tableSize);
    tableSize_ = static_cast<uint32_t>(*tableSize);
    const uint32_t* embeddingDim = attrs->GetAttrPointer<uint32_t>(ATTR_EMBEDDINGDIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, embeddingDim);
    embeddingDim_ = static_cast<uint32_t>(*embeddingDim);
    const bool* amsgrad = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amsgrad);
    amsgrad_ = static_cast<uint32_t>(*amsgrad);
    const bool* maximize = attrs->GetAttrPointer<bool>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maximize);
    maximize_ = static_cast<uint32_t>(*maximize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::DoOpTiling()
{
    tilingData.set_keyNum(keyNum_);
    tilingData.set_bitWidth(bitWidth_);
    tilingData.set_tableSize(tableSize_);
    tilingData.set_embeddingDim(embeddingDim_);
    tilingData.set_amsgrad(amsgrad_);
    tilingData.set_maximize(maximize_);

    if (IsMergeFp32Dim(bitWidth_, embeddingDim_)) {
        // fp32 合并访存布局（kernel 侧同条件分派）：TX 沿 dim 每线程连续 merge 元素，
        // blockX 主窗口封顶 4（dim>=88 为 8），高窗口（>736）封顶 32，见文件头注释
        uint32_t merge = (embeddingDim_ % 4 == 0) ? 4 : 2;
        uint32_t cap = MERGE_BLOCK_X_CAP;
        if (embeddingDim_ > MERGE_VF_HIGH_DIM) {
            cap = MERGE_BLOCK_X_CAP_XLOOP;
        } else if (embeddingDim_ >= MERGE_LARGE_DIM_THRESHOLD) {
            cap = MERGE_BLOCK_X_CAP_LARGE_DIM;
        }
        blockX_ = nextPowerOf2((embeddingDim_ + merge - 1) / merge);
        if (blockX_ > cap) {
            blockX_ = cap;
        }
        blockY_ = MAX_THREAD / blockX_;
        blockNum_ = (keyNum_ + blockY_ - 1) / blockY_;
    } else if (embeddingDim_ > MERGE_FP32_XLOOP_DIM) {
        // probe 摊销布局（全 dtype/奇偶）：legacy kernel 内 xLoop 一次 probe 走全程，
        // blockNum 不再乘 ceil(dim/256)（旧公式超 keyNum 的 block 全空转），见文件头注释。
        // blockX 按 dtype 分叉：fp16/bf16=64、其余（fp32/int）=32，均为全档扫描标定值
        blockX_ = (bitWidth_ == 2) ? XLOOP_LAYOUT_BLOCK_X_FP16 : XLOOP_LAYOUT_BLOCK_X;
        blockY_ = MAX_THREAD / blockX_;
        blockNum_ = (keyNum_ + blockY_ - 1) / blockY_;
    } else if (embeddingDim_ < MAX_THREAD) {
        blockX_ = (embeddingDim_ + MIN_THREAD - 1) / MIN_THREAD * MIN_THREAD;
        blockY_ = MAX_THREAD / blockX_;
        blockNum_ = (keyNum_ + blockY_ - 1) / blockY_;
    } else {
        blockX_ = MAX_THREAD;
        blockY_ = 1;
        blockNum_ = keyNum_ * ((embeddingDim_ + MAX_THREAD - 1) / MAX_THREAD);
    }
    tilingData.set_blockX(blockX_);
    tilingData.set_blockY(blockY_);
    tilingData.set_blockNum(blockNum_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t EmbeddingHashTableApplyAdamWTiling::GetTilingKey() const
{
    uint64_t tilingKey = 100; // 100: embedding base tilingKey
    return tilingKey + bitWidth_;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(std::min(static_cast<uint32_t>(blockNum_), coreNum_));
    context_->SetScheduleMode(1);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4EmbeddingHashTableApplyAdamW(gert::TilingContext* context)
{
    EmbeddingHashTableApplyAdamWTiling hashTiling(context);
    return hashTiling.DoTiling();
}

static ge::graphStatus TilingPrepare4EmbeddingHashTableApplyAdamW(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableApplyAdamW)
    .Tiling(Tiling4EmbeddingHashTableApplyAdamW)
    .TilingParse<EmbeddingHashTableApplyAdamWCompileInfo>(TilingPrepare4EmbeddingHashTableApplyAdamW);
} // namespace optiling
