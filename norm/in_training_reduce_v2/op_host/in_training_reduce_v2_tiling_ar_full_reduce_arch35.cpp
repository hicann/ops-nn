/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file in_training_reduce_v2_tiling_ar_full_reduce_arch35.cpp
 * \brief AR full_reduce（R 全载）切核 / UB 切分。裁剪自 instance_norm ar_full_reduce：
 *        删 gamma/beta/y/mean/var/rstd/meanFp32 项、删 avgFactor/epsilon 字段。
 */
#include <vector>
#include <algorithm>
#include "in_training_reduce_v2_tiling.h"

using namespace ge;

namespace optiling {
constexpr int64_t TILINGKEY_AR_FULL_REDUCE = 200000;
constexpr int64_t RESERVE_FOR_ALIGN = 512;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint32_t NUM_2 = 2;

bool INTrainingReduceV2ARFullReduceTiling::IsCapable()
{
    // 一期仅 NCHW / NCDHW / ND（NHWC/NDHWC C>1 二期）。
    if (format != FORMAT_NCHW && format != FORMAT_NCDHW && format != FORMAT_ND) {
        return false;
    }
    // A1-Main 骨架仅 R 全载路径：R=0 由后续 REDUCE_EMPTY 承接（本迭代不含）。
    if (r <= 0) {
        return false;
    }

    uint64_t ubfp32 = ubBlockSize / sizeof(float);
    // 数据类型的元素宽度
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16) {
        elemSize = FP16_BYTE;
    }

    uint64_t rAlign = Ops::Base::CeilAlign(r * elemSize, ubBlockSize) / elemSize;
    // 计算二分累加折叠点：小于 rAlign 的最大 2 的幂
    uint64_t binAddQuotient = rAlign == 0 ? 1 : (1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(rAlign)));
    binAddQuotient = (binAddQuotient == rAlign) ? binAddQuotient / NUM_2 : binAddQuotient;
    // 折叠临时缓存单行字节数（保守按字节计）
    uint64_t binAddBufferOneline = Ops::Base::CeilAlign((binAddQuotient + vlfp32 - 1) / vlfp32, ubfp32) * sizeof(float);

    // 单行 UB 占用（删 gamma/beta/y/mean/var/rstd/meanFp32）：
    //   inQueueX_:            rAlign * elemSize        # 双 buf → * 2
    //   outQueueSum_:         sizeof(float)            # 双 buf → * 2
    //   outQueueSquareSum_:   sizeof(float)            # 双 buf → * 2
    //   binaryAddBuf_:        binAddBufferOneline      # Σx  折叠 scratch
    //   squareBinaryAddBuf_:  binAddBufferOneline      # Σx² 折叠 scratch（独立，消除跨 VEC_SCOPE 冒险）
    uint64_t cInner = (aicoreParams_.ubSize - RESERVE_FOR_ALIGN) /
                      (rAlign * elemSize * NUM_2 + sizeof(float) * NUM_2 * NUM_2 + binAddBufferOneline * NUM_2);
    if (cInner < 1) {
        // 单行 R 全载超单次 UB 容量 → sub-R 分块路径（DESIGN §6.3 路 A），同 TilingKey + isSubRTiling 标志。
        return DoSubRTiling(rAlign, binAddQuotient, elemSize);
    }
    // 可全载的行数不超过 C
    cInner = std::min(cInner, static_cast<uint64_t>(a0));
    uint64_t cOuter = (a0 + cInner - 1) / cInner;
    uint64_t cTail = a0 - (cOuter - 1) * cInner;
    uint64_t totalTileCnt = cOuter * a1;
    uint64_t perCoreCnt = Ops::Base::CeilDiv(totalTileCnt, aicoreParams_.blockDim);
    blockNum_ = Ops::Base::CeilDiv(totalTileCnt, perCoreCnt);

    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = rAlign;
    td_.cInner = cInner;
    td_.cOuter = cOuter;
    td_.cTail = cTail;
    td_.binaryAddQuotient = binAddQuotient;
    td_.perCoreCnt = perCoreCnt;
    td_.isSubRTiling = 0;
    td_.rFactor = 0;
    td_.numChunks = 0;
    td_.tailLen = 0;
    return true;
}

// sub-R 分块 tiling（DESIGN §6.3 路 A）：R 超单次 UB 容量时，按 rFactor 分块搬入，
// 每块归约累进独立 fp32 累加器，跨块固定顺序树归约。按行(N*C)切核。
// rFactor 尽量取大 ⇒ numChunks 尽量少 ⇒ 部分和缓存尽量省；VL_FP32 对齐。
bool INTrainingReduceV2ARFullReduceTiling::DoSubRTiling(uint64_t rAlign, uint64_t binAddQuotient, int64_t elemSize)
{
    uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    // 输出双 buffer：sum + square_sum，各 CeilAlign(4,32)=32B，×2 buf
    uint64_t outReserve = Ops::Base::CeilAlign(sizeof(float), static_cast<uint64_t>(ubBlockSize)) * NUM_2 * NUM_2;
    // 部分和缓存上限（Σx/Σx² 各 ceil(numChunks/VL)*VL 个 fp32）；给足 1/8 usable。
    uint64_t partialReserve = usable / 8;
    uint64_t inputBudget = usable - outReserve - partialReserve;
    // 输入块双 buffer：rFactor * elemSize * 2
    uint64_t rFactor = inputBudget / (static_cast<uint64_t>(elemSize) * NUM_2);
    rFactor = rFactor / vlfp32 * vlfp32; // VL_FP32 对齐（下取整）
    if (rFactor < static_cast<uint64_t>(vlfp32)) {
        rFactor = vlfp32;
    }
    uint64_t rCeilVL = Ops::Base::CeilAlign(static_cast<uint64_t>(r), static_cast<uint64_t>(vlfp32));
    if (rFactor > rCeilVL) {
        rFactor = rCeilVL;
    }
    uint64_t numChunks = (static_cast<uint64_t>(r) + rFactor - 1) / rFactor;
    uint64_t tailLen = static_cast<uint64_t>(r) - (numChunks - 1) * rFactor;

    // 按行(N*C)切核：每行一个 (n,c) 的 R 个空间元素独立规约
    uint64_t totalRows = static_cast<uint64_t>(a1) * static_cast<uint64_t>(a0);
    uint64_t perCoreCnt = Ops::Base::CeilDiv(totalRows, aicoreParams_.blockDim);
    if (perCoreCnt < 1) {
        perCoreCnt = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(totalRows, perCoreCnt);

    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = rAlign;
    td_.cInner = 1;
    td_.cOuter = 1;
    td_.cTail = a0;
    td_.binaryAddQuotient = binAddQuotient;
    td_.perCoreCnt = perCoreCnt;
    td_.isSubRTiling = 1;
    td_.rFactor = rFactor;
    td_.numChunks = numChunks;
    td_.tailLen = tailLen;
    return true;
}

ge::graphStatus INTrainingReduceV2ARFullReduceTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

uint64_t INTrainingReduceV2ARFullReduceTiling::GetTilingKey() const { return TILINGKEY_AR_FULL_REDUCE; }

ge::graphStatus INTrainingReduceV2ARFullReduceTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(INTrainingReduceV2, INTrainingReduceV2ARFullReduceTiling,
                             IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_PRIORITY);
} // namespace optiling
