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
#include <limits>
#include "in_training_reduce_v2_tiling.h"

using namespace ge;

namespace optiling {
constexpr int64_t TILINGKEY_AR_FULL_REDUCE = 200000;
constexpr int64_t RESERVE_FOR_ALIGN = 512;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint32_t NUM_2 = 2;
// 部分和缓存初始预留比例：numChunks 未知时先按 usable/8 估一版 rFactor，估偏了由联合求解收敛。
constexpr uint64_t PARTIAL_RESERVE_DIV = 8;
// 联合求解迭代上限：每轮 rFactor 严格变小，故必然停机，本上限只决定"停机前能不能收敛到解"。
// 常见 R（rFactor 还很大、numChunks 还很小）1 ~ 2 轮就够；但 R 逼近 UB 容量上限时 rFactor 已被
// partial 挤到很小，每轮只能再缩一点点，轮数会急剧上升。按 ubSize=253952 穷举实测，收敛所需最大
// 轮数在 fp32 R≈2.50e8 / fp16 R≈5.01e8（即 0.93 GiB 输入，partial 独占 UB 的真正无解点）附近达到
// 峰值 19 / 24 轮。取 32 留余量：小于此值会把本来有解的 shape 误拒（返回 false，落到无模板可用），
// 而不是下发越界 tiling —— 失败方向安全，但仍是错判。
constexpr uint32_t SUB_R_SOLVE_MAX_ITER = 32;

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

// sub-R 路径 Kernel 侧 UB 精确占用。逐项对应 InitSubR()：
//   inQueueX_          : CeilAlign(rFactor * elemSize, BLOCK_SIZE) * DOUBLE_BUFFER_NUM
//   sum/sqPartialBuf_  : CeilAlign(numChunks, VL_FP32) * sizeof(float)，两块
//   out 双 queue       : CeilAlign(sizeof(float), BLOCK_SIZE) * DOUBLE_BUFFER_NUM，两条
// 改 InitSubR() 的 buffer 规划时必须同步改这里，否则 Host 的容量校验会失真。
uint64_t INTrainingReduceV2ARFullReduceTiling::CalcSubRUbBytes(uint64_t rFactor, uint64_t numChunks,
                                                               int64_t elemSize) const
{
    uint64_t blockSize = static_cast<uint64_t>(ubBlockSize);
    uint64_t inBytes = Ops::Base::CeilAlign(rFactor * static_cast<uint64_t>(elemSize), blockSize) * NUM_2;
    uint64_t partialBytes = Ops::Base::CeilAlign(numChunks, static_cast<uint64_t>(vlfp32)) * sizeof(float) * NUM_2;
    uint64_t outBytes = Ops::Base::CeilAlign(sizeof(float), blockSize) * NUM_2 * NUM_2;
    return inBytes + partialBytes + outBytes;
}

// Kernel sub-R 路径把 numN / numC / numR / perCoreCnt 收窄成 uint32_t（见 ProcessSubR()），
// 且 InTrainingReduceV2SubR::Process() 用 uint32 乘出 numN_ * numC_。收窄前 Host 必须证明
// 这些值落在 32 位内，否则会静默截断成错误的行数 / 规约长度，而不是报错。
bool INTrainingReduceV2ARFullReduceTiling::CheckSubRNarrowable() const
{
    constexpr uint64_t maxU32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    uint64_t numN = static_cast<uint64_t>(a1);
    uint64_t numC = static_cast<uint64_t>(a0);
    uint64_t numR = static_cast<uint64_t>(r);
    if (numN > maxU32 || numC > maxU32 || numR > maxU32 || numC > maxU32 / numN) {
        OP_LOGE(context_->GetNodeName(), "sub-R tiling requires N, C, R and N*C to fit in uint32: N=%lu C=%lu R=%lu.",
                numN, numC, numR);
        return false;
    }
    return true;
}

// sub-R 分块 tiling（DESIGN §6.3 路 A）：R 超单次 UB 容量时，按 rFactor 分块搬入，
// 每块归约累进独立 fp32 累加器，跨块固定顺序树归约。按行(N*C)切核。
// rFactor 尽量取大 ⇒ numChunks 尽量少 ⇒ 部分和缓存尽量省；VL_FP32 对齐。
bool INTrainingReduceV2ARFullReduceTiling::DoSubRTiling(uint64_t rAlign, uint64_t binAddQuotient, int64_t elemSize)
{
    if (!CheckSubRNarrowable()) {
        return false;
    }
    uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    // 输出双 buffer：sum + square_sum，各 CeilAlign(4,32)=32B，×2 buf
    uint64_t outReserve = Ops::Base::CeilAlign(sizeof(float), static_cast<uint64_t>(ubBlockSize)) * NUM_2 * NUM_2;
    // 部分和缓存（Σx/Σx² 各 ceil(numChunks/VL)*VL 个 fp32）的初始预留：此刻 numChunks 还依赖
    // 待定的 rFactor，先按 1/8 usable 估一版，估偏了由下面的联合求解修正。
    uint64_t partialReserve = usable / PARTIAL_RESERVE_DIV;
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
    uint64_t numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);

    // 联合求解：partialReserve 只是估值，R 特别大时实际部分和缓存会撑破预留。按 Ascend950 实测
    // 参数（ubSize=253952，usable=253440，VL_FP32=64）首版公式的越界起点是 fp32 R=109707265
    // （rFactor=27648、numChunks=3969 ⇒ 申请 253568B > 253440B）、fp16 R=219668481
    // （rFactor=55360、numChunks=3969 ⇒ 申请 253824B）。这里用 Kernel 侧的精确公式
    // 复核总占用，超了就按实际 partial 占用重算 rFactor —— rFactor 变小 ⇒ 输入 buffer 省下的
    // 字节多于 numChunks 增加所需的槽位，故迭代单调收敛。收敛不到就明确拒绝，不下发越界 tiling。
    for (uint32_t iter = 0; iter < SUB_R_SOLVE_MAX_ITER; ++iter) {
        if (CalcSubRUbBytes(rFactor, numChunks, elemSize) <= usable) {
            break;
        }
        uint64_t partialBytes = Ops::Base::CeilAlign(numChunks, static_cast<uint64_t>(vlfp32)) * sizeof(float) * NUM_2;
        if (usable <= outReserve + partialBytes) {
            break; // 部分和缓存本身已占满 UB，无解
        }
        uint64_t nextBudget = usable - outReserve - partialBytes;
        uint64_t nextRFactor = nextBudget / (static_cast<uint64_t>(elemSize) * NUM_2) / vlfp32 * vlfp32;
        if (nextRFactor < static_cast<uint64_t>(vlfp32) || nextRFactor >= rFactor) {
            break; // 无法继续收缩
        }
        rFactor = nextRFactor;
        numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);
    }
    OP_CHECK_IF(CalcSubRUbBytes(rFactor, numChunks, elemSize) > usable,
                OP_LOGE(context_->GetNodeName(),
                        "sub-R tiling cannot fit UB: r=%ld needs %luB > %luB usable (rFactor=%lu numChunks=%lu).", r,
                        CalcSubRUbBytes(rFactor, numChunks, elemSize), usable, rFactor, numChunks),
                return false);

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
