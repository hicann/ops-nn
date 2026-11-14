/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_MAX_V2_REGBASE_H
#define QUANT_MAX_V2_REGBASE_H

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

#include "kernel_operator.h"
#include "quant_max_v2_tiling_data.h"
#include "quant_max_v2_common.h"
#include "quant_max_v2_vf.h"
#include "../inc/platform.h"

namespace QuantMaxV2 {
using namespace AscendC;

// QuantMode、HasGroup 在编译期选择特化，避免运行时分支和无用资源分配。
template <typename T, typename U, uint64_t RoundMode, uint64_t QuantMode, uint64_t HasGroup>
class QuantMaxV2Kernel;

// 动态分组：仅实例化该场景需要的缓冲和排程。
template <typename T, typename U, uint64_t RoundMode>
class QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>
    : public QuantMaxV2Base<T, U, RoundMode> {
public:
    __aicore__ inline explicit QuantMaxV2Kernel(const QuantMaxV2TilingData* tilingData) : tilingData_(tilingData){};
    // 统一七参签名（与入口共用调用点对齐）。本特化 scale 为 nullptr 占位、不消费（见实现内 (void)scale）。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                                GM_ADDR workspace);
    // 动态完整流水：首遍求max、跨核归并、第二遍量化及结果输出。
    __aicore__ inline void Process()
    {
        ProcessTiles<false>();
        SyncAll();
        MergeScale();
        ProcessTiles<true>();
        MergeAmax();
        EmitYScale();
    }

private:
    template <bool Reverse>
    __aicore__ inline void ProcessTiles();
    __aicore__ inline void MergeScale();
    __aicore__ inline void MergeAmax();
    __aicore__ inline void EmitYScale();

private:
    template <bool WriteY, bool ComputeMax>
    __aicore__ inline void Compute(int64_t dataCount, int64_t groupIndex);
    __aicore__ inline void ComputeDynamicBf16Max(int64_t dataCount);
    __aicore__ inline void FinalizeDynamicBf16GroupMax(int64_t groupIndex);
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    // 采用 4 个输入、3 个输出缓冲；BF16 使用位级求最大值，通用类型第二遍采用三块倒序预取。缓冲深度由已有消融结果选定。
    constexpr static int32_t BUFFER_NUM = 4;
    constexpr static int32_t OUTPUT_BUFFER_NUM = 3;
    constexpr static int32_t NUM_ONE_QUEUE = 1;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int64_t ALIGN_BYTES = 32;
    constexpr static int64_t VF_BYTES = platform::GetVRegSize();
    constexpr static int64_t FLOAT_ALIGN_ELEM = ALIGN_BYTES / sizeof(float); // 8

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, OUTPUT_BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> groupListBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TBuf<TPosition::VECCALC> yScaleBuf_;
    TBuf<TPosition::VECCALC> bf16TileMaxBuf_;
    TQue<QuePosition::VECIN, NUM_ONE_QUEUE> inQueueGroupMax_;
    TQue<QuePosition::VECOUT, NUM_ONE_QUEUE> outQueueAmax_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int64_t> groupListGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> yScaleGm_;
    GlobalTensor<float> coreGroupMaxGm_;
    LocalTensor<float> scaleLocal_;
    LocalTensor<int64_t> groupListLocal_;
    LocalTensor<float> maxLocal_;

    const QuantMaxV2TilingData* tilingData_;
    int32_t coreIdx_ = 0;
    int64_t blockLen_ = 1;
    float scaleValue_ = 0.0f;
    int64_t numGroups_ = 0;
    int64_t groupStride_ = 0;
    int64_t dim1_ = 0;
    uint16_t vectorLength_ = 0;
    // 整块搬运复用预构建的描述符参数。
    DataCopyExtParams fullInParams_;
    DataCopyExtParams fullOutParams_;
    int64_t totalCoreNum_;
    int64_t ubFactor_;
    int64_t actualCoreNum_;
    int64_t blockTailCoreNum_;
    int64_t blockFactor_;
    int64_t blockTailFactor_;
    int64_t firstGroupForCore_ = 0;
    int64_t lastGroupForCore_ = 0;
    int64_t tileMaxCount_ = 0;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale, GM_ADDR workspace)
{
    (void)scale;
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    coreIdx_ = GetBlockIdx();
    numGroups_ = tilingData_->numGroups;
    groupStride_ = (numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM;
    dim1_ = tilingData_->dim1;
    totalCoreNum_ = tilingData_->totalCoreNum;
    ubFactor_ = tilingData_->ubFactor;
    fullInParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(T)), 0, 0, 0};
    fullOutParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(U)), 0, 0, 0};
    actualCoreNum_ = tilingData_->actualCoreNum;
    blockTailCoreNum_ = tilingData_->blockTailCoreNum;
    blockFactor_ = tilingData_->blockFactor;
    blockTailFactor_ = tilingData_->blockTailFactor;
    vectorLength_ = VF_BYTES / sizeof(float);

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    // 分组模板始终提供 groupList。
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    // amax 为必选输出。
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));
    yScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale));
    coreGroupMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE),
                                    groupStride_ * totalCoreNum_);

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, OUTPUT_BUFFER_NUM, ubFactor_ * sizeof(U));
    pipe_.InitBuffer(scaleBuf_, numGroups_ * sizeof(float));
    pipe_.InitBuffer(groupListBuf_, numGroups_ * sizeof(int64_t));
    pipe_.InitBuffer(maxBuf_,
                     ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(float));
    pipe_.InitBuffer(yScaleBuf_, groupStride_ * sizeof(float));
    // 动态 bf16 专用：位级 tile 最大值的 16 位暂存区。容量取本核 tile 数上界——
    // ceil(blockFactor/ubFactor) 基础 tile 数 + 相交组段数（每组首/尾可能各带一个短 tile），
    // 再对齐到 16 位向量宽度；Host 侧 CalcUbFactor 用同一公式做不动点求解，保证 UB 装得下。
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        // Host 按展平元素分核。
        const int64_t maxCoreElements = tilingData_->blockFactor;
        const int64_t maxGroupSegments = (numGroups_ < tilingData_->blockFactor) ? numGroups_ :
                                                                                   tilingData_->blockFactor;
        const int64_t maxTileCount = QuantMaxExactDivide(maxCoreElements + ubFactor_ - 1, ubFactor_,
                                                         tilingData_->tileDivReciprocal) +
                                     maxGroupSegments;
        int64_t tileCapacity = ((maxTileCount + VF_BYTES / sizeof(uint16_t) - 1) / (VF_BYTES / sizeof(uint16_t))) *
                               (VF_BYTES / sizeof(uint16_t));
        if (tileCapacity == 0) {
            tileCapacity = VF_BYTES / sizeof(uint16_t);
        }
        pipe_.InitBuffer(bf16TileMaxBuf_, tileCapacity * sizeof(uint16_t));
    }
    pipe_.InitBuffer(inQueueGroupMax_, NUM_ONE_QUEUE, groupStride_ * totalCoreNum_ * sizeof(float) + VF_BYTES);
    pipe_.InitBuffer(outQueueAmax_, NUM_ONE_QUEUE,
                     ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(T));

    scaleLocal_ = scaleBuf_.Get<float>();
    groupListLocal_ = groupListBuf_.Get<int64_t>();
    maxLocal_ = maxBuf_.Get<float>();

    DataCopyExtParams groupListCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(groupListLocal_, groupListGm_[0], groupListCopyParams, {false, 0, 0, 0});

    Duplicate<float>(maxLocal_, 0.0f, groupStride_);

    // Process 通过标量 GetValue 读取 groupList，必须保证 MTE2 到标量流水的同步，仅 MTE2_V 不足。
    int32_t eventIDMTE2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE2ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
}

// 两趟 tile 主循环（模板参 Reverse 选趟次）：首遍求本核各相交组的局部绝对值最大值（bf16
// 走位级暂存链）写入核间矩阵；第二遍反向遍历组段、按组取量化系数完成 y 的量化写出；深度预取（3 tile 在飞）+
// 输出延迟写出。
template <typename T, typename U, uint64_t RoundMode>
template <bool Reverse>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::ProcessTiles()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }

    groupListLocal_ = groupListBuf_.Get<int64_t>();
    scaleLocal_ = scaleBuf_.Get<float>();

    int64_t coreStart = 0;
    if (coreIdx_ < actualCoreNum_ - blockTailCoreNum_) {
        blockLen_ = blockFactor_;
        coreStart = coreIdx_ * blockLen_;
    } else {
        blockLen_ = blockTailFactor_;
        coreStart = (actualCoreNum_ - blockTailCoreNum_) * blockFactor_ +
                    (coreIdx_ - (actualCoreNum_ - blockTailCoreNum_)) * blockLen_;
    }

    int64_t coreEnd = coreStart + blockLen_;
    // 首遍缓存本核覆盖的组范围，第二遍直接复用，减少标量 UB 查找。
    int64_t firstGroup = Reverse ? firstGroupForCore_ : 0;
    int64_t lastGroup = Reverse ? lastGroupForCore_ : 0;
    if (!Reverse && numGroups_ > 1) {
        int64_t low = 0;
        int64_t high = numGroups_;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            if (groupListLocal_.GetValue(mid) * dim1_ <= coreStart) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        firstGroup = (low == numGroups_) ? 0 : low;

        low = firstGroup;
        high = numGroups_;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            if (groupListLocal_.GetValue(mid) * dim1_ < coreEnd) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        lastGroup = (low == numGroups_) ? -1 : low;
    }
    if constexpr (!Reverse) {
        firstGroupForCore_ = firstGroup;
        lastGroupForCore_ = lastGroup;
    }
    // 累计边界可逐组传递：正序传递结束位置，倒序传递起始位置，每组只需一次 GetValue。
    int64_t groupStart = (firstGroup == 0) ? 0 : groupListLocal_.GetValue(firstGroup - 1) * dim1_;
    int64_t groupEndReverse = (lastGroup >= 0) ? groupListLocal_.GetValue(lastGroup) * dim1_ : 0;
    for (int64_t groupStep = firstGroup; groupStep <= lastGroup; ++groupStep) {
        int64_t groupIdx = Reverse ? firstGroup + lastGroup - groupStep : groupStep;
        int64_t groupEnd = 0;
        if constexpr (Reverse) {
            groupEnd = groupEndReverse; // 本组终点等于上一个较高编号组的起点。
        } else {
            groupEnd = groupListLocal_.GetValue(groupIdx) * dim1_;
        }
        int64_t groupStartCur = 0;
        if constexpr (Reverse) {
            groupStartCur = (groupIdx == 0) ? 0 : groupListLocal_.GetValue(groupIdx - 1) * dim1_;
            groupEndReverse = groupStartCur; // 下一较低编号组的终点等于本组起点。
        } else {
            groupStartCur = groupStart;
            groupStart = groupEnd;
        }
        int64_t segStart = (coreStart > groupStartCur) ? coreStart : groupStartCur;
        int64_t segEnd = (coreEnd < groupEnd) ? coreEnd : groupEnd;
        int64_t segLen = segEnd - segStart;

        if (segLen <= 0) {
            continue;
        }

        // bf16 首遍每组开始时复位暂存游标：每 tile 追加一个 16 位最大值，组收尾一次性归约。
        if constexpr (!Reverse && IsSameType<T, bfloat16_t>::value) {
            tileMaxCount_ = 0;
        }

        if constexpr (Reverse) {
            // 溢出防护（见 common.h DYNAMIC_AMAX_FLOOR）：amax 极小时把分母钳到安全下限，
            // q = dstTypeMax/amax 恒有限；过大量化值由 SAT Cast 饱和到目标类型最大值。
            float amax = maxLocal_.GetValue(groupIdx);
            if (amax < DYNAMIC_AMAX_FLOOR) {
                amax = DYNAMIC_AMAX_FLOOR;
            }
            scaleValue_ = amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f;
        }
        int64_t ubFactor = ubFactor_;
        int64_t loopNum = QuantMaxExactDivide(segLen, ubFactor, tilingData_->tileDivReciprocal);
        int64_t loopTail = segLen - loopNum * ubFactor;

        if constexpr (Reverse) {
            // 预取深度必须小于输入缓冲数，否则 AllocTensor 可能等待尚未执行的 Compute 而死锁；BF16
            // 保持三块预取、两块延迟写回。
            constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
            constexpr int64_t outputDelay = OUTPUT_BUFFER_NUM - 1;
            const int64_t tileCount = loopNum + (loopTail != 0);
            if (tileCount != 0) {
                const int64_t lastTile = tileCount - 1;
                for (int64_t revSeed = 0; revSeed < tileCount && revSeed < prefetchDepth; ++revSeed) {
                    const int64_t seed = lastTile - revSeed;
                    const int64_t seedLen = (seed == loopNum && loopTail != 0) ? loopTail : ubFactor;
                    CopyInX(seedLen, segStart + seed * ubFactor);
                }
                for (int64_t revStep = 0; revStep < tileCount; ++revStep) {
                    const int64_t i = lastTile - revStep;
                    if (i >= prefetchDepth) {
                        CopyInX(ubFactor, segStart + (i - prefetchDepth) * ubFactor);
                    }
                    const int64_t tileLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
                    Compute<true, false>(tileLen, groupIdx);
                    if (revStep >= outputDelay) {
                        const int64_t previous = i + outputDelay;
                        const int64_t previousLen = (previous == loopNum && loopTail != 0) ? loopTail : ubFactor;
                        CopyOutY(previousLen, segStart + previous * ubFactor);
                    }
                }
                const int64_t remaining = (tileCount < outputDelay) ? tileCount : outputDelay;
                for (int64_t i = remaining - 1; i >= 0; --i) {
                    const int64_t tileLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
                    CopyOutY(tileLen, segStart + i * ubFactor);
                }
            }
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            // bf16 首遍不走通用 Compute<false,true>：改走位级 tile max + 16 位暂存链
            // （见 ComputeDynamicBf16Max），消除逐元素的 bf16→fp32 加宽与 float 归并开销；
            // tile 调度结构与通用正序一致（三块预取）。
            constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
            const int64_t tileCount = loopNum + (loopTail != 0);
            for (int64_t i = 0; i < tileCount && i < prefetchDepth; ++i) {
                const int64_t tileLen = (i == loopNum) ? loopTail : ubFactor;
                CopyInX(tileLen, segStart + i * ubFactor);
            }
            for (int64_t i = 0; i < tileCount; ++i) {
                if (i + prefetchDepth < tileCount) {
                    const int64_t next = i + prefetchDepth;
                    const int64_t nextLen = (next == loopNum) ? loopTail : ubFactor;
                    CopyInX(nextLen, segStart + next * ubFactor);
                }
                ComputeDynamicBf16Max((i == loopNum) ? loopTail : ubFactor);
            }
        } else {
            // 正序同样采用三块预取：先填充，再逐轮补充，充分利用四个输入缓冲。
            constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
            const int64_t tileCount = loopNum + (loopTail != 0);
            for (int64_t i = 0; i < tileCount && i < prefetchDepth; ++i) {
                const int64_t tileLen = (i == loopNum) ? loopTail : ubFactor;
                CopyInX(tileLen, segStart + i * ubFactor);
            }
            for (int64_t i = 0; i < tileCount; ++i) {
                if (i + prefetchDepth < tileCount) {
                    const int64_t next = i + prefetchDepth;
                    const int64_t nextLen = (next == loopNum) ? loopTail : ubFactor;
                    CopyInX(nextLen, segStart + next * ubFactor);
                }
                Compute<false, true>((i == loopNum) ? loopTail : ubFactor, groupIdx);
            }
        }

        if constexpr (!Reverse && IsSameType<T, bfloat16_t>::value) {
            FinalizeDynamicBf16GroupMax(groupIdx);
        }
    }

    if constexpr (Reverse) {
        return;
    }
    LocalTensor<float> tileMaxLocal = maxBuf_.Get<float>();
    int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_ * groupStride_], tileMaxLocal, copyParams);

    // SyncAll 前等局部最大值写回完成；归并随后会覆盖 maxBuf_，而 SyncAll 本身不保证本核 MTE3
    // 已读完，需避免写后读依赖冲突。
    int32_t eventIDMTE3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    int32_t eventIDMTE3ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
}

// tile 级计算：取一块 x，按 WriteY/ComputeMax 组合调用 GroupedQuantComputeVf（见
// vf.h），完成量化写出与/或组内最大值续接。
template <typename T, typename U, uint64_t RoundMode>
template <bool WriteY, bool ComputeMax>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::Compute(
    int64_t dataCount, int64_t groupIndex)
{
    // bf16 首遍（仅求最大值）整体改道位级实现，不经本函数的通用 fp32 归并链。
    if constexpr (IsSameType<T, bfloat16_t>::value && !WriteY && ComputeMax) {
        ComputeDynamicBf16Max(dataCount);
        return;
    }

    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<U> outLocal;
    __ubuf__ U* outAddr = nullptr;
    if constexpr (WriteY) {
        outLocal = outQueueY_.template AllocTensor<U>();
        outAddr = (__ubuf__ U*)outLocal.GetPhyAddr();
    }

    // 运行时门控：双链只在与实测收益相符的小 shape（每核 tile 数 ≤ 64，L2 驻留）启用。
    // HBM 级 shape（每核 128 tile）配对加载会打断 DMA 流水，dur 回退 3.8us，保持单链。
    // 两个变体是彼此独立的模板实例化，单链代码不承担配对环的寄存器压力。
    const uint32_t dualChain = (WriteY && !ComputeMax && IsSameType<T, bfloat16_t>::value &&
                                blockFactor_ / ubFactor_ <= DUAL_CHAIN_MAX_TILES_PER_CORE) ?
                                   1u :
                                   0u;
    if (dualChain != 0) {
        asc_vf_call<GroupedQuantComputeVf<T, U, RoundMode, WriteY, ComputeMax, true>>(
            (__ubuf__ T*)xLocal.GetPhyAddr(), outAddr, (__ubuf__ float*)maxLocal_.GetPhyAddr(), scaleValue_,
            static_cast<uint32_t>(dataCount), groupIndex, vectorLength_);
    } else {
        asc_vf_call<GroupedQuantComputeVf<T, U, RoundMode, WriteY, ComputeMax, false>>(
            (__ubuf__ T*)xLocal.GetPhyAddr(), outAddr, (__ubuf__ float*)maxLocal_.GetPhyAddr(), scaleValue_,
            static_cast<uint32_t>(dataCount), groupIndex, vectorLength_);
    }

    inQueueX_.FreeTensor(xLocal);
    if constexpr (WriteY) {
        outQueueY_.EnQue(outLocal);
    }
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC,
                                        // bf16 首遍收尾：本核该组全部 tile 的 16 位最大值归约为组最大值写入核间矩阵。
                                        TPL_GROUP_MODE_GROUP>::FinalizeDynamicBf16GroupMax(int64_t groupIndex)
{
    uint32_t count = static_cast<uint32_t>(tileMaxCount_);
    const uint32_t vectorLengthB16 = VF_BYTES / sizeof(uint16_t);

    // 不同 VF 在向量流水上的 UB 存取有序，收尾 VF 可直接读取逐 tile 结果，无需增加标量 V_S 等待。
    asc_vf_call<GroupedQuantBf16GroupMaxVf<T, U, RoundMode>>(
        (__ubuf__ uint16_t*)bf16TileMaxBuf_.Get<uint16_t>().GetPhyAddr(),
        (__ubuf__ float*)maxLocal_.GetPhyAddr() + groupIndex, count, vectorLengthB16);
    tileMaxCount_ = 0;
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC,
                                        // 动态 bf16 首遍专用：位级求本 tile 绝对值最大值（16 位暂存，四路累加）。
                                        TPL_GROUP_MODE_GROUP>::ComputeDynamicBf16Max(int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    const uint32_t vectorLengthB16 = VF_BYTES / sizeof(uint16_t);

    asc_vf_call<GroupedQuantBf16TileMaxVf>(
        (__ubuf__ uint16_t*)xLocal.GetPhyAddr(), (__ubuf__ uint16_t*)bf16TileMaxBuf_.Get<uint16_t>().GetPhyAddr(),
        static_cast<uint32_t>(dataCount), static_cast<uint32_t>(tileMaxCount_), vectorLengthB16);

    ++tileMaxCount_;
    inQueueX_.FreeTensor(xLocal);
}

// 输入搬运：从 GM 搬 xLen 个元素（偏移 xInOffset）进输入队列（DataCopyPad 带填充）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::CopyInX(
    int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 组段起点可能不对齐，统一使用带填充搬运；32 字节对齐的快速搬运仅用于不分组模板。
    if (xLen == ubFactor_) {
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], fullInParams_, padParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    }
    inQueueX_.EnQue(xLocal);
}

// 输出搬运：把量化结果 yLen 个元素从输出队列写回 GM y（偏移 yOutOffset）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::CopyOutY(
    int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.template DeQue<U>();
    if (yLen == ubFactor_) {
        DataCopyPad(yGm_[yOutOffset], outLocal, fullOutParams_);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
        DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    }
    outQueueY_.FreeTensor(outLocal);
}

// 跨核归并组最大值：把各核写入核间矩阵的逐组局部最大值归并成全核一致的 merged 向量（core0/core1
// 全量、其余核只归并本核覆盖段），归并前预发第二遍反向首块搬运。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::MergeScale()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }
    LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
    if (numGroups_ == 1) {
        // 单组矩阵行连续；统一用 DataCopyPad 按字节长度搬运，避免依赖 32B 对齐前提。
        DataCopyExtParams singleCopyParams{1, static_cast<uint32_t>(actualCoreNum_ * groupStride_ * sizeof(float)), 0,
                                           0, 0};
        DataCopyPad<float>(groupMax, coreGroupMaxGm_, singleCopyParams, {false, 0, 0, 0});
    } else {
        DataCopyExtParams copyParams{static_cast<uint16_t>(actualCoreNum_),
                                     static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(groupMax, coreGroupMaxGm_, copyParams, {false, 0, 0, 0});
    }
    inQueueGroupMax_.EnQue(groupMax);
    groupMax = inQueueGroupMax_.DeQue<float>();
    int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    LocalTensor<float> quantScale = scaleBuf_.Get<float>();
    LocalTensor<float> merged = maxBuf_.Get<float>();
    if (numGroups_ == 1) {
        // 每核单组最大值的填充位置为零，直接归约连续矩阵，替代标量跨步读取。
        asc_vf_call<GroupedQuantMergeSingleGroupVf>(
            (__ubuf__ float*)groupMax.GetPhyAddr(), (__ubuf__ float*)merged.GetPhyAddr(),
            static_cast<uint32_t>(actualCoreNum_ * groupStride_), vectorLength_);
        int32_t vectorToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        const float amax = merged.GetValue(0);
        quantScale.SetValue(0, amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f);
    } else if (numGroups_ >= FLOAT_ALIGN_ELEM) {
        // 普通计算核只归并本核覆盖的组；核 0 输出 amax、核 1 输出 y_scale，仍归并全部组。MTE2 保持搬入完整矩阵，仅减少
        // VF 的跨步加载。
        const bool needFullRange = (coreIdx_ == 0) || (coreIdx_ == 1 && actualCoreNum_ > 1);
        const int64_t vfStart = needFullRange ? 0 : firstGroupForCore_;
        const int64_t vfLen = needFullRange ? numGroups_ : (lastGroupForCore_ - firstGroupForCore_ + 1);
        // 保持原矩阵行跨度，从 vfStart 列开始处理 vfLen 组。vfStart 向下对齐到向量长度，避免整向量加载越过行尾。
        const int64_t alignedStart = (vfStart / vectorLength_) * vectorLength_;
        const int64_t adjustedLen = vfStart + vfLen - alignedStart;
        asc_vf_call<GroupedQuantMergeScaleVf>((__ubuf__ float*)groupMax.GetPhyAddr() + alignedStart,
                                              (__ubuf__ float*)merged.GetPhyAddr() + alignedStart, adjustedLen,
                                              static_cast<uint16_t>(actualCoreNum_), groupStride_, vectorLength_);
        int32_t vectorToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        for (int64_t groupIdx = firstGroupForCore_; groupIdx <= lastGroupForCore_; ++groupIdx) {
            const float amax = merged.GetValue(groupIdx);
            quantScale.SetValue(groupIdx, amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f);
        }
    } else {
        for (int64_t groupIdx = 0; groupIdx < numGroups_; ++groupIdx) {
            float amax = 0.0f;
            for (int64_t core = 0; core < actualCoreNum_; ++core) {
                const float part = groupMax.GetValue(core * groupStride_ + groupIdx);
                amax = part > amax ? part : amax;
            }
            merged.SetValue(groupIdx, amax);
            quantScale.SetValue(groupIdx, amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f);
        }
    }
    inQueueGroupMax_.FreeTensor(groupMax);
}

// amax 落盘：从 merged 取各组最大值，经 CastAmax 转为输入 dtype 写出 amax[G]。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::MergeAmax()
{
    if (coreIdx_ >= 1) {
        return;
    }
    // MergeScale 已在 maxLocal_ 留下完整的逐组最大值。
    LocalTensor<float> merged = maxBuf_.Get<float>();

    int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);

    LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();
    asc_vf_call<GroupedQuantCastAmaxVf<T, U, RoundMode>>(
        (__ubuf__ float*)merged.GetPhyAddr(), (__ubuf__ T*)amaxLocal.GetPhyAddr(), numGroups_, vectorLength_);
    outQueueAmax_.EnQue(amaxLocal);
    amaxLocal = outQueueAmax_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(amaxGm_, amaxLocal, copyParams);
    outQueueAmax_.FreeTensor(amaxLocal);
}

// y_scale 输出（core1）：yScale[g] = max(amax[g], 0) / dstTypeMax，一条 VF（乘倒数）完成。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_GROUP>::EmitYScale()
{
    // 核 1 写 y_scale，与核 0 转换并写 amax 并行；所有活跃核已在 MergeScale 得到完整逐组最大值。
    if (coreIdx_ != (actualCoreNum_ > 1 ? 1 : 0)) {
        return;
    }
    // 2 至 7 组时由标量写归并结果，此 VF 读取前需要独立的 S_V 同步；核 0 的 MergeAmax 同步不能覆盖核 1。
    int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
    LocalTensor<float> merged = maxBuf_.Get<float>();
    LocalTensor<float> yScale = yScaleBuf_.Get<float>();
    // 用 VF 替代逐组标量读写；amax 非负，dstTypeMax <= 0 时 y_scale 为零。
    if (tilingData_->dstTypeMax > 0.0f) {
        asc_vf_call<GroupedQuantEmitYScaleVf<T, U, RoundMode>>(
            (__ubuf__ float*)merged.GetPhyAddr(), (__ubuf__ float*)yScale.GetPhyAddr(),
            static_cast<uint32_t>(numGroups_), 1.0f / tilingData_->dstTypeMax, vectorLength_);
    } else {
        Duplicate<float>(yScale, 0.0f, static_cast<int32_t>(numGroups_));
    }
    int32_t vectorToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(yScaleGm_, yScale, copyParams);
}

// 动态不分组：仅实例化该场景需要的缓冲和排程。
template <typename T, typename U, uint64_t RoundMode>
class QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>
    : public QuantMaxV2Base<T, U, RoundMode> {
public:
    __aicore__ inline explicit QuantMaxV2Kernel(const QuantMaxV2TilingData* tilingData) : tilingData_(tilingData){};
    // 统一七参签名（与入口共用调用点对齐）。本特化 scale/groupList 为 nullptr 占位、不消费（单段直走，无组簿记）。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                                GM_ADDR workspace);
    // 动态完整流水：首遍求max、跨核归并、第二遍量化及结果输出。
    __aicore__ inline void Process()
    {
        ProcessTiles<false>();
        SyncAll();
        MergeScale();
        ProcessTiles<true>();
        MergeAmax();
        EmitYScale();
    }

private:
    template <bool Reverse>
    __aicore__ inline void ProcessTiles();
    __aicore__ inline void MergeScale();
    __aicore__ inline void MergeAmax();
    __aicore__ inline void EmitYScale();

private:
    template <bool WriteY, bool ComputeMax>
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void ComputeDynamicBf16Max(int64_t dataCount);
    __aicore__ inline void FinalizeDynamicBf16GroupMax();
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    // 与动态分组模板一致，使用 4 个输入、3 个输出缓冲。
    constexpr static int32_t BUFFER_NUM = 4;
    constexpr static int32_t OUTPUT_BUFFER_NUM = 3;
    constexpr static int32_t NUM_ONE_QUEUE = 1;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int64_t ALIGN_BYTES = 32;
    constexpr static int64_t VF_BYTES = platform::GetVRegSize();
    constexpr static int64_t FLOAT_ALIGN_ELEM = ALIGN_BYTES / sizeof(float); // 8

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, OUTPUT_BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TBuf<TPosition::VECCALC> yScaleBuf_;
    TBuf<TPosition::VECCALC> bf16TileMaxBuf_;
    TQue<QuePosition::VECIN, NUM_ONE_QUEUE> inQueueGroupMax_;
    TQue<QuePosition::VECOUT, NUM_ONE_QUEUE> outQueueAmax_;

    GlobalTensor<T> xGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> yScaleGm_;
    GlobalTensor<float> coreGroupMaxGm_;
    LocalTensor<float> scaleLocal_;
    LocalTensor<float> maxLocal_;

    const QuantMaxV2TilingData* tilingData_;
    int32_t coreIdx_ = 0;
    float scaleValue_ = 0.0f;
    int64_t groupStride_ = 0; // 固定单组，按 8 个 float 对齐。
    uint16_t vectorLength_ = 0;
    // 整块搬运复用预构建的描述符参数。
    DataCopyExtParams fullInParams_;
    DataCopyExtParams fullOutParams_;
    int64_t totalCoreNum_;
    int64_t ubFactor_;
    int64_t actualCoreNum_;
    int64_t blockTailCoreNum_;
    int64_t blockFactor_;
    int64_t blockTailFactor_;
    int64_t coreStart_ = 0;
    int64_t loopNum_ = 0;
    int64_t loopTail_ = 0;
    int64_t tileCount_ = 0;
    int64_t tileMaxCount_ = 0;
    // 记录 MergeScale 中已预取的第二遍倒序块数。
    int64_t issuedPrefetchCount_ = 0;
    // 保留首遍最后一块原始X，第二遍倒序直接消费，避免一次GM重读。
    LocalTensor<T> retainedTail_[RETAINED_TAIL_NUM];
    int64_t retainedTailCount_ = 0;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale, GM_ADDR workspace)
{
    (void)scale;
    (void)groupList;
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    coreIdx_ = GetBlockIdx();
    groupStride_ = FLOAT_ALIGN_ELEM; // 单组结果填充到 8 个 float。
    totalCoreNum_ = tilingData_->totalCoreNum;
    ubFactor_ = tilingData_->ubFactor;
    fullInParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(T)), 0, 0, 0};
    fullOutParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(U)), 0, 0, 0};
    actualCoreNum_ = tilingData_->actualCoreNum;
    blockTailCoreNum_ = tilingData_->blockTailCoreNum;
    blockFactor_ = tilingData_->blockFactor;
    blockTailFactor_ = tilingData_->blockTailFactor;
    vectorLength_ = VF_BYTES / sizeof(float);
    if (coreIdx_ < actualCoreNum_) {
        const int64_t normalCoreNum = actualCoreNum_ - blockTailCoreNum_;
        if (coreIdx_ < normalCoreNum) {
            coreStart_ = coreIdx_ * blockFactor_;
            loopNum_ = tilingData_->tensorBlockLoopNum;
            loopTail_ = tilingData_->tensorBlockLoopTail;
            tileCount_ = tilingData_->tensorBlockTileCount;
        } else {
            coreStart_ = tilingData_->tensorTailCoreStart + (coreIdx_ - normalCoreNum) * blockTailFactor_;
            loopNum_ = tilingData_->tensorTailLoopNum;
            loopTail_ = tilingData_->tensorTailLoopTail;
            tileCount_ = tilingData_->tensorTailTileCount;
        }
    }

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    // amax 为必选输出。
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));
    yScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale));
    coreGroupMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE),
                                    groupStride_ * totalCoreNum_);

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, OUTPUT_BUFFER_NUM, ubFactor_ * sizeof(U));
    pipe_.InitBuffer(scaleBuf_, sizeof(float));
    pipe_.InitBuffer(maxBuf_, groupStride_ * sizeof(float));
    pipe_.InitBuffer(yScaleBuf_, groupStride_ * sizeof(float));
    // 动态 bf16 专用：位级 tile 最大值 16 位暂存区。单段无组簿记，容量直接取 Host
    // 不动点解出的 tensorBf16TileMaxBufferSize（见 tiling CalcUbFactor），不再现场推导。
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        pipe_.InitBuffer(bf16TileMaxBuf_, tilingData_->tensorBf16TileMaxBufferSize);
    }
    pipe_.InitBuffer(inQueueGroupMax_, NUM_ONE_QUEUE, groupStride_ * totalCoreNum_ * sizeof(float) + VF_BYTES);
    pipe_.InitBuffer(outQueueAmax_, NUM_ONE_QUEUE, groupStride_ * sizeof(T));

    scaleLocal_ = scaleBuf_.Get<float>();
    maxLocal_ = maxBuf_.Get<float>();

    Duplicate<float>(maxLocal_, 0.0f, groupStride_);
}

// 两趟 tile 主循环（单段连续，无组簿记）：首遍求全局最大值；第二遍反向量化写出，续接 MergeScale
// 预发的反向首块种子（issuedPrefetchCount_）。
template <typename T, typename U, uint64_t RoundMode>
template <bool Reverse>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::ProcessTiles()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }

    scaleLocal_ = scaleBuf_.Get<float>();

    // 首遍、归并期间预取和第二遍共用 Host 排程。
    const int64_t segStart = coreStart_;
    const int64_t ubFactor = ubFactor_;
    const int64_t loopNum = loopNum_;
    const int64_t loopTail = loopTail_;
    const int64_t tileCount = tileCount_;

    // bf16 首遍开始时复位暂存游标（tensor 路仅一组，全程一次收尾）。
    if constexpr (!Reverse && IsSameType<T, bfloat16_t>::value) {
        tileMaxCount_ = 0;
    }

    if constexpr (Reverse) {
        // 整个张量仅有一个量化系数，只读一次。溢出防护同分组路（见 common.h）。
        float amax = maxLocal_.GetValue(0);
        if (amax < DYNAMIC_AMAX_FLOOR) {
            amax = DYNAMIC_AMAX_FLOOR;
        }
        scaleValue_ = amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f;
        // 输入缓冲数必须大于预取深度，避免额外 AllocTensor 阻塞导致死锁。
        constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
        constexpr int64_t outputDelay = OUTPUT_BUFFER_NUM - 1;
        if (tileCount != 0) {
            const int64_t lastTile = tileCount - 1;
            // MergeScale 已预取 issuedPrefetchCount_ 块，第二遍从此处补充；若未预取，则从零开始。
            for (int64_t revSeed = issuedPrefetchCount_; revSeed < tileCount && revSeed < prefetchDepth; ++revSeed) {
                const int64_t seed = lastTile - revSeed;
                const int64_t seedLen = (seed == loopNum && loopTail != 0) ? loopTail : ubFactor;
                CopyInX(seedLen, segStart + seed * ubFactor);
            }
            for (int64_t revStep = 0; revStep < tileCount; ++revStep) {
                const int64_t i = lastTile - revStep;
                if (i >= prefetchDepth) {
                    CopyInX(ubFactor, segStart + (i - prefetchDepth) * ubFactor);
                }
                const int64_t tileLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
                Compute<true, false>(tileLen);
                if (revStep >= outputDelay) {
                    const int64_t previous = i + outputDelay;
                    const int64_t previousLen = (previous == loopNum && loopTail != 0) ? loopTail : ubFactor;
                    CopyOutY(previousLen, segStart + previous * ubFactor);
                }
            }
            const int64_t remaining = (tileCount < outputDelay) ? tileCount : outputDelay;
            for (int64_t i = remaining - 1; i >= 0; --i) {
                const int64_t tileLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
                CopyOutY(tileLen, segStart + i * ubFactor);
            }
        }
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        // bf16 首遍不走通用 Compute<false,true>：改走位级 tile max + 16 位暂存链
        // （见 ComputeDynamicBf16Max）；tile 调度结构与通用正序一致（三块预取）。
        constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
        for (int64_t i = 0; i < tileCount && i < prefetchDepth; ++i) {
            const int64_t tileLen = (i == loopNum) ? loopTail : ubFactor;
            CopyInX(tileLen, segStart + i * ubFactor);
        }
        for (int64_t i = 0; i < tileCount; ++i) {
            if (i + prefetchDepth < tileCount) {
                const int64_t next = i + prefetchDepth;
                const int64_t nextLen = (next == loopNum) ? loopTail : ubFactor;
                CopyInX(nextLen, segStart + next * ubFactor);
            }
            ComputeDynamicBf16Max((i == loopNum) ? loopTail : ubFactor);
        }
    } else {
        // 正序使用三块预取，先填充再补充，充分利用四个输入缓冲。
        constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
        for (int64_t i = 0; i < tileCount && i < prefetchDepth; ++i) {
            const int64_t tileLen = (i == loopNum) ? loopTail : ubFactor;
            CopyInX(tileLen, segStart + i * ubFactor);
        }
        for (int64_t i = 0; i < tileCount; ++i) {
            if (i + prefetchDepth < tileCount) {
                const int64_t next = i + prefetchDepth;
                const int64_t nextLen = (next == loopNum) ? loopTail : ubFactor;
                CopyInX(nextLen, segStart + next * ubFactor);
            }
            Compute<false, true>((i == loopNum) ? loopTail : ubFactor);
        }
    }

    if constexpr (!Reverse && IsSameType<T, bfloat16_t>::value) {
        FinalizeDynamicBf16GroupMax();
    }

    if constexpr (Reverse) {
        return;
    }
    LocalTensor<float> tileMaxLocal = maxBuf_.Get<float>();
    int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_ * groupStride_], tileMaxLocal, copyParams);

    // SyncAll 前等写回完成，防止归并提前覆盖仍由 MTE3 读取的 maxBuf_。
    int32_t eventIDMTE3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    int32_t eventIDMTE3ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
}

// tile 级计算：取一块 x，按 WriteY/ComputeMax 组合调用 GroupedQuantComputeVf（见 vf.h）。
template <typename T, typename U, uint64_t RoundMode>
template <bool WriteY, bool ComputeMax>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::Compute(
    int64_t dataCount)
{
    // bf16 首遍（仅求最大值）整体改道位级实现，不经本函数的通用 fp32 归并链。
    if constexpr (IsSameType<T, bfloat16_t>::value && !WriteY && ComputeMax) {
        ComputeDynamicBf16Max(dataCount);
        return;
    }

    LocalTensor<T> xLocal;
    // bf16 二遍反向起始块优先取首遍保留的尾部输入（保留机制仅随 bf16 首遍的专用链落地，
    // fp16/fp32 首遍走标准队列即时释放，无可复用缓冲）。
    if constexpr (WriteY && IsSameType<T, bfloat16_t>::value) {
        if (retainedTailCount_ > 0) {
            xLocal = retainedTail_[--retainedTailCount_];
        } else {
            xLocal = inQueueX_.template DeQue<T>();
        }
    } else {
        xLocal = inQueueX_.template DeQue<T>();
    }
    LocalTensor<U> outLocal;
    __ubuf__ U* outAddr = nullptr;
    if constexpr (WriteY) {
        outLocal = outQueueY_.template AllocTensor<U>();
        outAddr = (__ubuf__ U*)outLocal.GetPhyAddr();
    }

    // 双链在 tensor 路预注册逐例门判负关闭（实测 -0.6~-1.7pt 不过门），恒走单链
    // （DualChain 取默认 false）；组路门控见 GROUP base 的 Compute。
    asc_vf_call<GroupedQuantComputeVf<T, U, RoundMode, WriteY, ComputeMax>>(
        (__ubuf__ T*)xLocal.GetPhyAddr(), outAddr, (__ubuf__ float*)maxLocal_.GetPhyAddr(), scaleValue_,
        static_cast<uint32_t>(dataCount), 0, vectorLength_);

    inQueueX_.FreeTensor(xLocal);
    if constexpr (WriteY) {
        outQueueY_.EnQue(outLocal);
    }
}

template <typename T, typename U, uint64_t RoundMode>
// bf16 首遍收尾：全部 tile 的 16 位最大值归约为全局最大值。
__aicore__ inline void
QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::FinalizeDynamicBf16GroupMax()
{
    uint32_t count = static_cast<uint32_t>(tileMaxCount_);
    const uint32_t vectorLengthB16 = VF_BYTES / sizeof(uint16_t);

    // 跨 VF 的 UB 存取在向量流水有序，无需额外的标量 V_S 等待。
    asc_vf_call<GroupedQuantBf16GroupMaxVf<T, U, RoundMode>>(
        (__ubuf__ uint16_t*)bf16TileMaxBuf_.Get<uint16_t>().GetPhyAddr(), (__ubuf__ float*)maxLocal_.GetPhyAddr(),
        count, vectorLengthB16);
    tileMaxCount_ = 0;
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC,
                                        // 动态 bf16 首遍专用：位级求本 tile 绝对值最大值（16 位暂存，四路累加）。
                                        TPL_GROUP_MODE_TENSOR>::ComputeDynamicBf16Max(int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    const uint32_t vectorLengthB16 = VF_BYTES / sizeof(uint16_t);

    asc_vf_call<GroupedQuantBf16TileMaxVf>(
        (__ubuf__ uint16_t*)xLocal.GetPhyAddr(), (__ubuf__ uint16_t*)bf16TileMaxBuf_.Get<uint16_t>().GetPhyAddr(),
        static_cast<uint32_t>(dataCount), static_cast<uint32_t>(tileMaxCount_), vectorLengthB16);

    ++tileMaxCount_;
    // 首遍最后 RETAINED_TAIL_NUM 块恰是二遍反向的前几块：输入缓冲保留不释放，二遍直接消费免 GM 重读。
    if (tileMaxCount_ + RETAINED_TAIL_NUM > tileCount_) {
        retainedTail_[retainedTailCount_++] = xLocal;
    } else {
        inQueueX_.FreeTensor(xLocal);
    }
}

// 输入搬运：从 GM 搬 xLen 个元素（偏移 xInOffset）进输入队列（DataCopyPad 带填充）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::CopyInX(
    int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 统一使用 DataCopyPad 按字节长度搬运：blockFactor 不保证 32B 对齐，Pad 版自行处理，
    // 不再需要运行时对齐判定分支。整 tile 复用预置参数，尾 tile 按实际长度构造。
    if (xLen == ubFactor_) {
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], fullInParams_, padParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    }
    inQueueX_.EnQue(xLocal);
}

// 输出搬运：把量化结果 yLen 个元素从输出队列写回 GM y（偏移 yOutOffset）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::CopyOutY(
    int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.template DeQue<U>();
    if (yLen == ubFactor_) {
        DataCopyPad(yGm_[yOutOffset], outLocal, fullOutParams_);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
        DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    }
    outQueueY_.FreeTensor(outLocal);
}

// 单张量跨核归并：core0 把各核局部最大值（连续存放）归并为全局 amax，并预发第二遍反向首块种子。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::MergeScale()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }
    // 不分组路径的核间矩阵连续且按 32 字节对齐。
    LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
    // 跨核矩阵按字节长度整段搬运；DataCopyPad 自行处理对齐，不依赖 32B 前提。
    DataCopyExtParams mergeCopyParams{1, static_cast<uint32_t>(actualCoreNum_ * groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(groupMax, coreGroupMaxGm_, mergeCopyParams, {false, 0, 0, 0});
    inQueueGroupMax_.EnQue(groupMax);
    int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    // 等待归并矩阵前先发射独立的第二遍 X 预取。MTE2_S
    // 标志先于这些搬运提交，仅等待矩阵完成；跨核同步已完成，保持四缓冲、三块预取。
    if (coreIdx_ < actualCoreNum_) {
        const int64_t coreStartN2 = coreStart_;
        const int64_t loopNumN2 = loopNum_;
        const int64_t loopTailN2 = loopTail_;
        const int64_t tileCountN2 = tileCount_;
        constexpr int64_t prefetchDepth = BUFFER_NUM - 1;
        // 二遍倒序起始的 retained 块已在 UB，预发种子从其后继块继续。
        issuedPrefetchCount_ = retainedTailCount_;
        for (int64_t prefetchIdx = issuedPrefetchCount_; prefetchIdx < tileCountN2 && prefetchIdx < prefetchDepth;
             ++prefetchIdx) {
            const int64_t seedIdx = tileCountN2 - 1 - prefetchIdx;
            const int64_t seedLen = (seedIdx == loopNumN2 && loopTailN2 != 0) ? loopTailN2 : ubFactor_;
            CopyInX(seedLen, coreStartN2 + seedIdx * ubFactor_);
            ++issuedPrefetchCount_;
        }
    }

    groupMax = inQueueGroupMax_.DeQue<float>();
    WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);

    LocalTensor<float> quantScale = scaleBuf_.Get<float>();
    LocalTensor<float> merged = maxBuf_.Get<float>();
    // 单组最大值的填充位置为零，用连续向量归约替代逐核标量跨步读取。
    asc_vf_call<GroupedQuantMergeSingleGroupVf>((__ubuf__ float*)groupMax.GetPhyAddr(),
                                                (__ubuf__ float*)merged.GetPhyAddr(),
                                                static_cast<uint32_t>(actualCoreNum_ * groupStride_), vectorLength_);
    int32_t vectorToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
    SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    const float amax = merged.GetValue(0);
    quantScale.SetValue(0, amax > 0.0f ? tilingData_->dstTypeMax / amax : 0.0f);
    inQueueGroupMax_.FreeTensor(groupMax);
}

// amax 落盘：全局最大值转为输入 dtype 写出 amax[1]。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::MergeAmax()
{
    if (coreIdx_ >= 1) {
        return;
    }
    // MergeScale 已在 maxLocal_ 留下单组最大值。
    LocalTensor<float> merged = maxBuf_.Get<float>();

    int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);

    LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();
    asc_vf_call<GroupedQuantCastAmaxVf<T, U, RoundMode>>((__ubuf__ float*)merged.GetPhyAddr(),
                                                         (__ubuf__ T*)amaxLocal.GetPhyAddr(), 1, vectorLength_);
    outQueueAmax_.EnQue(amaxLocal);
    amaxLocal = outQueueAmax_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(amaxGm_, amaxLocal, copyParams);
    outQueueAmax_.FreeTensor(amaxLocal);
}

// y_scale 输出（core1）：yScale[0] = max(amax, 0) / dstTypeMax。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_DYNAMIC, TPL_GROUP_MODE_TENSOR>::EmitYScale()
{
    // 核 1 写 y_scale，与核 0 转换和写 amax 并行。
    if (coreIdx_ != (actualCoreNum_ > 1 ? 1 : 0)) {
        return;
    }
    int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
    LocalTensor<float> merged = maxBuf_.Get<float>();
    LocalTensor<float> yScale = yScaleBuf_.Get<float>();
    // VF 生成 y_scale；amax 非负，dstTypeMax <= 0 时输出全零。
    if (tilingData_->dstTypeMax > 0.0f) {
        asc_vf_call<GroupedQuantEmitYScaleVf<T, U, RoundMode>>((__ubuf__ float*)merged.GetPhyAddr(),
                                                               (__ubuf__ float*)yScale.GetPhyAddr(), 1,
                                                               1.0f / tilingData_->dstTypeMax, vectorLength_);
    } else {
        Duplicate<float>(yScale, 0.0f, 1);
    }
    int32_t vectorToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(yScaleGm_, yScale, copyParams);
}

// 静态分组：仅实例化该场景需要的缓冲和排程。
template <typename T, typename U, uint64_t RoundMode>
class QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>
    : public QuantMaxV2Base<T, U, RoundMode> {
public:
    __aicore__ inline explicit QuantMaxV2Kernel(const QuantMaxV2TilingData* tilingData) : tilingData_(tilingData){};
    // 统一七参签名（与入口共用调用点对齐）。scale/groupList 均为真实输入，本特化消费。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                                GM_ADDR workspace);
    // 静态模式量化时同步计算 amax。
    // 静态完整流水：量化并求max，同步后归并输出。
    __aicore__ inline void Process()
    {
        ProcessTiles();
        SyncAll();
        MergeAmax();
    }

private:
    __aicore__ inline void ProcessTiles();
    __aicore__ inline void MergeAmax();

private:
    // 前 8 个归并核分别归约核间矩阵的连续片段，将各组局部最大值写入 slabMaxGm_。
    __aicore__ inline void MergeAmaxSlab();

private:
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void Compute(int64_t dataCount, int64_t groupIndex);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    // 输入预取深度为 X_BUFFER_NUM - 1，使搬运与计算重叠；输出延后一块写回。
    constexpr static int32_t X_BUFFER_NUM = 4;
    constexpr static int32_t Y_BUFFER_NUM = 2;
    constexpr static int32_t NUM_ONE_QUEUE = 1;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int64_t ALIGN_BYTES = 32;
    constexpr static int64_t VF_BYTES = platform::GetVRegSize();
    constexpr static int64_t FLOAT_ALIGN_ELEM = ALIGN_BYTES / sizeof(float); // 8
    // 分层归并使用 8 个归并核，在核间矩阵后追加 [8][groupStride] 暂存区。
    constexpr static int32_t MERGE_SLAB_COUNT = 8;
    constexpr static int64_t MERGE_SLAB_MIN_GROUPS = 64;

    TPipe pipe_;
    TQue<QuePosition::VECIN, X_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, Y_BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> groupListBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TQue<QuePosition::VECIN, NUM_ONE_QUEUE> inQueueGroupMax_;
    TQue<QuePosition::VECOUT, NUM_ONE_QUEUE> outQueueAmax_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> coreGroupMaxGm_;
    // 分层归并暂存区紧接核间矩阵，大小为 8 * groupStride 个 float。
    GlobalTensor<float> slabMaxGm_;
    LocalTensor<float> scaleLocal_;
    LocalTensor<int64_t> groupListLocal_;
    LocalTensor<float> maxLocal_;

    const QuantMaxV2TilingData* tilingData_;
    int32_t coreIdx_ = 0;
    int64_t blockLen_ = 1;
    float scaleValue_ = 0.0f;
    int64_t numGroups_ = 0;
    int64_t groupStride_ = 0;
    int64_t dim1_ = 0;
    uint16_t vectorLength_ = 0;
    // 整块搬运复用预构建的描述符参数。
    DataCopyExtParams fullInParams_;
    DataCopyExtParams fullOutParams_;
    int64_t totalCoreNum_;
    int64_t ubFactor_;
    int64_t actualCoreNum_;
    int64_t blockTailCoreNum_;
    int64_t blockFactor_;
    int64_t blockTailFactor_;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale, GM_ADDR workspace)
{
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    coreIdx_ = GetBlockIdx();
    numGroups_ = tilingData_->numGroups;
    groupStride_ = (numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM;
    dim1_ = tilingData_->dim1;
    totalCoreNum_ = tilingData_->totalCoreNum;
    ubFactor_ = tilingData_->ubFactor;
    fullInParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(T)), 0, 0, 0};
    fullOutParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(U)), 0, 0, 0};
    actualCoreNum_ = tilingData_->actualCoreNum;
    blockTailCoreNum_ = tilingData_->blockTailCoreNum;
    blockFactor_ = tilingData_->blockFactor;
    blockTailFactor_ = tilingData_->blockTailFactor;
    vectorLength_ = VF_BYTES / sizeof(float);

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    // 分组模板始终提供 groupList。
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    // amax 必选，始终分配归并 workspace。
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));
    coreGroupMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE),
                                    groupStride_ * totalCoreNum_);
    // 分层归并暂存区位于 workspace 核间矩阵之后。
    slabMaxGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE + groupStride_ * totalCoreNum_ * sizeof(float)),
        MERGE_SLAB_COUNT * groupStride_);

    pipe_.InitBuffer(inQueueX_, X_BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, Y_BUFFER_NUM, ubFactor_ * sizeof(U));
    pipe_.InitBuffer(scaleBuf_, numGroups_ * sizeof(float));
    pipe_.InitBuffer(groupListBuf_, numGroups_ * sizeof(int64_t));
    pipe_.InitBuffer(maxBuf_,
                     ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(float));
    // 分层归并下归并队列要被两个阶段共用：MergeAmaxSlab 每核读 ceil(cores/8) 行；MergeAmax 的
    // 终归并（core0）读全部有效片段行 ceil(cores/ceil(cores/8))（恒 <= MERGE_SLAB_COUNT 片）。
    // 因此行容量取两者上界 max(ceil(cores/8), MERGE_SLAB_COUNT)——9~56 核区间 ceil(cores/8) < 8，
    // 若只按前者分配会让终归并的 DataCopy 越过队列尾部（SEC-2.1/GEN-1.1）。Host CalcUbFactor 的
    // groupMaxSize 预算使用同一公式，两侧不得单独改动。
    const bool mergeHierarchical = numGroups_ >= MERGE_SLAB_MIN_GROUPS && actualCoreNum_ > MERGE_SLAB_COUNT;
    const int64_t slabRows = (actualCoreNum_ + MERGE_SLAB_COUNT - 1) / MERGE_SLAB_COUNT;
    const int64_t mergeRows = mergeHierarchical ? (slabRows > MERGE_SLAB_COUNT ? slabRows : MERGE_SLAB_COUNT) :
                                                  actualCoreNum_;
    pipe_.InitBuffer(inQueueGroupMax_, NUM_ONE_QUEUE, mergeRows * groupStride_ * sizeof(float) + VF_BYTES);
    pipe_.InitBuffer(outQueueAmax_, NUM_ONE_QUEUE,
                     ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(T));

    scaleLocal_ = scaleBuf_.Get<float>();
    groupListLocal_ = groupListBuf_.Get<int64_t>();
    maxLocal_ = maxBuf_.Get<float>();

    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleLocal_, scaleGm_[0], scaleCopyParams, {false, 0, 0, 0});

    DataCopyExtParams groupListCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(groupListLocal_, groupListGm_[0], groupListCopyParams, {false, 0, 0, 0});

    Duplicate<float>(maxLocal_, 0.0f, groupStride_);

    // Process 通过标量读取 scale/groupList，必须保证 MTE2 到标量流水的同步，仅 MTE2_V 不足。
    int32_t eventIDMTE2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE2ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
}

// 单趟主循环：量化与组内最大值同路（GroupedQuantComputeVf<true,true>，双链软件流水），按核遍历相交组段，深度预取 +
// 输出延迟写出。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::ProcessTiles()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }

    groupListLocal_ = groupListBuf_.Get<int64_t>();
    scaleLocal_ = scaleBuf_.Get<float>();

    int64_t coreStart = 0;
    if (coreIdx_ < actualCoreNum_ - blockTailCoreNum_) {
        blockLen_ = blockFactor_;
        coreStart = coreIdx_ * blockLen_;
    } else {
        blockLen_ = blockTailFactor_;
        coreStart = (actualCoreNum_ - blockTailCoreNum_) * blockFactor_ +
                    (coreIdx_ - (actualCoreNum_ - blockTailCoreNum_)) * blockLen_;
    }

    int64_t coreEnd = coreStart + blockLen_;
    // groupList 为累计结束行号，两端边界均单调不减。
    int64_t firstGroup = 0;
    int64_t lastGroup = 0;
    if (numGroups_ > 1) {
        int64_t low = 0;
        int64_t high = numGroups_;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            if (groupListLocal_.GetValue(mid) * dim1_ <= coreStart) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        firstGroup = (low == numGroups_) ? 0 : low;

        low = firstGroup;
        high = numGroups_;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            if (groupListLocal_.GetValue(mid) * dim1_ < coreEnd) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        lastGroup = (low == numGroups_) ? -1 : low;
    }
    // 将本组终点传递为下一组起点，减少一次标量 GetValue。
    int64_t groupStart = (firstGroup == 0) ? 0 : groupListLocal_.GetValue(firstGroup - 1) * dim1_;
    for (int64_t groupStep = firstGroup; groupStep <= lastGroup; ++groupStep) {
        int64_t groupIdx = groupStep;
        int64_t groupEnd = groupListLocal_.GetValue(groupIdx) * dim1_;
        int64_t segStart = (coreStart > groupStart) ? coreStart : groupStart;
        int64_t segEnd = (coreEnd < groupEnd) ? coreEnd : groupEnd;
        int64_t segLen = segEnd - segStart;
        groupStart = groupEnd;

        if (segLen <= 0) {
            continue;
        }

        scaleValue_ = scaleLocal_.GetValue(groupIdx);
        const int64_t ubFactor = ubFactor_;
        const int64_t loopNum = QuantMaxExactDivide(segLen, ubFactor, tilingData_->tileDivReciprocal);
        const int64_t loopTail = segLen - loopNum * ubFactor;

        // 先预取 PREFETCH_DEPTH 块，再每轮补充一块；显式计算块长度，避免设备代码使用 lambda。
        constexpr int64_t prefetchDepth = X_BUFFER_NUM - 1;
        constexpr int64_t outputDelay = Y_BUFFER_NUM - 1;
        const int64_t tileCount = loopNum + (loopTail != 0);
        for (int64_t prefetchIdx = 0; prefetchIdx < tileCount && prefetchIdx < prefetchDepth; ++prefetchIdx) {
            const int64_t seedLen = (prefetchIdx == loopNum && loopTail != 0) ? loopTail : ubFactor;
            CopyInX(seedLen, segStart + prefetchIdx * ubFactor);
        }
        for (int64_t i = 0; i < tileCount; ++i) {
            if (i + prefetchDepth < tileCount) {
                const int64_t next = i + prefetchDepth;
                const int64_t nextLen = (next == loopNum && loopTail != 0) ? loopTail : ubFactor;
                CopyInX(nextLen, segStart + next * ubFactor);
            }
            const int64_t curLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
            Compute(curLen, groupIdx);
            if (i >= outputDelay) {
                const int64_t prev = i - outputDelay;
                const int64_t prevLen = (prev == loopNum && loopTail != 0) ? loopTail : ubFactor;
                CopyOutY(prevLen, segStart + prev * ubFactor);
            }
        }
        const int64_t drainFrom = tileCount > outputDelay ? tileCount - outputDelay : 0;
        for (int64_t i = drainFrom; i < tileCount; ++i) {
            const int64_t drainLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
            CopyOutY(drainLen, segStart + i * ubFactor);
        }
    }

    LocalTensor<float> tileMaxLocal = maxBuf_.Get<float>();
    int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_ * groupStride_], tileMaxLocal, copyParams);

    // SyncAll 前等写回完成；MergeAmax 会覆盖 maxBuf_，需防止 MTE3 仍在读取，跨核屏障本身不能替代该等待。
    int32_t eventIDMTE3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    int32_t eventIDMTE3ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
}

// 输入搬运：从 GM 搬 xLen 个元素（偏移 xInOffset）进输入队列（DataCopyPad 带填充）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::CopyInX(
    int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 组段可能未对齐，统一使用带填充搬运；对齐快速搬运仅用于不分组模板。
    if (xLen == ubFactor_) {
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], fullInParams_, padParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    }
    inQueueX_.EnQue(xLocal);
}

// tile 级计算：取一块 x，调用 GroupedQuantComputeVf<true,true> 同路完成量化写出与组最大值。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::Compute(
    int64_t dataCount, int64_t groupIndex)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<U> outLocal = outQueueY_.template AllocTensor<U>();

    asc_vf_call<GroupedQuantComputeVf<T, U, RoundMode, true, true>>(
        (__ubuf__ T*)xLocal.GetPhyAddr(), (__ubuf__ U*)outLocal.GetPhyAddr(), (__ubuf__ float*)maxLocal_.GetPhyAddr(),
        scaleValue_, static_cast<uint32_t>(dataCount), groupIndex, vectorLength_);

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

// 输出搬运：把量化结果 yLen 个元素从输出队列写回 GM y（偏移 yOutOffset）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::CopyOutY(
    int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.template DeQue<U>();
    if (yLen == ubFactor_) {
        DataCopyPad(yGm_[yOutOffset], outLocal, fullOutParams_);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
        DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    }
    outQueueY_.FreeTensor(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::MergeAmaxSlab()
{
    // 前 MERGE_SLAB_COUNT 个核分别归约核间矩阵连续片段，将逐组局部最大值写入 GM。
    if (coreIdx_ >= MERGE_SLAB_COUNT || coreIdx_ >= actualCoreNum_) {
        return;
    }
    const int64_t slabCores = (actualCoreNum_ + MERGE_SLAB_COUNT - 1) / MERGE_SLAB_COUNT;
    const int64_t slabStart = coreIdx_ * slabCores;
    int64_t slabEnd = slabStart + slabCores;
    if (slabEnd > actualCoreNum_) {
        slabEnd = actualCoreNum_;
    }
    const int64_t slabLen = slabEnd - slabStart;
    if (slabLen <= 0) {
        return;
    }

    LocalTensor<float> slabLocal = maxBuf_.Get<float>();
    LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
    DataCopyExtParams slabReadParams{1, static_cast<uint32_t>(slabLen * groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(groupMax, coreGroupMaxGm_[slabStart * groupStride_], slabReadParams, {false, 0, 0, 0});
    inQueueGroupMax_.EnQue(groupMax);
    groupMax = inQueueGroupMax_.DeQue<float>();
    int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);

    if (numGroups_ >= FLOAT_ALIGN_ELEM) {
        asc_vf_call<GroupedQuantMergeAmaxVf>((__ubuf__ float*)groupMax.GetPhyAddr(),
                                             (__ubuf__ float*)slabLocal.GetPhyAddr(), numGroups_,
                                             static_cast<uint16_t>(slabLen), groupStride_, vectorLength_);
        int32_t reductionToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(reductionToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(reductionToScalar);
    } else {
        for (int64_t groupIdx = 0; groupIdx < numGroups_; ++groupIdx) {
            float maximum = 0.0f;
            for (int64_t core = 0; core < slabLen; ++core) {
                const float part = groupMax.GetValue(core * groupStride_ + groupIdx);
                maximum = part > maximum ? part : maximum;
            }
            slabLocal.SetValue(groupIdx, maximum);
        }
    }
    inQueueGroupMax_.FreeTensor(groupMax);

    int32_t vectorToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    DataCopyExtParams slabCopyParams{1, static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(slabMaxGm_[coreIdx_ * groupStride_], slabLocal, slabCopyParams);
    // 第二次 SyncAll 前等待写回，避免 slabMaxGm_ 的读写冲突。
    int32_t mte3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
    SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

// amax 收尾：G>=64 走 8 核并行分层归并（先 MergeAmaxSlab 归约片段、core0 再终归并），8<=G<64
// 各核跨步归并核间矩阵；结果经 CastAmax 写出 amax[G]。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_GROUP>::MergeAmax()
{
    // 组数至少 64 时启用 8 片分层归并，避免小组数引入额外开销。
    const bool useHierarchical = numGroups_ >= MERGE_SLAB_MIN_GROUPS && actualCoreNum_ > MERGE_SLAB_COUNT;

    if (useHierarchical) {
        MergeAmaxSlab();
        // 所有核均到达第二次屏障，非归并核直接通过。
        SyncAll();

        if (coreIdx_ >= 1) {
            return;
        }
        // 核 0 连续读入有效片段的逐组最大值，并完成最终归并。
        LocalTensor<float> merged = maxBuf_.Get<float>();
        LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
        int32_t slabCount = (actualCoreNum_ + MERGE_SLAB_COUNT - 1) / MERGE_SLAB_COUNT;
        // 有效片段数；尾部片段可能没有数据。恒有 effectiveSlabCount <= MERGE_SLAB_COUNT <= mergeRows，
        // 归并队列按该上界分配，读取长度不会越过队列尾部。
        int32_t effectiveSlabCount = static_cast<int32_t>((actualCoreNum_ + slabCount - 1) / slabCount);
        DataCopyExtParams finalCopyParams{1, static_cast<uint32_t>(effectiveSlabCount * groupStride_ * sizeof(float)),
                                          0, 0, 0};
        DataCopyPad<float>(groupMax, slabMaxGm_, finalCopyParams, {false, 0, 0, 0});
        inQueueGroupMax_.EnQue(groupMax);
        groupMax = inQueueGroupMax_.DeQue<float>();
        int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
        WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);

        asc_vf_call<GroupedQuantMergeAmaxVf>((__ubuf__ float*)groupMax.GetPhyAddr(),
                                             (__ubuf__ float*)merged.GetPhyAddr(), numGroups_,
                                             static_cast<uint16_t>(effectiveSlabCount), groupStride_, vectorLength_);
        int32_t reductionToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(reductionToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(reductionToScalar);
        inQueueGroupMax_.FreeTensor(groupMax);

        int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
        SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
        WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);

        LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();
        asc_vf_call<GroupedQuantCastAmaxVf<T, U, RoundMode>>(
            (__ubuf__ float*)merged.GetPhyAddr(), (__ubuf__ T*)amaxLocal.GetPhyAddr(), numGroups_, vectorLength_);
        outQueueAmax_.EnQue(amaxLocal);
        amaxLocal = outQueueAmax_.DeQue<T>();
        DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(amaxGm_, amaxLocal, amaxCopyParams);
        outQueueAmax_.FreeTensor(amaxLocal);
        return;
    }

    // 组数或核数较少时使用原单核归并。
    if (coreIdx_ >= 1) {
        return;
    }
    LocalTensor<float> merged = maxBuf_.Get<float>();
    LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
    // 核间矩阵连续，单次搬运替代逐核描述符；groupStride 按 8 个 float 对齐。
    // 跨核矩阵按字节长度整段搬运；DataCopyPad 自行处理对齐，不依赖 32B 前提。
    DataCopyExtParams mergeCopyParams{1, static_cast<uint32_t>(actualCoreNum_ * groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(groupMax, coreGroupMaxGm_, mergeCopyParams, {false, 0, 0, 0});
    inQueueGroupMax_.EnQue(groupMax);
    groupMax = inQueueGroupMax_.DeQue<float>();
    int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    if (numGroups_ == 1) {
        asc_vf_call<GroupedQuantMergeSingleGroupVf>(
            (__ubuf__ float*)groupMax.GetPhyAddr(), (__ubuf__ float*)merged.GetPhyAddr(),
            static_cast<uint32_t>(actualCoreNum_ * groupStride_), vectorLength_);
        int32_t reductionToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(reductionToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(reductionToScalar);
    } else if (numGroups_ >= FLOAT_ALIGN_ELEM) {
        asc_vf_call<GroupedQuantMergeAmaxVf>((__ubuf__ float*)groupMax.GetPhyAddr(),
                                             (__ubuf__ float*)merged.GetPhyAddr(), numGroups_,
                                             static_cast<uint16_t>(actualCoreNum_), groupStride_, vectorLength_);
        int32_t reductionToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        SetFlag<AscendC::HardEvent::V_S>(reductionToScalar);
        WaitFlag<AscendC::HardEvent::V_S>(reductionToScalar);
    } else {
        for (int64_t groupIdx = 0; groupIdx < numGroups_; ++groupIdx) {
            float maximum = 0.0f;
            for (int64_t core = 0; core < actualCoreNum_; ++core) {
                const float part = groupMax.GetValue(core * groupStride_ + groupIdx);
                maximum = part > maximum ? part : maximum;
            }
            merged.SetValue(groupIdx, maximum);
        }
    }
    inQueueGroupMax_.FreeTensor(groupMax);

    int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);

    LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();
    asc_vf_call<GroupedQuantCastAmaxVf<T, U, RoundMode>>(
        (__ubuf__ float*)merged.GetPhyAddr(), (__ubuf__ T*)amaxLocal.GetPhyAddr(), numGroups_, vectorLength_);
    outQueueAmax_.EnQue(amaxLocal);
    amaxLocal = outQueueAmax_.DeQue<T>();
    DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(amaxGm_, amaxLocal, amaxCopyParams);
    outQueueAmax_.FreeTensor(amaxLocal);
}

// 静态不分组：仅实例化该场景需要的缓冲和排程。
template <typename T, typename U, uint64_t RoundMode>
class QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>
    : public QuantMaxV2Base<T, U, RoundMode> {
public:
    __aicore__ inline explicit QuantMaxV2Kernel(const QuantMaxV2TilingData* tilingData) : tilingData_(tilingData){};
    // 统一七参签名（与入口共用调用点对齐）。groupList 为 nullptr 占位、不消费；scale 为单值（G=1）。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                                GM_ADDR workspace);
    // 静态模式量化时同步计算 amax。
    // 静态完整流水：量化并求max，同步后归并输出。
    __aicore__ inline void Process()
    {
        ProcessTiles();
        SyncAll();
        MergeAmax();
    }

private:
    __aicore__ inline void ProcessTiles();
    __aicore__ inline void MergeAmax();

private:
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    // 输入预取 X_BUFFER_NUM - 1 块，输出延后一块写回，与分组模板保持一致。
    constexpr static int32_t X_BUFFER_NUM = 4;
    constexpr static int32_t Y_BUFFER_NUM = 2;
    constexpr static int32_t NUM_ONE_QUEUE = 1;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int64_t ALIGN_BYTES = 32;
    constexpr static int64_t VF_BYTES = platform::GetVRegSize();
    constexpr static int64_t FLOAT_ALIGN_ELEM = ALIGN_BYTES / sizeof(float); // 8

    TPipe pipe_;
    TQue<QuePosition::VECIN, X_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, Y_BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TQue<QuePosition::VECIN, NUM_ONE_QUEUE> inQueueGroupMax_;
    TQue<QuePosition::VECOUT, NUM_ONE_QUEUE> outQueueAmax_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> coreGroupMaxGm_;
    LocalTensor<float> scaleLocal_;
    LocalTensor<float> maxLocal_;

    const QuantMaxV2TilingData* tilingData_;
    int32_t coreIdx_ = 0;
    float scaleValue_ = 0.0f;
    int64_t groupStride_ = 0; // 固定单组，按 8 个 float 对齐。
    uint16_t vectorLength_ = 0;
    // 整块搬运复用预构建的描述符参数。
    DataCopyExtParams fullInParams_;
    DataCopyExtParams fullOutParams_;
    int64_t totalCoreNum_;
    int64_t ubFactor_;
    int64_t actualCoreNum_;
    int64_t blockTailCoreNum_;
    int64_t blockFactor_;
    int64_t blockTailFactor_;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale, GM_ADDR workspace)
{
    (void)groupList;
    (void)yScale;
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    coreIdx_ = GetBlockIdx();
    groupStride_ = FLOAT_ALIGN_ELEM; // 单组结果填充到 8 个 float。
    totalCoreNum_ = tilingData_->totalCoreNum;
    ubFactor_ = tilingData_->ubFactor;
    fullInParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(T)), 0, 0, 0};
    fullOutParams_ = DataCopyExtParams{1, static_cast<uint32_t>(ubFactor_ * sizeof(U)), 0, 0, 0};
    actualCoreNum_ = tilingData_->actualCoreNum;
    blockTailCoreNum_ = tilingData_->blockTailCoreNum;
    blockFactor_ = tilingData_->blockFactor;
    blockTailFactor_ = tilingData_->blockTailFactor;
    vectorLength_ = VF_BYTES / sizeof(float);

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    // amax 必选，始终分配归并 workspace。
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));
    coreGroupMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE),
                                    groupStride_ * totalCoreNum_);

    pipe_.InitBuffer(inQueueX_, X_BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, Y_BUFFER_NUM, ubFactor_ * sizeof(U));
    pipe_.InitBuffer(scaleBuf_, sizeof(float));
    pipe_.InitBuffer(maxBuf_, groupStride_ * sizeof(float));
    pipe_.InitBuffer(inQueueGroupMax_, NUM_ONE_QUEUE, groupStride_ * totalCoreNum_ * sizeof(float) + VF_BYTES);
    pipe_.InitBuffer(outQueueAmax_, NUM_ONE_QUEUE, groupStride_ * sizeof(T));

    scaleLocal_ = scaleBuf_.Get<float>();
    maxLocal_ = maxBuf_.Get<float>();

    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleLocal_, scaleGm_[0], scaleCopyParams, {false, 0, 0, 0});

    Duplicate<float>(maxLocal_, 0.0f, groupStride_);

    // Process 通过标量读取 scale，必须保证 MTE2 到标量流水的同步。
    int32_t eventIDMTE2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE2ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2ToS);
}

// 单趟主循环（单段连续，G==1）：量化与全局最大值同路（双链软件流水），深度预取 + 输出延迟写出。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::ProcessTiles()
{
    if (coreIdx_ >= actualCoreNum_) {
        return;
    }

    scaleLocal_ = scaleBuf_.Get<float>();
    // 整个张量共用一个 scale，只读取一次，无逐组加载。
    scaleValue_ = scaleLocal_.GetValue(0);

    int64_t coreStart = 0;
    int64_t blockLen = 0;
    if (coreIdx_ < actualCoreNum_ - blockTailCoreNum_) {
        blockLen = blockFactor_;
        coreStart = coreIdx_ * blockLen;
    } else {
        blockLen = blockTailFactor_;
        coreStart = (actualCoreNum_ - blockTailCoreNum_) * blockFactor_ +
                    (coreIdx_ - (actualCoreNum_ - blockTailCoreNum_)) * blockLen;
    }

    const int64_t segLen = blockLen;
    const int64_t ubFactor = ubFactor_;
    const int64_t loopNum = QuantMaxExactDivide(segLen, ubFactor, tilingData_->tileDivReciprocal);
    const int64_t loopTail = segLen - loopNum * ubFactor;

    // 先预取 PREFETCH_DEPTH 块，再每轮补充一块；不分组仅处理单个连续段。
    constexpr int64_t prefetchDepth = X_BUFFER_NUM - 1;
    constexpr int64_t outputDelay = Y_BUFFER_NUM - 1;
    const int64_t tileCount = loopNum + (loopTail != 0);
    for (int64_t prefetchIdx = 0; prefetchIdx < tileCount && prefetchIdx < prefetchDepth; ++prefetchIdx) {
        const int64_t seedLen = (prefetchIdx == loopNum && loopTail != 0) ? loopTail : ubFactor;
        CopyInX(seedLen, coreStart + prefetchIdx * ubFactor);
    }
    for (int64_t i = 0; i < tileCount; ++i) {
        if (i + prefetchDepth < tileCount) {
            const int64_t next = i + prefetchDepth;
            const int64_t nextLen = (next == loopNum && loopTail != 0) ? loopTail : ubFactor;
            CopyInX(nextLen, coreStart + next * ubFactor);
        }
        const int64_t curLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
        Compute(curLen);
        if (i >= outputDelay) {
            const int64_t prev = i - outputDelay;
            const int64_t prevLen = (prev == loopNum && loopTail != 0) ? loopTail : ubFactor;
            CopyOutY(prevLen, coreStart + prev * ubFactor);
        }
    }
    const int64_t drainFrom = tileCount > outputDelay ? tileCount - outputDelay : 0;
    for (int64_t i = drainFrom; i < tileCount; ++i) {
        const int64_t drainLen = (i == loopNum && loopTail != 0) ? loopTail : ubFactor;
        CopyOutY(drainLen, coreStart + i * ubFactor);
    }

    LocalTensor<float> tileMaxLocal = maxBuf_.Get<float>();
    int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_ * groupStride_], tileMaxLocal, copyParams);

    // SyncAll 前等待写回，避免 maxBuf_ 的读写冲突。
    int32_t eventIDMTE3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    int32_t eventIDMTE3ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
}

// 输入搬运：从 GM 搬 xLen 个元素（偏移 xInOffset）进输入队列（DataCopyPad 带填充）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::CopyInX(
    int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 统一使用 DataCopyPad 按字节长度搬运：blockFactor 不保证 32B 对齐，Pad 版自行处理，
    // 不再需要运行时对齐判定分支。整 tile 复用预置参数，尾 tile 按实际长度构造。
    if (xLen == ubFactor_) {
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], fullInParams_, padParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    }
    inQueueX_.EnQue(xLocal);
}

// tile 级计算：取一块 x，调用 GroupedQuantComputeVf<true,true> 同路完成量化写出与全局最大值。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::Compute(
    int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<U> outLocal = outQueueY_.template AllocTensor<U>();

    asc_vf_call<GroupedQuantComputeVf<T, U, RoundMode, true, true>>(
        (__ubuf__ T*)xLocal.GetPhyAddr(), (__ubuf__ U*)outLocal.GetPhyAddr(), (__ubuf__ float*)maxLocal_.GetPhyAddr(),
        scaleValue_, static_cast<uint32_t>(dataCount), 0, vectorLength_);

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

// 输出搬运：把量化结果 yLen 个元素从输出队列写回 GM y（偏移 yOutOffset）。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::CopyOutY(
    int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.template DeQue<U>();
    if (yLen == ubFactor_) {
        DataCopyPad(yGm_[yOutOffset], outLocal, fullOutParams_);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
        DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    }
    outQueueY_.FreeTensor(outLocal);
}

// amax 收尾：单组跨核归并（各核局部最大值连续存放）后转为输入 dtype 写出 amax[1]。
template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxV2Kernel<T, U, RoundMode, TPL_QUANT_MODE_STATIC, TPL_GROUP_MODE_TENSOR>::MergeAmax()
{
    // 固定 G=1，归并各核的单组最大值。
    if (coreIdx_ >= 1) {
        return;
    }
    LocalTensor<float> merged = maxBuf_.Get<float>();
    LocalTensor<float> groupMax = inQueueGroupMax_.AllocTensor<float>();
    // 跨核矩阵按字节长度整段搬运；DataCopyPad 自行处理对齐，不依赖 32B 前提。
    DataCopyExtParams mergeCopyParams{1, static_cast<uint32_t>(actualCoreNum_ * groupStride_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(groupMax, coreGroupMaxGm_, mergeCopyParams, {false, 0, 0, 0});
    inQueueGroupMax_.EnQue(groupMax);
    groupMax = inQueueGroupMax_.DeQue<float>();
    int32_t copyToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
    SetFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<AscendC::HardEvent::MTE2_S>(copyToScalar);

    // 各核单组最大值填充位置为零，用一个 VF 归约连续矩阵。
    asc_vf_call<GroupedQuantMergeSingleGroupVf>((__ubuf__ float*)groupMax.GetPhyAddr(),
                                                (__ubuf__ float*)merged.GetPhyAddr(),
                                                static_cast<uint32_t>(actualCoreNum_ * groupStride_), vectorLength_);
    int32_t reductionToScalar = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
    SetFlag<AscendC::HardEvent::V_S>(reductionToScalar);
    WaitFlag<AscendC::HardEvent::V_S>(reductionToScalar);
    inQueueGroupMax_.FreeTensor(groupMax);

    LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        // BF16 绝对值的最大值可以由 BF16 精确表示，无需算术舍入。
        const uint32_t maxBits = merged.template ReinterpretCast<uint32_t>().GetValue(0);
        amaxLocal.template ReinterpretCast<uint16_t>().SetValue(0, static_cast<uint16_t>(maxBits >> 16));
        int32_t scalarToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3);
        SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
    } else {
        int32_t scalarToVector = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
        SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
        WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
        asc_vf_call<GroupedQuantCastAmaxVf<T, U, RoundMode>>((__ubuf__ float*)merged.GetPhyAddr(),
                                                             (__ubuf__ T*)amaxLocal.GetPhyAddr(), 1, vectorLength_);
    }
    outQueueAmax_.EnQue(amaxLocal);
    amaxLocal = outQueueAmax_.DeQue<T>();
    DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(amaxGm_, amaxLocal, amaxCopyParams);
    outQueueAmax_.FreeTensor(amaxLocal);
}

} // namespace QuantMaxV2
#endif // QUANT_MAX_V2_REGBASE_H
