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
 * \file gn_training_reduce_base.h
 * \brief GNTrainingReduce base (non-group) kernel implementation.
 */

#ifndef GN_TRAINING_REDUCE_BASE_H
#define GN_TRAINING_REDUCE_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_common/op_kernel/platform_util.h"
#include "op_common/op_kernel/math_util.h"
#include "adv_api/reduce/reduce.h"
#include "gn_training_reduce_tiling_struct.h"

namespace NsGNTrainingReduce {

using namespace AscendC;

constexpr int32_t N_REDUCES = 2; // Σx / Σx² 两次独立归约
constexpr int32_t MAX_PATTERN_RANK = GN_TRAINING_REDUCE_MAX_PATTERN_RANK;
constexpr uint32_t VL_BYTES = Ops::Base::GetVRegSize(); // 向量寄存器字节数
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float);  // 单次 repeat fp32 lane 数
constexpr uint16_t REP_F32_U16 = static_cast<uint16_t>(REP_F32);
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();  // 32B datablock
constexpr uint32_t UB_BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float); // = 8
constexpr int32_t AXIS_INTERVAL = 2;                              // 偶位 A、奇位 R
constexpr size_t REDUCE_SHAPE_DIM = 2;                            // ReduceSum srcShape 二维
constexpr uint64_t UINT64_BITS = 64;
constexpr uint64_t UINT64_TOP_BIT_IDX = UINT64_BITS - 1;
constexpr uint64_t NEAREST_POW2_SMALL_BOUND = 2; // v ≤ 2 → 最近二次幂 1
constexpr float PAD_CLEAR_VALUE = 0.0f;          // sum reducer pad_value

// Σx 补偿（double-double）标量归约的 R 上限：单 tile 覆盖整段 R 且 R 不超过该值时启用。
// R 越大组内随机和的条件数越低、fp32 树归约已满足容差，故无需补偿；该阈值同时把标量
// 归约工作量限制在 laneA*rProd ≤ preBuf/4 的量级。
constexpr int64_t COMPENSATED_SUM_MAX_R = 4096;
// VEC 访问 UB 要求 32B 对齐：对齐补偿路径的每 A 标量结果按 8 float 步长写回 src 临时槽。
constexpr uint32_t COMPENSATED_OUT_STRIDE = 8;

// Σx 累加数值稳定缩放（极端输入超出 fp32 上限时须按
// IEEE 754 溢出为 ±Inf，不得出现 +Inf + -Inf 的伪 NaN）。缩放因子取 2 的幂 → 逐位精确
// （significand 不变），有限结果与不缩放 fp32 累加完全一致；指数余量 2^-32 使中间 partial
// 不溢出，末次反缩放再把精确符号的 ±Inf 还原，与 fp64 golden 的溢出符号对齐。
constexpr float SUM_ACC_SCALE = 1.0f / 4294967296.0f; // 2^-32，精确
constexpr float SUM_ACC_UNSCALE = 4294967296.0f;      // 2^32，精确

constexpr AscendC::Reg::CastTrait CAST_TRAIT_TO_FP32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};

// ── PreElewise VF：processIdx=0 仅 Cast；=1 Cast+Square ──
template <typename DType>
__simd_vf__ inline void CastOnlyVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                       uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        }
        AscendC::Reg::Muls(f32Reg, f32Reg, SUM_ACC_SCALE, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

template <typename DType>
__simd_vf__ inline void CastSquareVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        }
        AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// ── pad 清零 VF──
__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    for (uint16_t aIdx = 0; aIdx < aU16; ++aIdx) {
        int32_t aOff = static_cast<int32_t>(aIdx) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off = aOff + static_cast<int32_t>(extStart) +
                          static_cast<int32_t>(r) * static_cast<int32_t>(REP_F32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                 uint16_t repCount)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                     int32_t windowOff, uint32_t padEnd, uint32_t partialStartInBlock)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    uint32_t cntEnd = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);
    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// ── 二分主块 + 尾块合并──
__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                          uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

// ── Σx 补偿（double-double two-sum）的向量化归约（AR 布局：每 A lane 的 R 连续）──
// 对每个 A lane 的 rProd 个 fp32 元素做 error-free two-sum，向量 lane 各自独立累加
// （hi/lo 两个累加器），末尾对 hi、lo 分别做向量水平归约再相加。数值语义与标量版
// 完全一致（含 ±Inf/NaN：NaN 误差 lane 清零，令 hi 主导传播），但标量开销由
// O(laneA·rProd) 降为 O(laneA·log2(VL))，消除小 R 补偿路径的标量瓶颈。
__simd_vf__ inline void CompensatedSumArVfImpl(__ubuf__ float* src, uint32_t laneA, uint32_t rProd, uint32_t chunkLanes)
{
    // 每 lane 做 Neumaier 两和（精度 ~2^-48），逐 chunk 向量累加后再跨 lane 归约；
    // 同时以 psum 记录普通和，输入含 ±Inf/NaN 时回退到 psum（与 golden/标量路径同语义）。
    // 注意：Reg::Duplicate(dst,src,mask) 是「广播 src[pos]」而非逐 lane 拷贝，故用 t+0 复制。
    // 掩码现取现用（CreateMask/UpdateMask 会复用同一物理掩码寄存器）。
    AscendC::Reg::RegTensor<float> hi, lo, psum, x, t, bb, s1, err, zero, shi, slo, ssum, res;
    AscendC::Reg::MaskReg one = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(zero, 0.0f);
    for (uint32_t a = 0; a < laneA; ++a) {
        AscendC::Reg::Duplicate(hi, 0.0f);
        AscendC::Reg::Duplicate(lo, 0.0f);
        AscendC::Reg::Duplicate(psum, 0.0f);
        __ubuf__ float* sp = src + a * rProd;
        uint32_t r = 0;
        for (; r + chunkLanes <= rProd; r += chunkLanes) {
            uint32_t cnt = chunkLanes;
            AscendC::Reg::MaskReg m = AscendC::Reg::UpdateMask<float>(cnt);
            AscendC::Reg::LoadAlign(x, sp + r);
            AscendC::Reg::Add(psum, psum, x, m);
            AscendC::Reg::Add(t, hi, x, m);     // t = hi + x
            AscendC::Reg::Sub(bb, t, hi, m);    // bb = t - hi
            AscendC::Reg::Sub(s1, t, bb, m);    // s1 = t - bb
            AscendC::Reg::Sub(s1, hi, s1, m);   // s1 = hi - (t - bb)
            AscendC::Reg::Sub(err, x, bb, m);   // err = x - bb
            AscendC::Reg::Add(err, s1, err, m); // err = (hi-(t-bb)) + (x-bb)
            AscendC::Reg::Add(lo, lo, err, m);  // lo += err
            AscendC::Reg::Add(hi, t, zero, m);  // hi = t
        }
        if (r < rProd) {
            uint32_t rem = rProd - r;
            AscendC::Reg::MaskReg mt = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::LoadAlign(x, sp + r);
            // 累加器用 MERGING：尾块只更新有效 lane，保留 hi/lo/psum 其余 lane 的既有累加
            // （默认 ZEROING 会把非活跃 lane 清零，直接抹掉其余 lane 的累加）。
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(psum, psum, x, mt);
            AscendC::Reg::Add(t, hi, x, mt);
            AscendC::Reg::Sub(bb, t, hi, mt);
            AscendC::Reg::Sub(s1, t, bb, mt);
            AscendC::Reg::Sub(s1, hi, s1, mt);
            AscendC::Reg::Sub(err, x, bb, mt);
            AscendC::Reg::Add(err, s1, err, mt);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(lo, lo, err, mt);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(hi, t, zero, mt);
        }
        uint32_t cntf = chunkLanes;
        AscendC::Reg::MaskReg fm = AscendC::Reg::UpdateMask<float>(cntf);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(shi, hi, fm);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(slo, lo, fm);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(ssum, psum, fm);
        AscendC::Reg::Add(res, shi, slo, one);
        AscendC::Reg::MaskReg resNan;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(resNan, res, res, one);
        AscendC::Reg::Select(res, ssum, res, resNan);
        AscendC::Reg::StoreAlign(src + a * COMPENSATED_OUT_STRIDE, res, one);
    }
}

// ── Σx 补偿归约（RA 布局：每 R 行 A 连续，元素 (a,r) 位于 pre[a + r*laneA]）──
// 与 AR 版对偶：向量 lane = A，逐 R 行对整段 A 做 Neumaier 两和，末尾 hi+lo 即各 A 的
// 补偿和，按 A 连续直接写 cache（无需标量中转）。要求 laneA%8==0 保证每行行首 32B 对齐。
__simd_vf__ inline void CompensatedSumRaVfImpl(__ubuf__ float* src, __ubuf__ float* dst, uint32_t laneA, uint32_t rProd)
{
    AscendC::Reg::RegTensor<float> hi, lo, psum, x, t, bb, s1, err, zero, res;
    AscendC::Reg::Duplicate(zero, 0.0f);
    for (uint32_t a0 = 0; a0 < laneA; a0 += REP_F32) {
        uint32_t na = laneA - a0;
        uint32_t cntf = na;
        AscendC::Reg::MaskReg fm = (na >= REP_F32) ? AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>() :
                                                     AscendC::Reg::UpdateMask<float>(cntf);
        AscendC::Reg::Duplicate(hi, 0.0f);
        AscendC::Reg::Duplicate(lo, 0.0f);
        AscendC::Reg::Duplicate(psum, 0.0f);
        for (uint32_t r = 0; r < rProd; ++r) {
            AscendC::Reg::LoadAlign(x, src + r * laneA + a0);
            // 累加器 MERGING：不足整向量的末行只更新有效 lane，保留其余 lane 的累加。
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(psum, psum, x, fm);
            AscendC::Reg::Add(t, hi, x, fm);
            AscendC::Reg::Sub(bb, t, hi, fm);
            AscendC::Reg::Sub(s1, t, bb, fm);
            AscendC::Reg::Sub(s1, hi, s1, fm);
            AscendC::Reg::Sub(err, x, bb, fm);
            AscendC::Reg::Add(err, s1, err, fm);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(lo, lo, err, fm);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(hi, t, zero, fm);
        }
        // 每个 A lane 独立持有该 A 的补偿和 hi+lo，不做跨 lane 归约；输入含 ±Inf/NaN
        // 时逐 lane 回退到普通和 psum。
        AscendC::Reg::Add(res, hi, lo, fm);
        AscendC::Reg::MaskReg resNan;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(resNan, res, res, fm);
        AscendC::Reg::Select(res, psum, res, resNan);
        AscendC::Reg::StoreAlign(dst + a0, res, fm);
    }
}

// ── DoCaching：把层 levelOff 就地吸收其下 j ∈ [0, cacheLevelCnt) 低层并写回──
__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                        int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);
        for (uint16_t j = 0; j < cacheLevelCnt; ++j) {
            int32_t lowerLevelOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerLevelOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// ── PostElewise：原始矩直出（fp32 密集搬移；Σx 路径末次反缩放，）──
__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* rootPtr, __ubuf__ float* outPtr, uint32_t laneN,
                                          uint16_t repeatTime, float scale)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
        AscendC::Reg::Muls(f32Reg, f32Reg, scale, mask);
        AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
    }
}

struct UBAxisDesc {
    int32_t gmIdx;
    int64_t actualNum;
    int64_t paddedNum;
    int64_t gmStride;
};

template <typename DType>
class GNTrainingReduceBaseKernel {
public:
    using DT = DType;

    __aicore__ inline GNTrainingReduceBaseKernel() {}
    __aicore__ inline ~GNTrainingReduceBaseKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, const GNTrainingReduceTilingData* td,
                                AscendC::TPipe* pipe);
    __aicore__ inline void Process(int32_t processIdx);

protected:
    __aicore__ inline void UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd);
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx);
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen);

    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    __aicore__ inline void PostElewise(int64_t aLen);
    __aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen, int32_t processIdx);

    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen);

    __aicore__ inline void PreElewise(__ubuf__ DT* src, __ubuf__ float* dst);
    __aicore__ inline void ReduceSumCompensated(int32_t levelOff);
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf);
    __aicore__ inline void DoCachingVf(uint16_t cacheID);

    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;

    const GNTrainingReduceTilingData* td_ = nullptr;
    bool isTailR_ = false;
    // 单 tile 即覆盖整段 R 且 R 较小时，Σx 走补偿（double-double）标量归约以消除
    // 全值域极端输入下的灾难性抵消误差（见 ReduceSumCompensated）；Σx² 全正无抵消，仍走向量路径。
    bool useCompensatedSum_ = false;
    int32_t processIdx_ = 0;
    int64_t rSplitChunkCnt_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    GlobalTensor<DT> gmX_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> preInBuf_;
    TBuf<QuePosition::VECCALC> preReduceResult_;
    TBuf<QuePosition::VECCALC> preReduceResultTail_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    event_t evMTE2toV_;
    event_t evVtoMTE2_;
    event_t evVtoMTE3_;
    event_t evMte3toV_;
    event_t evVtoS_;
    event_t evStoV_;
};

template <typename DType>
__aicore__ inline int32_t GNTrainingReduceBaseKernel<DType>::LastAAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType>
__aicore__ inline int32_t GNTrainingReduceBaseKernel<DType>::LastRAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

template <typename DType>
__aicore__ inline uint64_t GNTrainingReduceBaseKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= NEAREST_POW2_SMALL_BOUND) {
        return 1;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = UINT64_TOP_BIT_IDX - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

template <typename DType>
__aicore__ inline uint64_t GNTrainingReduceBaseKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

template <typename DType>
__aicore__ inline uint16_t GNTrainingReduceBaseKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType>
__aicore__ inline int64_t GNTrainingReduceBaseKernel<DType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
    const int64_t start = rChunkIdx * td_->rUbFactor;
    return (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum,
                                                               const GNTrainingReduceTilingData* td,
                                                               AscendC::TPipe* pipe)
{
    td_ = td;
    pipe_ = pipe;
    isTailR_ = (td_->axisNum % AXIS_INTERVAL == 0);

    // 仅在「单个 R tile 即容纳整段 R」且 R 不超过阈值时启用补偿求和：小分组的
    // 组内条件数可高达 ~1e9，fp32 归约无法满足 2^-13 相对容差，而 R 较大时随机
    // 分组和的条件数随 √R 下降，fp32 已足够。补偿路径按 interleaved fp32 两和
    // （two-sum）累加，精度约 2^-48，标量开销有界（laneA*rProd ≤ preBuf/4）。
    const int64_t compRProd = td_->rUbFactorAlign * td_->innerRProdAlign;
    useCompensatedSum_ = (td_->rLoopCntTotal == 1) && (compRProd <= COMPENSATED_SUM_MAX_R);

    rSplitChunkCnt_ = Ops::Base::CeilDiv(td_->axisShape[td_->rSplitIdx], td_->rUbFactor);
    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(td_->rLoopCntTotal)));
    bisectionTail_ = td_->rLoopCntTotal - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    {
        int64_t outStrideAcc = 1;
        for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
            outStride_[i] = 1;
        }
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % AXIS_INTERVAL == 0) {
                outStride_[i] = outStrideAcc;
                outStrideAcc *= td_->axisShape[i];
            }
        }
    }

    gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
    sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
    squareSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(squareSum));

    // 逻辑大小之外各多分配一个向量的读余量：VF 中无 mask 的 Reg::LoadAlign 按整寄存器
    // 宽度读取，末次 repeat 的读会越过逻辑元素末尾最多 63 个 fp32（见 VF 实现），这些
    // 越界 lane 虽被后续 mask 丢弃，但按内存安全契约不得越过本 buffer，故留出余量。
    // host tiling 以同一常量把这 4 份余量计入 UB 预算，两侧必须保持一致。
    static_assert(static_cast<int64_t>(VL_BYTES) == GN_TRAINING_REDUCE_UB_READ_SLACK,
                  "UB read slack must equal one vector register width");
    const int64_t preBufAllocSize = td_->preBufSize + GN_TRAINING_REDUCE_UB_READ_SLACK;
    pipe_->InitBuffer(preInBuf_, preBufAllocSize);
    pipe_->InitBuffer(preReduceResult_, preBufAllocSize);
    pipe_->InitBuffer(preReduceResultTail_, preBufAllocSize);
    pipe_->InitBuffer(cacheBuf_, td_->cacheBufUbSize + GN_TRAINING_REDUCE_UB_READ_SLACK);
    pipe_->InitBuffer(outBuf_, td_->postBufSize);

    evMTE2toV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evVtoMTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    evVtoMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    evMte3toV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    evVtoS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    evStoV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
    } else {
        aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
    }
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::UnravelALoop(int64_t aLoopIdx, int64_t aIdx[],
                                                                       int64_t& aSplitChunkIdx)
{
    int64_t aLoopRem = aLoopIdx;
    aSplitChunkIdx = aLoopRem % td_->aSplitChunkCnt;
    aLoopRem /= td_->aSplitChunkCnt;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        aIdx[k] = aLoopRem % td_->axisShape[k];
        aLoopRem /= td_->axisShape[k];
    }
}

template <typename DType>
__aicore__ inline int64_t GNTrainingReduceBaseKernel<DType>::UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[],
                                                                          int64_t& rChunkIdx, int64_t& rLen)
{
    rChunkIdx = rIdx % rSplitChunkCnt_;
    int64_t rLoopRem = rIdx / rSplitChunkCnt_;
    int64_t gmOff = 0;
    for (int32_t k = td_->rSplitIdx - AXIS_INTERVAL; k >= 1; k -= AXIS_INTERVAL) {
        rOuterIdx[k] = rLoopRem % td_->axisShape[k];
        rLoopRem /= td_->axisShape[k];
        gmOff += rOuterIdx[k] * td_->axisStride[k];
    }
    rLen = RLenOfChunk(rChunkIdx);
    return gmOff + rChunkIdx * td_->rUbFactor * td_->axisStride[td_->rSplitIdx];
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::Process(int32_t processIdx)
{
    processIdx_ = processIdx;
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return;
    }

    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    UnravelBlockLoop(aLoopStart, aLoopEnd);

    const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
    const int64_t aSplitStride = td_->axisStride[td_->aSplitIdx];
    const int64_t aSplitOutStr = outStride_[td_->aSplitIdx];

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
            chunkOutOff += aIdx[k] * outStride_[k];
        }

        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEnd = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : td_->aUbFactor;

        // Tuple Reduce：跨 processIdx 复用 preInBuf / outBuf，反向同步按「全局」首末轮裁剪
        // （processIdx==0 首个 aLoop 跳 Wait；processIdx==N_REDUCES-1 末个 aLoop 跳 Set）。
        const bool isVeryFirstALoop = (processIdx == 0) && (aLoopIdx == aLoopStart);
        const bool isVeryLastALoop = (processIdx == N_REDUCES - 1) && (aLoopIdx == aLoopEnd - 1);
        if (!isVeryFirstALoop) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            WaitFlag<HardEvent::MTE3_V>(evMte3toV_);
        }

        chunkGmOff += aChunkStart * aSplitStride;
        chunkOutOff += aChunkStart * aSplitOutStr;

        DoOneAChunk(chunkGmOff, aLen);
        PostElewise(aLen);
        CopyOut(chunkOutOff, aLen, processIdx_);

        if (!isVeryLastALoop) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            SetFlag<HardEvent::MTE3_V>(evMte3toV_);
        }
    }
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);
        }

        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkIdxMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkIdxMain, rLenMain);

        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr());
        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());

        DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain);

        SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);

        PreElewise(preIn, preRes);
        ClearChunkExtensionVf(preRes, rLenMain);
        if (isTailR_) {
            ClearInnerBurstTailPadVf(preRes, rLenMain);
        }

        if (rIdx < bisectionTail_) {
            int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
            int64_t rChunkIdxTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail = UnravelRLoop(rIdx + bisectionPos_, rOuterIdxTail, rChunkIdxTail, rLenTail);

            __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(
                preReduceResultTail_.Get<float>().GetPhyAddr());

            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);

            DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail);

            SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);

            PreElewise(preIn, preResTail);
            ClearChunkExtensionVf(preResTail, rLenTail);
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preResTail, rLenTail);
            }
            MergeTmpBufVf(preRes, preResTail);
        }

        const uint16_t cacheID = GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = Ops::Base::CeilAlign(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

        if (processIdx_ == 0 && useCompensatedSum_) {
            // Σx 且单 tile 覆盖整段 R：走补偿标量归约，消除全值域极端输入的抵消误差。
            ReduceSumCompensated(levelOff);
        } else if (isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneA,
                                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                   laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        }
        DoCachingVf(cacheID);

        if (rIdx != bisectionPos_ - 1) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
        }
    }
}

template <typename DType>
__aicore__ inline int32_t GNTrainingReduceBaseKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DT));

    if (isTailR_) {
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else if (i == lastR) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem);
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    } else {
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else if (i == lastA) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem);
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    }
    return k;
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes);

    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    const int64_t copyPadBytes = Ops::Base::CeilAlign(static_cast<int64_t>(extParams.blockLen),
                                                      static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;
    extParams.dstStride = (target0Bytes - copyPadBytes) / static_cast<int64_t>(UB_BLOCK_BYTES);

    if (axisCnt > 1) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[1].actualNum);
        extParams.srcStride = ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen);
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }
    extParams.rsv = 0;

    int64_t ubStride[MAX_PATTERN_RANK] = {0};
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < axisCnt; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
    }

    if (axisCnt > 2) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].actualNum);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
        loopParams.loop2Size = 1;
    }
    if (axisCnt > 3) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].actualNum);
        loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
    }

    const bool useLoopMode = (axisCnt > 2);
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    int64_t outerProd = 1;
    for (int32_t k = 4; k < axisCnt; ++k) {
        outerProd *= ubAxes[k].actualNum;
    }

    auto preInLocal = preInBuf_.Get<DT>();
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t outerRem = outerFlat;
        for (int32_t k = 4; k < axisCnt; ++k) {
            const int64_t axisSize = ubAxes[k].actualNum;
            const int64_t axisIdx = outerRem % axisSize;
            outerRem /= axisSize;
            addGmOffElem += axisIdx * ubAxes[k].gmStride;
            addUbOffBytes += axisIdx * ubStride[k];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preInLocal[ubOffElems], gmX_[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::PreElewise(__ubuf__ DT* src, __ubuf__ float* dst)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(
        Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    if (processIdx_ == 0) {
        asc_vf_call<CastOnlyVfImpl<DT>>(src, dst, totalElems, repeatTime);
    } else {
        asc_vf_call<CastSquareVfImpl<DT>>(src, dst, totalElems, repeatTime);
    }
}

// ── Σx 补偿（double-double）标量归约（仅单 tile 覆盖整段 R 且 R 较小时）──
// 逐 (A lane) 用 error-free two-sum 累加 R 个 pre-scale 后的 fp32 元素：
//   t = s + x; bb = t - s; err = (s - (t - bb)) + (x - bb); c += err; s = t
// 结果 s + c 含约 48 bit 有效精度，可对齐 fp64 golden 在全值域极端输入下的结果；
// 高部分与低部分分别经 PostElewise 反缩放（sum 路径）后写回。布局随 isTailR_ 切换：
//   AR: src[A, Rprod] row-major → (a,r) @ a*Rprod + r
//   RA: src[Rprod, A] row-major → (r,a) @ r*laneA + a
template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::ReduceSumCompensated(int32_t levelOff)
{
    auto pre = preReduceResult_.template Get<float>();
    auto cache = cacheBuf_.template Get<float>();
    const int32_t laneA = static_cast<int32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const int32_t rProd = static_cast<int32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
    // AR 布局：laneA 个 A lane 各含 rProd 连续元素 → 向量 lane = R，逐 A 做两和；
    // RA 布局：rProd 行各含 laneA 连续 A → 向量 lane = A，逐 R 行做两和。
    // 不满足整向量宽/对齐条件时回退标量精确 two-sum（数值语义一致）。
    if (isTailR_ && rProd >= static_cast<int32_t>(COMPENSATED_OUT_STRIDE) &&
        (rProd % static_cast<int32_t>(COMPENSATED_OUT_STRIDE) == 0)) {
        // rProd ≥ 整向量宽用 64-lane chunk；不足一个向量时降到 8-lane chunk，保证每 lane
        // 仍累积多个元素（否则退化为普通和、丢失补偿）。
        uint32_t chunkLanes = (rProd >= static_cast<int32_t>(REP_F32)) ? REP_F32 : COMPENSATED_OUT_STRIDE;
        __ubuf__ float* prePtr = reinterpret_cast<__ubuf__ float*>(pre.GetPhyAddr());
        asc_vf_call<CompensatedSumArVfImpl>(prePtr, static_cast<uint32_t>(laneA), static_cast<uint32_t>(rProd),
                                            chunkLanes);
        // VF 结果写回 src 的 32B 对齐临时槽；标量读取后写入 cache。
        SetFlag<HardEvent::V_S>(evVtoS_);
        WaitFlag<HardEvent::V_S>(evVtoS_);
        for (int32_t a = 0; a < laneA; ++a) {
            cache.SetValue(static_cast<uint32_t>(levelOff + a),
                           pre.GetValue(static_cast<uint32_t>(a * COMPENSATED_OUT_STRIDE)));
        }
        SetFlag<HardEvent::S_V>(evStoV_);
        WaitFlag<HardEvent::S_V>(evStoV_);
        return;
    }
    if (!isTailR_ && (laneA % static_cast<int32_t>(COMPENSATED_OUT_STRIDE) == 0)) {
        __ubuf__ float* prePtr = reinterpret_cast<__ubuf__ float*>(pre.GetPhyAddr());
        __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cache.GetPhyAddr()) +
                                   static_cast<int64_t>(levelOff);
        asc_vf_call<CompensatedSumRaVfImpl>(prePtr, cachePtr, static_cast<uint32_t>(laneA),
                                            static_cast<uint32_t>(rProd));
        return;
    }
    // PreElewise / ClearChunkExtension 为 VF（向量 pipe）；标量回退走标量 pipe 读同一 UB。
    SetFlag<HardEvent::V_S>(evVtoS_);
    WaitFlag<HardEvent::V_S>(evVtoS_);
    for (int32_t a = 0; a < laneA; ++a) {
        float s = 0.0f;
        float c = 0.0f;
        const int32_t base = isTailR_ ? (a * rProd) : a;
        const int32_t step = isTailR_ ? 1 : laneA;
        for (int32_t r = 0; r < rProd; ++r) {
            const float x = pre.GetValue(static_cast<uint32_t>(base + r * step));
            const float t = s + x;
            const float bb = t - s;
            const float err = (s - (t - bb)) + (x - bb);
            // 补偿项对有限输入是精确舍入误差；当输入含 ±Inf/NaN 时 two-sum 恒等式失真
            // （Inf-Inf=NaN），此时跳过误差累加，令 s（=普通顺序和）主导，使 ±Inf/NaN
            // 与 golden 的传播语义一致（NaN 不污染有限分组的补偿）。
            if (err == err) {
                c += err;
            }
            s = t;
        }
        cache.SetValue(static_cast<uint32_t>(levelOff + a), s + c);
    }
    // cacheBuf_ 随后由 DoCachingVf / PostElewise（VF）向量 pipe 读取 → 标量写须对向量可见。
    SetFlag<HardEvent::S_V>(evStoV_);
    WaitFlag<HardEvent::S_V>(evStoV_);
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
{
    if (rLen >= td_->rUbFactor) {
        return;
    }
    if (isTailR_) {
        const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = Ops::Base::CeilAlign(rLenInner, UB_BLOCK_F32);
        const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = Ops::Base::CeilDiv(extLanes, REP_F32);
        asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, static_cast<uint16_t>(aBundleEntries),
                                              static_cast<uint16_t>(repPerA));
    } else {
        const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->innerRProdAlign);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = Ops::Base::CeilDiv(totalClear, REP_F32);
        asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
    }
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
{
    const uint32_t bsInput = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT));
    const int32_t lastR = LastRAxis();
    const uint32_t validR = (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) :
                                                        static_cast<uint32_t>(td_->axisShape[lastR]);
    if (validR % bsInput == 0) {
        return;
    }
    const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                   Ops::Base::CeilAlign(static_cast<uint32_t>(td_->axisShape[lastR]), bsInput);
    const uint32_t rowCnt = (td_->rSplitIdx == lastR) ?
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign) /
                                    rowStride;
    const uint32_t padEndInRow = Ops::Base::CeilAlign(validR, bsInput);
    const uint32_t partialBlockIdx = validR / UB_BLOCK_F32;
    const uint32_t partialStartInBlock = validR % UB_BLOCK_F32;
    const uint32_t padEnd = padEndInRow - partialBlockIdx * UB_BLOCK_F32;
    asc_vf_call<ClearInnerBurstTailPadVfImpl>(base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
                                              static_cast<int32_t>(partialBlockIdx * UB_BLOCK_F32), padEnd,
                                              partialStartInBlock);
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::MergeTmpBufVf(__ubuf__ float* mainBuf,
                                                                        __ubuf__ float* tailBuf)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(
        Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<MergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));
    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    asc_vf_call<DoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::PostElewise(int64_t aLen)
{
    (void)aLen;
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;
    __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(outBuf_.Get<float>().GetPhyAddr());
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));
    const float outScale = (processIdx_ == 0) ? SUM_ACC_UNSCALE : 1.0f;
    asc_vf_call<PostElewiseVfImpl>(rootPtr, outPtr, laneN, repeatTime, outScale);

    SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
    WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);
}

template <typename DType>
__aicore__ inline void GNTrainingReduceBaseKernel<DType>::CopyOut(int64_t outerOutOff, int64_t aLen, int32_t processIdx)
{
    int64_t innerAProd = 1;
    for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= LastAAxis(); k += AXIS_INTERVAL) {
        innerAProd *= td_->axisShape[k];
    }

    DataCopyExtParams outParams;
    if (isTailR_) {
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = td_->axisShape[lastA];
        if (td_->aSplitIdx == lastA) {
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            outParams.blockCount = 1;
        } else {
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
        }
    }
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    outParams.rsv = 0;
    if (processIdx == 0) {
        DataCopyPad(sumGm_[outerOutOff], outBuf_.Get<float>(), outParams);
    } else {
        DataCopyPad(squareSumGm_[outerOutOff], outBuf_.Get<float>(), outParams);
    }
}

} // namespace NsGNTrainingReduce

#endif // GN_TRAINING_REDUCE_BASE_H
