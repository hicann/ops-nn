/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_dx_split_d.h
 * \brief L2NormalizeGrad DX split-D kernel (TilingKey 7010).
 *
 * Applies when inner == 1 but D (reduced-axis length) exceeds UB. The row is streamed in chunks
 * of tilingData.ubFactorElems (host 反推). Pass 1 (FormerProcess) reduces each chunk (sum(x*x), sum(y*dy)) into a
 * per-chunk accumulator buffer, then reduces the accumulators to per-row scalars sq/s. Pass 2
 * (LatterProcess) re-streams the row and writes dx = (dy - y*s) / max(sqrt(sq), eps).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
using namespace AscendC;

constexpr uint32_t SPLIT_D_MAX_CHUNKS = 256; // 累加槽数; chunk 数超过它时按组累加(D 无上限)

// ---------------------------------------------------------------------------
// s = sum(y*dy) 的补偿求和(double-float)
//
// 为什么需要: 本模板把整条归约轴汇聚到 1 个核的 VL 条车道上, 车道部分和的量级会涨到
// 远大于最终 s 的水平(实测 (256,255) dim=[1,0] 一例: 车道和 ~289, s=34), 之后每一次
// 无补偿合并都按车道和的量级舍入 —— 折算到 s 上就是数 ULP。实测该例 NPU 的 s 偏 4.67 ULP,
// 而同数据 fp32 竞品只偏 0.69 ULP; 对消点(dy ≈ y*s)上这点差异被放大成 1e-9 的绝对误差,
// 小值域判据直接判红。纯树形归约(DichotomyAdd)只能到 ~2.8 ULP, 不够。
//
// 只对 s 做: sq = sum(x*x) 全为正、条件数为 1, 无对消, 不需要补偿。
// ---------------------------------------------------------------------------

constexpr uint32_t DF_LO_OFFSET = 2U * static_cast<uint32_t>(V_LENGTH); // (hi,lo) 暂存槽里 lo 的起点
constexpr uint16_t HALF_DIV = 2U;
constexpr uint16_t DF_MIN_VEC_STRIDE = 8U; // UB 向量读需 32B 对齐 => stride 最小 8 个 fp32
constexpr float MAX_FINITE_FP32 = 3.402823466e+38F;

// TwoSum: a + b 的**精确**二元表示 —— s 为舍入和, e 为舍入残差, s + e == a + b。
// 只含加减、不含乘法, 因此不受 FMA 收缩影响。
__aicore__ inline void TwoSumVec(RegTensor<float>& sumReg, RegTensor<float>& errReg, RegTensor<float>& aReg,
                                 RegTensor<float>& bReg, MaskReg& mask)
{
    RegTensor<float> zReg, t1Reg, t2Reg, absReg, zeroReg;
    MaskReg finiteMask;
    Add(sumReg, aReg, bReg, mask);
    Sub(zReg, sumReg, aReg, mask);  // z  = s - a
    Sub(t1Reg, sumReg, zReg, mask); // t1 = s - z
    Sub(t1Reg, aReg, t1Reg, mask);  // t1 = a - (s - z)
    Sub(t2Reg, bReg, zReg, mask);   // t2 = b - z
    Add(errReg, t1Reg, t2Reg, mask);
    // 和溢出成 inf 时残差是 inf - inf = NaN, 把它加回去会把本该是 inf 的结果污染成 NaN
    // (输入本身含 inf/nan 时同理)。残差非有限即置 0 —— 与 in_training_update_grad_gamma_beta
    // 的 AddWithResidual 同一处理。补偿求和的意义只在有限域, 非有限值按 IEEE 自然传播即可。
    Abs(absReg, errReg, mask);
    Compares<float, CMPMODE::LE>(finiteMask, absReg, MAX_FINITE_FP32, mask);
    Duplicate(zeroReg, 0.0F, mask);
    Select(errReg, errReg, zeroReg, finiteMask);
}

// (hi, lo) += v
__aicore__ inline void DfAccumVec(RegTensor<float>& hiReg, RegTensor<float>& loReg, RegTensor<float>& vReg,
                                  MaskReg& mask)
{
    RegTensor<float> sReg, eReg;
    TwoSumVec(sReg, eReg, hiReg, vReg, mask);
    Add(eReg, eReg, loReg, mask);
    TwoSumVec(hiReg, loReg, sReg, eReg, mask);
}

// (hi, lo) += (hi2, lo2)
__aicore__ inline void DfAccumPairVec(RegTensor<float>& hiReg, RegTensor<float>& loReg, RegTensor<float>& hi2Reg,
                                      RegTensor<float>& lo2Reg, MaskReg& mask)
{
    RegTensor<float> sReg, eReg;
    TwoSumVec(sReg, eReg, hiReg, hi2Reg, mask);
    Add(eReg, eReg, loReg, mask);
    Add(eReg, eReg, lo2Reg, mask);
    TwoSumVec(hiReg, loReg, sReg, eReg, mask);
}

// 标量 double-float 累加: (hi, lo) += (h, l)。跨 chunk 合并用, 同样只含加减。
__aicore__ inline void DfAccumScalar(float& hi, float& lo, float h, float l)
{
    float sVal = hi + h;
    float zVal = sVal - hi;
    float eVal = (hi - (sVal - zVal)) + (h - zVal);
    eVal = eVal + lo + l;
    // 与向量版同理: 残差非有限时丢弃, 让非有限值按 IEEE 自然传播。
    // (用区间自比较同时判掉 NaN 与 ±inf。)
    // **两处都要判**: 只清 eVal 不够 —— 和溢出成 inf 后 lo = eVal - (hNew - sVal)
    // 里的 (inf - inf) 仍会产出 NaN, 最终 hi + lo 又变回 NaN。向量版由 TwoSumVec
    // 对其 err 输出统一兜底(第二次调用的 err 即是 lo), 标量版必须显式再判一次。
    if (!(eVal > -MAX_FINITE_FP32 && eVal < MAX_FINITE_FP32)) {
        eVal = 0.0f;
    }
    float hNew = sVal + eVal;
    float loNew = eVal - (hNew - sVal);
    if (!(loNew > -MAX_FINITE_FP32 && loNew < MAX_FINITE_FP32)) {
        loNew = 0.0f;
    }
    lo = loNew;
    hi = hNew;
}

template <typename T_X>
class RegbaseDxSplitD : public RegbaseDxBase<T_X> {
    using Base = RegbaseDxBase<T_X>;
    using Base::blockFactor_;
    using Base::CopyIn2D;
    using Base::CopyOut2D;
    using Base::coreIdx_;
    using Base::dxGm_;
    using Base::dyGm_;
    using Base::eps_;
    using Base::InitCommon;
    using Base::InitQueues;
    using Base::Ppipe_;
    using Base::tiling_;
    using Base::usedCoreNum_;
    using Base::xGm_;
    using Base::yGm_;

public:
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const L2NormalizeGradTilingData* tilingData) : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        rows_ = tiling_->outer;
        cols_ = tiling_->dimLen; // D
        if (!InitCommon(x, y, dy, dx, cols_)) {
            return; // 本核无任务
        }
        ubFactorD_ = tiling_->ubFactorElems; // 2VL 对齐的分块长度,由 host 从 ubSize 反推下发
        numChunks_ = tiling_->numChunks;     // host 下发,内核不再 DivCeil

        // UB 尺寸一律透传 host 下发值,内核不自行推算(见 tiling_data 注释)
        InitQueues(inQueueX_, inQueueY_, inQueueDy_, outQueueDx_);
        Ppipe_->InitBuffer(reduceBufSq_, tiling_->reduceBufBytes);
        Ppipe_->InitBuffer(accumBufSq_, tiling_->accumBufBytes);
        Ppipe_->InitBuffer(accumBufS_, tiling_->accumBufBytes);
        Ppipe_->InitBuffer(tmpSumSqBuf_, tiling_->tmpBufBytes);
        Ppipe_->InitBuffer(tmpSumSBuf_, tiling_->tmpBufBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx++) {
            FormerProcess(rowIdx);
            LatterProcess(rowIdx);
        }
    }

    // Reduce the whole row into per-row scalars sq (sum x*x) and s (sum y*dy).
    __aicore__ inline void FormerProcess(int64_t rowIdx)
    {
        LocalTensor<float> accumSqLocal = accumBufSq_.Get<float>();
        LocalTensor<float> accumSLocal = accumBufS_.Get<float>();
        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();
        // accum 缓冲固定 SPLIT_D_MAX_CHUNKS(=256) 个槽。chunk 数超过它时必须**按组**累加
        // (组内向量规约 + 组间标量累加), 不能按 numChunks_ 直接 Duplicate/写槽:
        // issue #31 的 1 维 D=1.51e8 需要约 3.7 万个槽, 原实现直接 Duplicate(accum, 0, 36928) 就把
        // 256 槽的缓冲写穿 → errcode 341(VEC 访问 UB 越界)。分组后 D 不再有上限。
        const int64_t maxSlots = static_cast<int64_t>(SPLIT_D_MAX_CHUNKS);
        float totalSq = 0.0f;
        totalSHi_ = 0.0f;
        totalSLo_ = 0.0f;
        int64_t colIdx = 0;
        while (colIdx < cols_) {
            int64_t remainChunks = DivCeil(cols_ - colIdx, ubFactorD_);
            int64_t groupSlots = Min(remainChunks, maxSlots);
            // 不清零、不对齐: 归约范围直接取真实槽数 groupSlots,ReduceSum<AR> 对非 32B 对齐的
            // 末轴走 ReduceARReuseSourceUnAligned 分支,pad 槽根本不进入归约,自然无需 Duplicate。
            for (int64_t slot = 0; slot < groupSlots; slot++, colIdx += ubFactorD_) {
                int64_t cnt = Min(ubFactorD_, cols_ - colIdx);
                ReduceChunk(rowIdx, colIdx, cnt, accumSqLocal, accumSLocal, slot);
                DfAccumScalar(totalSHi_, totalSLo_, chunkHi_, chunkLo_);
            }
            uint32_t accShape[2] = {1U, static_cast<uint32_t>(groupSlots)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSqLocal, accumSqLocal, accShape, false);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            totalSq += tmpSumSqLocal.GetValue(0);
            // s 的跨块合并在 ReduceChunk 里已按 double-float 逐块累进到 (totalSHi_, totalSLo_),
            // 这里不再有 s 的组级 ReduceSum。
        }
        // 组间总和写回 slot0, 供 LatterProcess 以 DIST_BRC_B32 广播读取。
        // s 到这里才把补偿残差并回主值——全程只此一次舍入。
        tmpSumSqLocal.SetValue(0, totalSq);
        tmpSumSLocal.SetValue(0, totalSHi_ + totalSLo_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    // Load one chunk, compute sum(x*x) and sum(y*dy) over it, store into accum[chunkIdx].
    __aicore__ inline void ReduceChunk(int64_t rowIdx, int64_t colIdx, int64_t cnt, LocalTensor<float>& accumSqLocal,
                                       LocalTensor<float>& accumSLocal, int64_t chunkIdx)
    {
        CopyIn(inQueueX_, xGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyIn(inQueueY_, yGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyIn(inQueueDy_, dyGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();

        LocalTensor<float> reduceSqLocal = reduceBufSq_.Get<float>();
        // 1VL 对齐 => VF 循环恰好铺满 [0, cntAlignVL); Mul 的 ZEROING 已把末轮多余 lane 置 0,
        // 全掩码整 VL 写出即完成尾部清零,无需 Duplicate。
        int64_t cntAlignVL = AlignUp(cnt, static_cast<int64_t>(V_LENGTH));

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cnt, static_cast<int64_t>(oneRepeat)));
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ float* reduceSqAddr = (__ubuf__ float*)reduceSqLocal.GetPhyAddr();
        // accumSLocal 原是 s 的 chunk 槽位数组; 改走 double-float 后不再需要槽位,
        // 它整块转作 (hi,lo) 暂存(host 已按 4 个 VL 保底其尺寸)。
        __local_mem__ float* dfAddr = (__ubuf__ float*)accumSLocal.GetPhyAddr();
        // s 路径不再把逐元素积落 UB 再 ReduceSum: 那条路是"车道内顺序累加 + 车道间合并",
        // 车道和量级远大于 s, 合并时的舍入折算到 s 上有数 ULP。改为车道级 double-float 累加,
        // 残差随 (hi, lo) 一路带到最后, 只在最终落 fp32 时舍一次。
        // dfAddr 复用为 4 个 VL 的暂存槽: [0,VL)=hi, [VL,2VL)=0, [2VL,3VL)=lo, [3VL,4VL)=0。
        // 两块补零槽是给横向树用的: 取 +stride 偏移读时越过 VL 末尾的部分必须读到 0。
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, sqReg, sReg, accHiReg, accLoReg, zeroReg;
            MaskReg fullMask = CreateMask<float>(); // MaskPattern::ALL, 整 VL 写出
            uint32_t sreg = static_cast<uint32_t>(cnt);
            MaskReg maskReg;
            Duplicate(accHiReg, 0.0f, fullMask);
            Duplicate(accLoReg, 0.0f, fullMask);
            Duplicate(zeroReg, 0.0f, fullMask);
            for (uint16_t i = 0; i < repeatCount; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(sqReg, xReg, xReg, maskReg);
                DataCopy(reduceSqAddr + static_cast<uint32_t>(i * oneRepeat), sqReg, fullMask);
                LoadAndCast(yReg, yAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                // Mul 的 ZEROING 已把末轮越界 lane 置 0, 累加它们是恒等操作, 无需额外掩码。
                Mul(sReg, yReg, dyReg, maskReg);
                DfAccumVec(accHiReg, accLoReg, sReg, fullMask);
            }
            DataCopy(dfAddr, accHiReg, fullMask);
            DataCopy(dfAddr + V_LENGTH, zeroReg, fullMask);
            DataCopy(dfAddr + DF_LO_OFFSET, accLoReg, fullMask);
            DataCopy(dfAddr + DF_LO_OFFSET + V_LENGTH, zeroReg, fullMask);
        }
        // 车道间 double-float 树形合并: stride 逐级折半, 折到 DF_MIN_VEC_STRIDE 为止。
        // 不再往下折是因为 UB 的向量读要求 32B 对齐, stride < 8 时偏移只有 16B/8B/4B。
        // 折完 lane j (j < DF_MIN_VEC_STRIDE) 持有 {j, j+8, j+16, ...} 之和, 余下 8 条车道
        // 由标量 double-float 收尾——只有 8 次, 代价可忽略。
        __VEC_SCOPE__
        {
            RegTensor<float> hiReg, loReg, hi2Reg, lo2Reg;
            MaskReg fullMask = CreateMask<float>();
            for (uint16_t stride = V_LENGTH / HALF_DIV; stride >= DF_MIN_VEC_STRIDE; stride >>= 1U) {
                LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                DataCopy(hiReg, dfAddr);
                DataCopy(hi2Reg, dfAddr + stride);
                DataCopy(loReg, dfAddr + DF_LO_OFFSET);
                DataCopy(lo2Reg, dfAddr + DF_LO_OFFSET + stride);
                DfAccumPairVec(hiReg, loReg, hi2Reg, lo2Reg, fullMask);
                DataCopy(dfAddr, hiReg, fullMask);
                DataCopy(dfAddr + DF_LO_OFFSET, loReg, fullMask);
            }
        }
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);

        uint32_t chunkShape[2] = {1U, static_cast<uint32_t>(cntAlignVL)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(accumSqLocal[chunkIdx], reduceSqLocal, chunkShape,
                                                                      false);
        // s 的本块结果已是 (hi, lo) 对, 放在暂存槽 lane 0; 交给调用方按
        // double-float 跨块累加(走槽位数组再 ReduceSum 会把残差丢掉, 前功尽弃)。
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        chunkHi_ = 0.0f;
        chunkLo_ = 0.0f;
        for (uint16_t lane = 0; lane < DF_MIN_VEC_STRIDE; lane++) {
            DfAccumScalar(chunkHi_, chunkLo_, accumSLocal.GetValue(lane), accumSLocal.GetValue(DF_LO_OFFSET + lane));
        }
        // 标量读完才允许下一块的向量写覆盖暂存槽(WAR)。
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    // Re-stream the row and write dx = (dy - y*s) / max(sqrt(sq), eps).
    __aicore__ inline void LatterProcess(int64_t rowIdx)
    {
        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();
        __local_mem__ float* sqSumAddr = (__ubuf__ float*)tmpSumSqLocal.GetPhyAddr();
        __local_mem__ float* sSumAddr = (__ubuf__ float*)tmpSumSLocal.GetPhyAddr();

        for (int64_t colIdx = 0; colIdx < cols_; colIdx += ubFactorD_) {
            int64_t cnt = Min(ubFactorD_, cols_ - colIdx);
            CopyIn(inQueueY_, yGm_, rowIdx, colIdx, cnt);
            LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
            CopyIn(inQueueDy_, dyGm_, rowIdx, colIdx, cnt);
            LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
            LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();

            constexpr uint32_t oneRepeat = V_LENGTH;
            uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cnt, static_cast<int64_t>(oneRepeat)));
            __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
            __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
            __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
            __VEC_SCOPE__
            {
                RegTensor<float> yReg, dyReg, sqReg, sReg, nReg, ysReg, subReg, dxReg;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(sqReg, sqSumAddr);
                DataCopy<float, LoadDist::DIST_BRC_B32>(sReg, sSumAddr);
                Sqrt(nReg, sqReg, maskAll);
                Maxs(nReg, nReg, eps_, maskAll);
                uint32_t sreg = static_cast<uint32_t>(cnt);
                MaskReg maskReg;
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(yReg, yAddr, maskReg, i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                    Mul(ysReg, yReg, sReg, maskReg);
                    Sub(subReg, dyReg, ysReg, maskReg);
                    Div(dxReg, subReg, nReg, maskReg);
                    StoreDx<T_X>(dxAddr, static_cast<uint32_t>(i * oneRepeat), dxReg, maskReg);
                }
            }
            inQueueY_.FreeTensor(yLocal);
            inQueueDy_.FreeTensor(dyLocal);
            outQueueDx_.EnQue(dxLocal);
            CopyOutDx(rowIdx, colIdx, cnt);
        }
    }

    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t rowIdx,
                                  int64_t colIdx, int64_t cnt)
    {
        CopyIn2D(que, gm, rowIdx * cols_ + colIdx, 1, cnt, 0); // 单行分块,GM 上连续
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t colIdx, int64_t cnt)
    {
        CopyOut2D(outQueueDx_, rowIdx * cols_ + colIdx, 1, cnt, 0);
    }

private:
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueY_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;

    TBuf<TPosition::VECCALC> reduceBufSq_;
    TBuf<TPosition::VECCALC> accumBufSq_;
    TBuf<TPosition::VECCALC> accumBufS_;
    TBuf<TPosition::VECCALC> tmpSumSqBuf_;
    TBuf<TPosition::VECCALC> tmpSumSBuf_;

    // s 的补偿求和状态: chunkHi_/chunkLo_ 是 ReduceChunk 的本块结果,
    // totalSHi_/totalSLo_ 是跨块累进值(见 DfAccumScalar)。
    float chunkHi_ = 0.0f;
    float chunkLo_ = 0.0f;
    float totalSHi_ = 0.0f;
    float totalSLo_ = 0.0f;

    int64_t rows_;
    int64_t cols_;
    int64_t ubFactorD_;
    int64_t numChunks_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
