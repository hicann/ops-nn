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
 * \file cross_entropy_sum_exp_and_index_logit.h
 * \brief A5 (ascend950) kernel — VF（Vector Function / __simd_vf__ + Reg RegTensor）编程范式。
 *
 *   语义（vocab 并行 CrossEntropy 本地计算融合算子，all_reduce(SUM) 之前）：
 *     target_mask[i]   = (target[i] < vocabStart || target[i] >= vocabEnd) ? 1 : 0
 *     target_offset[i] = mask ? 0 : target[i] - vocabStart
 *     predicted[i]     = mask ? 0 : logits[i, offset[i]] - global_max[i]
 *     exp_logits[i,j]  = exp(logits[i,j] - global_max[i])
 *     sum_exp[i]       = sum_j exp_logits[i,j]（块算多级降维 + 跨 tile Kahan，确定性）
 *
 *   TilingKey 单默认调度模式（CE_SCH_MODE_DEFAULT），入口经 DTYPE 宏实例化 float / bfloat16_t（if constexpr 裁剪）。
 */
#ifndef CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_H_
#define CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cross_entropy_sum_exp_and_index_logit_struct.h"
#include "cross_entropy_sum_exp_and_index_logit_common.h"
#include "vf/compute.h"

namespace CrossEntropySumExpAndIndexLogit {
using namespace AscendC;

// ===== Kernel 主类 =====
template <typename T>
class KernelCrossEntropyRegbase {
public:
    __aicore__ inline void Init(GM_ADDR vocabParallelLogits, GM_ADDR target, GM_ADDR globalLogitsMax,
                                GM_ADDR predictedLogits, GM_ADDR sumExpLogits, GM_ADDR expLogits, GM_ADDR targetOffset,
                                GM_ADDR targetMask, const CrossEntropySumExpAndIndexLogitRegBaseTilingData* t)
    {
        vLocal_ = t->vLocal;
        vTile_ = t->vTile;
        vLoopNum_ = t->vLoopNum;
        lastVTile_ = t->lastVTile;
        vocabStart_ = t->vocabStart;
        vocabEnd_ = t->vocabEnd;
        rowBlockMax_ = t->rowBlockMax;
        reduceTmpBytes_ = t->reduceTmpBytes;

        // 两级均衡切分：核间 floor+remainder 均分 token，每核独立内循环参数
        //   头核(blockIdx < headCoreNum): tokensPerCore, blockNum=headBlockNum
        //   尾核(blockIdx >= headCoreNum): tokensPerCoreTail, blockNum=tailBlockNum
        int64_t blockIdx = GetBlockIdx();
        int64_t headCoreNum = t->headCoreNum;
        if (blockIdx < headCoreNum) {
            tokensThisCore_ = t->tokensPerCore;
            blockNum_ = t->headBlockNum;
            baseToken_ = blockIdx * tokensThisCore_;
        } else {
            tokensThisCore_ = t->tokensPerCoreTail;
            blockNum_ = t->tailBlockNum;
            baseToken_ = headCoreNum * t->tokensPerCore + (blockIdx - headCoreNum) * tokensThisCore_;
        }
        blockBase_ = (blockNum_ > 0) ? (tokensThisCore_ / blockNum_) : 0;
        blockRem_ = (blockNum_ > 0) ? (tokensThisCore_ - blockBase_ * blockNum_) : 0;

        logitsGm_.SetGlobalBuffer((__gm__ T*)vocabParallelLogits);    // [N, vLocal]
        targetGm_.SetGlobalBuffer((__gm__ int32_t*)target);           // [N]
        globalMaxGm_.SetGlobalBuffer((__gm__ T*)globalLogitsMax);     // [N]
        predictedGm_.SetGlobalBuffer((__gm__ float*)predictedLogits); // [N]
        sumExpGm_.SetGlobalBuffer((__gm__ float*)sumExpLogits);       // [N]
        expGm_.SetGlobalBuffer((__gm__ float*)expLogits);             // [N, vLocal]
        offsetGm_.SetGlobalBuffer((__gm__ int32_t*)targetOffset);     // [N]
        maskGm_.SetGlobalBuffer((__gm__ int32_t*)targetMask);         // [N]

        // 双缓冲 Queue：logitsIn(T) 搬入 + expOut(FP32) 搬出，均按 rowBlockMax 分配
        pipe_.InitBuffer(inQue_, BUFFER_NUM, rowBlockMax_ * vTile_ * sizeof(T));
        pipe_.InitBuffer(expOutQue_, BUFFER_NUM, rowBlockMax_ * vTile_ * sizeof(float));
        // 常驻 buffer：标量 GetValue 用 AlignUp32；VF 读写按 AlignUp256 对齐分配，
        //   targetBuf 被 VF1 LoadAlign 无 mask 按 256B 读，offset/mask 被 StoreAlign 带 mask 写 curN 个，
        //   均需 256B 对齐容量防越界读写相邻 UB buffer。
        int64_t alignRowFp32 = AlignUp32<float>(rowBlockMax_) * sizeof(float);
        int64_t vecRowI32 = AlignUp256<int32_t>(rowBlockMax_) * sizeof(int32_t);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            pipe_.InitBuffer(globalMaxInBuf_, AlignUp32<bfloat16_t>(rowBlockMax_) * sizeof(bfloat16_t));
        }
        pipe_.InitBuffer(globalMaxBuf_, alignRowFp32);
        pipe_.InitBuffer(targetBuf_, vecRowI32);
        pipe_.InitBuffer(offsetBuf_, vecRowI32);
        pipe_.InitBuffer(maskBuf_, vecRowI32);
        pipe_.InitBuffer(sumExpAccBuf_, alignRowFp32);
        pipe_.InitBuffer(sumExpCompBuf_, alignRowFp32);
        pipe_.InitBuffer(kahanYBuf_, alignRowFp32);
        pipe_.InitBuffer(tileSumBuf_, alignRowFp32);
        pipe_.InitBuffer(predictedBuf_, alignRowFp32);
        // ReduceSum(AR, float) sharedTmpBuffer / 块算多级降维中间结果：字节数由 host 下发
        pipe_.InitBuffer(reduceTmpBuf_, reduceTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        // 均衡内循环：blockBase_=tokensThisCore_/blockNum_, blockRem_=tokensThisCore_-blockBase_*blockNum_
        //   前 blockRem_ 块各 blockBase_+1 行，其余 blockBase_ 行；baseRow 迭代累加
        int64_t baseRow = baseToken_;
        for (int64_t blk = 0; blk < blockNum_; ++blk) {
            int64_t curN = blockBase_ + (blk < blockRem_ ? 1 : 0);
            LoadBlockScalars(baseRow, curN);  // DataCopyPad: target/globalMax → UB
            ComputeMaskOffset(baseRow, curN); // VF1: int32 Compare + Select
            ComputePredicted(baseRow, curN);  // 标量 GM gather（无法向量化）
            ComputeExpSum(baseRow, curN);     // VF3 exp + 块算多级降维 + 跨 tile Kahan
            baseRow += curN;
        }
    }

private:
    // 搬入本行块的 global_max（→FP32 UB）与 target（INT32 UB），供后续标量 GetValue
    __aicore__ inline void LoadBlockScalars(int64_t baseRow, int64_t curN)
    {
        LocalTensor<float> gmax = globalMaxBuf_.template Get<float>();
        LocalTensor<int32_t> tgt = targetBuf_.template Get<int32_t>();
        DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        DataCopyExtParams pI;
        pI.blockCount = NUM_ONE;
        pI.blockLen = static_cast<uint32_t>(curN * sizeof(int32_t));
        pI.srcStride = 0;
        pI.dstStride = 0;
        DataCopyPad(tgt, targetGm_[baseRow], pI, padI); // target[curN] → UB
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<T> gmaxIn = globalMaxInBuf_.template Get<T>();
            DataCopyPadExtParams<T> padD{false, 0, 0, 0};
            DataCopyExtParams pD;
            pD.blockCount = NUM_ONE;
            pD.blockLen = static_cast<uint32_t>(curN * sizeof(T));
            pD.srcStride = 0;
            pD.dstStride = 0;
            DataCopyPad(gmaxIn, globalMaxGm_[baseRow], pD, padD); // global_max(BF16) → UB
            SetWaitFlag<HardEvent::MTE2_V>();
            Cast(gmax, gmaxIn, RoundMode::CAST_NONE, AlignUp32<float>(curN)); // BF16→FP32
            SetWaitFlag<HardEvent::V_S>();                                    // gmax 供标量读
        } else {
            DataCopyPadExtParams<float> padD{false, 0, 0, 0};
            DataCopyExtParams pD;
            pD.blockCount = NUM_ONE;
            pD.blockLen = static_cast<uint32_t>(curN * sizeof(float));
            pD.srcStride = 0;
            pD.dstStride = 0;
            DataCopyPad(gmax, globalMaxGm_[baseRow], pD, padD); // global_max(FP32) → UB
        }
        SetWaitFlag<HardEvent::MTE2_S>(); // gmax 供标量读
        SetWaitFlag<HardEvent::MTE2_V>(); // target 供向量读（VF1 LoadAlign）
    }

    // mask = (target < vocabStart || target >= vocabEnd) ? 1 : 0
    // offset = mask ? 0 : target - vocabStart
    // VF1：A5 int32 Compare 直通（无需 Cast float），两次顺序 Select 合并 mask
    __aicore__ inline void ComputeMaskOffset(int64_t baseRow, int64_t curN)
    {
        LocalTensor<int32_t> tgt = targetBuf_.template Get<int32_t>();
        LocalTensor<int32_t> offset = offsetBuf_.template Get<int32_t>();
        LocalTensor<int32_t> msk = maskBuf_.template Get<int32_t>();

        uint16_t repeatTimes = static_cast<uint16_t>(
            CeilDiv(curN, static_cast<uint32_t>(REPEAT_SIZE / sizeof(int32_t))));

        // 调用 VF1（0/1 常量 VF 内 Duplicate 生成，不占 UB）
        asc_vf_call<MaskOffsetVF<T>>((__ubuf__ int32_t*)tgt.GetPhyAddr(), (__ubuf__ int32_t*)offset.GetPhyAddr(),
                                     (__ubuf__ int32_t*)msk.GetPhyAddr(), vocabStart_, vocabEnd_, curN, repeatTimes);

        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = NUM_ONE;
        copyParams.blockLen = static_cast<uint32_t>(curN * sizeof(int32_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(maskGm_[baseRow], msk, copyParams);
        DataCopyPad(offsetGm_[baseRow], offset, copyParams);
        SetWaitFlag<HardEvent::MTE3_S>(); // mask/offset 保留 UB 供 ComputePredicted
    }

    // 从 GM 原始 logits gather predicted + 减 global_max
    // target_offset 为 V_local 全局列偏移，UB 上 shifted 只驻留局部无法索引，故必须从 GM 读
    __aicore__ inline void ComputePredicted(int64_t baseRow, int64_t curN)
    {
        LocalTensor<float> predicted = predictedBuf_.template Get<float>();
        LocalTensor<int32_t> msk = maskBuf_.template Get<int32_t>();
        LocalTensor<int32_t> off = offsetBuf_.template Get<int32_t>();
        LocalTensor<float> gmax = globalMaxBuf_.template Get<float>();
        Duplicate(predicted, static_cast<float>(0.0f), AlignUp32<float>(rowBlockMax_)); // mask==1 行保持 0
        SetWaitFlag<HardEvent::V_S>(); // Duplicate(V) 完成后再 SetValue(S)，同时覆盖 VF1 StoreAlign
        for (int64_t i = 0; i < curN; ++i) {
            if (msk.GetValue(i) == 0) {
                int64_t o = static_cast<int64_t>(off.GetValue(i));       // 0 ~ vLocal-1
                T raw = logitsGm_.GetValue((baseRow + i) * vLocal_ + o); // GM 标量读
                // BF16→FP32：bfloat16_t 标量与 float 互转必须走 AscendC::Cast intrinsic，
                // 禁止 static_cast（device bisheng backend 无内置 bf16 隐式转换）
                float logit;
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    logit = ToFloat(raw); // 兼容 CANN8.5.0，新接口名 Cast（多类型转 float）
                } else {
                    logit = raw;
                }
                predicted.SetValue(i, logit - gmax.GetValue(i)); // 减 global_max
            }
        }
        SetWaitFlag<HardEvent::S_MTE3>();
        DataCopyExtParams vecParams;
        vecParams.blockCount = NUM_ONE;
        vecParams.blockLen = static_cast<uint32_t>(curN * sizeof(float));
        vecParams.srcStride = 0;
        vecParams.dstStride = 0;
        DataCopyPad(predictedGm_[baseRow], predicted, vecParams);
        SetWaitFlag<HardEvent::MTE3_V>();
    }

    // 块算快路径安全判据：BlockReduceSum 整块紧凑路径 totalRep=curN*len/64 要求"降维链每级 len
    //   都是 64 的倍数"直到 ≤64，否则跨行截断丢每行尾部（len%64 个元素）。
    //   逐级 ÷8 模拟：curV → curV/8 → ... 检查每个 >64 的 len 是否 %64==0。
    //   典型：2048(512×4)→256→32 全对齐✓；lastVTile 如 1408→176(176%64≠0)✗ 走 fallback。
    __aicore__ inline bool IsBlockReduceSafe(int64_t curV)
    {
        // FP32_REPEAT_ELEM=64=2^6，取余判断用位与替代，避免 A5 标量取余指令
        if ((curV & (FP32_REPEAT_ELEM - NUM_ONE)) != NUM_ZERO) {
            return false;
        } // 首级即非 64 倍数
        int64_t len = curV;
        while (len > FP32_REPEAT_ELEM) {
            if ((len & (FP32_REPEAT_ELEM - NUM_ONE)) != NUM_ZERO) {
                return false;
            }
            len /= FP32_PER_BLOCK;
        }
        return true;
    }

    // 核内：沿 vTile 循环，VF3 计算 exp 落 RegTensor→StoreAlign 回 expOutQue（xout），
    //   规约在 UB 上块算多级降维，Kahan 跨 tile 累加。
    __aicore__ inline void ComputeExpSum(int64_t baseRow, int64_t curN)
    {
        LocalTensor<float> acc = sumExpAccBuf_.template Get<float>();
        LocalTensor<float> comp = sumExpCompBuf_.template Get<float>();
        LocalTensor<float> kahanY = kahanYBuf_.template Get<float>();
        LocalTensor<float> tileSum = tileSumBuf_.template Get<float>();
        LocalTensor<uint8_t> redTmp = reduceTmpBuf_.template Get<uint8_t>();
        LocalTensor<float> gmax = globalMaxBuf_.template Get<float>();
        Duplicate(acc, static_cast<float>(0.0f), AlignUp32<float>(rowBlockMax_));
        Duplicate(comp, static_cast<float>(0.0f), AlignUp32<float>(rowBlockMax_)); // Kahan 补偿项清零
        PipeBarrier<PIPE_V>();

        for (int64_t vt = 0; vt < vLoopNum_; ++vt) {
            int64_t vOff = vt * vTile_;
            int64_t curV = (vt == vLoopNum_ - NUM_ONE) ? lastVTile_ : vTile_;

            // CopyIn logits[curN, curV]（行优先，行间 stride = vLocal - curV）
            LocalTensor<T> xin = inQue_.template AllocTensor<T>();
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyExtParams inParams;
            inParams.blockCount = static_cast<uint16_t>(curN);
            inParams.blockLen = static_cast<uint32_t>(curV * sizeof(T));
            inParams.srcStride = static_cast<uint32_t>((vLocal_ - curV) * sizeof(T));
            inParams.dstStride = 0;
            DataCopyPad(xin, logitsGm_[baseRow * vLocal_ + vOff], inParams, padParams);
            inQue_.EnQue(xin);
            xin = inQue_.template DeQue<T>();

            LocalTensor<float> xout = expOutQue_.template AllocTensor<float>();
            // 逐行调用 VF3（每行独立减 global_max 标量），exp 结果 StoreAlign 到 xout
            for (int64_t row = 0; row < curN; ++row) {
                float gmaxScalar = gmax.GetValue(row);
                SetWaitFlag<HardEvent::S_V>();
                uint16_t rt = static_cast<uint16_t>(CeilDiv(curV, static_cast<uint32_t>(REPEAT_SIZE / sizeof(float))));
                asc_vf_call<ExpSumTileVF<T>>((__ubuf__ T*)(xin.GetPhyAddr() + row * curV * sizeof(T)),
                                             (__ubuf__ float*)(xout.GetPhyAddr() + row * curV * sizeof(float)),
                                             gmaxScalar, static_cast<uint32_t>(NUM_ONE), static_cast<uint32_t>(curV),
                                             rt);
            }

            // 行规约求和累加（固定顺序，确定性）：tileSum[curN] = sum_j xout[i, j]
            ReduceRowsBlockwise(tileSum, xout, redTmp, curN, curV);

            // 搬出 exp_logits：xout 已是 exp 结果，EnQue → CopyOut
            expOutQue_.EnQue(xout);
            xout = expOutQue_.template DeQue<float>();
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(curN);
            outParams.blockLen = static_cast<uint32_t>(curV * sizeof(float));
            outParams.srcStride = 0;
            outParams.dstStride = static_cast<uint32_t>((vLocal_ - curV) * sizeof(float));
            DataCopyPad(expGm_[baseRow * vLocal_ + vOff], xout, outParams);
            expOutQue_.FreeTensor(xout);
            inQue_.FreeTensor(xin);

            // Kahan 补偿求和（跨 tile 累加抵消"大数吃小数"，vLoopNum 个 tile 顺序累加时误差累积）：
            //   y = tileSum - comp;  t = acc + y;  comp = (t - acc) - y;  acc = t
            Sub(kahanY, tileSum, comp, curN); // y = tileSum - comp（找回上轮丢失的低位）
            PipeBarrier<PIPE_V>();
            Add(tileSum, acc, kahanY, curN); // t = acc + y（借 tileSum 存 t，后续不再用 tileSum）
            PipeBarrier<PIPE_V>();
            Sub(comp, tileSum, acc, curN); // comp = t - acc（本轮实际吸收的高位）
            PipeBarrier<PIPE_V>();
            Sub(comp, comp, kahanY, curN); // comp = (t - acc) - y（新的丢失低位，下轮补偿）
            Adds(acc, tileSum, static_cast<float>(0.0f), curN); // acc = t
            PipeBarrier<PIPE_V>();
        }

        // CopyOut sum_exp_logits[curN]
        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopyExtParams vecParams;
        vecParams.blockCount = NUM_ONE;
        vecParams.blockLen = static_cast<uint32_t>(curN * sizeof(float));
        vecParams.srcStride = 0;
        vecParams.dstStride = 0;
        DataCopyPad(sumExpGm_[baseRow], acc, vecParams);
        SetWaitFlag<HardEvent::MTE3_V>();
    }

    // 行规约 dst[curN] = sum_j src[i*curV+j]（与详设 ReduceRowsBlockwise 一致）
    __aicore__ inline void ReduceRowsBlockwise(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                               const LocalTensor<uint8_t>& tmp, int64_t curN, int64_t curV)
    {
        if (IsBlockReduceSafe(curV)) {
            LocalTensor<float> reduceTmpF = tmp.template ReinterpretCast<float>();
            int64_t len = curV;
            LocalTensor<float> redSrc = src;                 // 首级源 = xout（只读）
            int64_t srcRowStride = curV;                     // 首级行间步长 = curV
            while (len > FP32_REPEAT_ELEM) {                 // len>64：BlockReduceSum 降维一级 len→len/8
                int64_t dstRowStride = len / FP32_PER_BLOCK; // 本级输出行间步长（紧凑）
                int64_t totalRep = curN * len / FP32_REPEAT_ELEM;
                int64_t doneRep = 0;
                int64_t sOff = 0;
                int64_t dOff = 0;
                while (doneRep < totalRep) {
                    int32_t rep = static_cast<int32_t>((totalRep - doneRep) > MAX_REPEAT ? MAX_REPEAT :
                                                                                           (totalRep - doneRep));
                    // (dst, src, repeatTime, mask=64, dstRepStride=1, srcBlkStride=1, srcRepStride=8)
                    BlockReduceSum<float>(reduceTmpF[dOff], redSrc[sOff], rep, FP32_REPEAT_ELEM, NUM_ONE, NUM_ONE,
                                          FP32_PER_BLOCK);
                    doneRep += rep;
                    sOff += static_cast<int64_t>(rep) * FP32_REPEAT_ELEM;
                    dOff += static_cast<int64_t>(rep) * FP32_PER_BLOCK;
                }
                PipeBarrier<PIPE_V>();
                redSrc = reduceTmpF; // 后续级 tmp 内 in-place（紧凑）
                srcRowStride = dstRowStride;
                len = dstRowStride; // 内轴 /8
            }
            // len≤64：每行一个 repeat 归约成 1 → dst[curN]
            WholeReduceSum<float>(dst, redSrc, static_cast<int32_t>(len), static_cast<int32_t>(curN), NUM_ONE, NUM_ONE,
                                  static_cast<int32_t>(srcRowStride / FP32_PER_BLOCK));
            PipeBarrier<PIPE_V>();
        } else {
            // fallback：不安全 curV（如非规整 lastVTile），高阶 ReduceSum<AR> 整块批量归约
            uint32_t srcShape[2] = {static_cast<uint32_t>(curN), static_cast<uint32_t>(curV)};
            ReduceSum<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, srcShape, true); // isReuseSource=false
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe pipe_;

    GlobalTensor<T> logitsGm_, globalMaxGm_;
    GlobalTensor<int32_t> targetGm_, offsetGm_, maskGm_;
    GlobalTensor<float> predictedGm_, sumExpGm_, expGm_;

    // 双缓冲主数据路径
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;      // 搬入 logits tile
    TQue<QuePosition::VECOUT, BUFFER_NUM> expOutQue_; // 搬出 exp_logits
    // 常驻辅助 buffer（TBuf 静态单块）
    TBuf<TPosition::VECCALC> globalMaxInBuf_; // BF16 搬入 global_max（仅 BF16 使用）
    TBuf<TPosition::VECCALC> globalMaxBuf_;   // global_max（FP32），标量 GetValue
    TBuf<TPosition::VECCALC> targetBuf_;      // target（INT32），VF1 LoadAlign 读
    TBuf<TPosition::VECCALC> offsetBuf_;      // target_offset，VF1 StoreAlign 写
    TBuf<TPosition::VECCALC> maskBuf_;        // target_mask，VF1 StoreAlign 写
    TBuf<TPosition::VECCALC> sumExpAccBuf_;   // sum_exp 累加器
    TBuf<TPosition::VECCALC> sumExpCompBuf_;  // Kahan 补偿项 c
    TBuf<TPosition::VECCALC> kahanYBuf_;      // Kahan 临时 y
    TBuf<TPosition::VECCALC> tileSumBuf_;     // 每 tile 部分和（规约 dst）
    TBuf<TPosition::VECCALC> predictedBuf_;   // predicted
    TBuf<TPosition::VECCALC> reduceTmpBuf_;   // 块算 tmp / ReduceSum(AR) sharedTmpBuffer

    // Tiling 缓存
    int64_t vLocal_{0};
    int64_t vTile_{0};
    int64_t vLoopNum_{0};
    int64_t lastVTile_{0};
    int64_t vocabStart_{0};
    int64_t vocabEnd_{0};
    int64_t rowBlockMax_{0};    // rowBlock 上限（InitBuffer 用）
    int64_t reduceTmpBytes_{0}; // ReduceSum(AR) sharedTmpBuffer 字节（host 下发）
    int64_t blockNum_{0};       // 本核内循环块数
    int64_t blockBase_{0};      // 内循环基础块行数 = tokensThisCore/blockNum_
    int64_t blockRem_{0};       // 前 blockRem_ 块多 1 行 = tokensThisCore%blockNum_
    int64_t tokensThisCore_{0}; // 本核 token 数
    int64_t baseToken_{0};      // 本核起始 token 偏移
};

} // namespace CrossEntropySumExpAndIndexLogit

#endif // CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_H_
