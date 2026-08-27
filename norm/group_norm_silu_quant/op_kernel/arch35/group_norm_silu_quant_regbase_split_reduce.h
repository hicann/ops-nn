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
 * \file group_norm_silu_quant_regbase_split_reduce.h
 * \brief GroupNormSiluQuant arch35 调度: split-reduce(tilingKey 1140)。
 *        N*num_groups < 核数 且单组 reduce 很大(如 G=1)时,按 shapeD 通道把单个 group 切到 coresPerGroup 个核,
 *        两阶段跨核归约:Pass1 各核对自己那段通道求 sum/sumsq 写 workspace;SyncAll;各核读回本组全部 partial
 *        合成全局 mean/rstd;Pass2 各核对自己通道段做 normalize+silu+quant 落盘。与其它 tilingKey 完全独立。
 */

#ifndef GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_

#include "group_norm_silu_quant_regbase_base.h"

namespace GroupNormSiluQuant {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluQuantSplitReduce {
public:
    static constexpr int64_t TILE = 4096; // 每次搬运/计算的元素数

    __aicore__ inline GroupNormSiluQuantSplitReduce(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu, GM_ADDR mean,
                                GM_ADDR rstd, GM_ADDR workspace, const GroupNormSiluQuantRegbaseTilingData* tilingData)
    {
        tiling = tilingData;
        blockIdx = GetBlockIdx();
        numGroups = tiling->numGroups;
        hwNum = tiling->hwNum;
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        shapeQuantScale = tiling->shapeQuantScale;
        perChannelScale = (shapeQuantScale != 1);
        eps = tiling->epsilon;
        activateSilu = (tiling->activateSilu != 0);
        coresPerGroup = tiling->coresPerGroup;

        groupGlobalIdx = blockIdx / coresPerGroup;
        subIdx = blockIdx % coresPerGroup;
        int64_t grp = groupGlobalIdx % numGroups;
        int64_t nIdx = groupGlobalIdx / numGroups;

        chanPerCore = CeilDiv(shapeD, coresPerGroup);
        chanStart = subIdx * chanPerCore;
        if (chanStart >= shapeD) {
            myChan = 0;
        } else {
            myChan = (chanStart + chanPerCore <= shapeD) ? chanPerCore : (shapeD - chanStart);
        }
        myElem = myChan * hwNum;

        int64_t chanBaseC = grp * shapeD + chanStart;
        xBase = nIdx * shapeC * hwNum + chanBaseC * hwNum;
        gammaBase = chanBaseC;
        qsBase = perChannelScale ? chanBaseC : 0;

        InitGmTensors(x, gamma, beta, quantScale, silu, mean, rstd, workspace);
        InitUbBuffers();
    }

    __aicore__ inline void InitGmTensors(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu,
                                         GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace)
    {
        xGm.SetGlobalBuffer((__gm__ T1*)x);
        if (gamma != nullptr) {
            hasGamma = true;
            gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        if (beta != nullptr) {
            hasBeta = true;
            betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        quantScaleGm.SetGlobalBuffer((__gm__ float*)quantScale);
        siluGm.SetGlobalBuffer((__gm__ int8_t*)silu);
        if constexpr (sizeof(T2) == sizeof(float)) {
            meanT2Gm.SetGlobalBuffer((__gm__ T2*)mean);
            rstdT2Gm.SetGlobalBuffer((__gm__ T2*)rstd);
        } else {
            meanGm.SetGlobalBuffer((__gm__ T1*)mean);
            rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);
        }
        wsGm.SetGlobalBuffer((__gm__ float*)workspace);
    }

    __aicore__ inline void InitUbBuffers()
    {
        int64_t chanAlign = (chanPerCore < 8 ? 8 : chanPerCore);
        pipe.InitBuffer(inQue, 2, TILE * sizeof(T1));
        pipe.InitBuffer(outQue, 2, TILE * sizeof(int8_t));
        pipe.InitBuffer(fbuf, TILE * sizeof(float));
        pipe.InitBuffer(rbuf, TILE * sizeof(float));
        pipe.InitBuffer(wbuf, TILE * sizeof(float));
        pipe.InitBuffer(reduf, 32 * sizeof(float)); // acc: [0]=sum [8]=sumsq [16]=tmp, 需 >=24 floats
        pipe.InitBuffer(partBuf, BLOCK_SIZE);
        pipe.InitBuffer(combBuf, RoundUp<float>(coresPerGroup * 2) * sizeof(float) + BLOCK_SIZE);
        // gamma/beta 的通道常驻缓冲(每通道 sizeof(T2)+4 字节)只在对应输入存在时才按通道数开:
        // 无 gamma/beta 时按 chanAlign 开是纯浪费, 通道多时直接把 UB 撑爆(issue #21: chanPerCore=169013
        // 需 2.03MB, UB 仅 253952B → MTE2 写越界 errcode 82)。缺省给一个最小块, 使 gF/bF 的 Get<>() 仍合法
        // ——其值不会被读: DoPass2Normalize 在 hasGamma/hasBeta 为假时取常量 1.0f/0.0f。
        // 本核通道数受 host MaybeSetSplitReduce 的 UB 预算校验约束, 二者必须同时维护。
        pipe.InitBuffer(gRawBuf, hasGamma ? chanAlign * sizeof(T2) : BLOCK_SIZE);
        pipe.InitBuffer(bRawBuf, hasBeta ? chanAlign * sizeof(T2) : BLOCK_SIZE);
        pipe.InitBuffer(gFBuf, hasGamma ? chanAlign * sizeof(float) : BLOCK_SIZE);
        pipe.InitBuffer(bFBuf, hasBeta ? chanAlign * sizeof(float) : BLOCK_SIZE);
        // WriteMeanRstd 专用(ProcessMeanAndRstd idiom):Reg 读满 VL_FP32,故 mean/rstd 各留 VL_FP32
        pipe.InitBuffer(mrFBuf, 2 * VL_FP32 * sizeof(float));
        pipe.InitBuffer(mrOBuf, 2 * VL_FP32 * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        // 用 FetchEventID(run-once,无需 Release)做 Scalar<->MTE/V 跨 pipe 同步
        event_t evS2M3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        event_t evM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t evS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t evV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t evM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        DoPass1PartialReduce(evV2S, evS2M3);
        SyncAll();
        float mean = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(mean, rstd, evM2S);

        if (myChan == 0) {
            return;
        }

        PreloadGammaBeta(evM2V, evV2S);
        DoPass2Normalize(mean, rstd);
        // pass2 全部完成后再写 mean/rstd(隔离 MTE3/时序干扰), 全屏障后由 subIdx==0 单核写
        PipeBarrier<PIPE_ALL>();
        if (subIdx == 0) {
            WriteMeanRstd(mean, rstd, evS2V);
        }
    }

    // Pass1: 本核 myElem 个连续元素求 sum, sumsq(向量内累加,末尾一次 V->S), partial 写 workspace[blockIdx*2]。
    __aicore__ inline void DoPass1PartialReduce(event_t evV2S, event_t evS2M3)
    {
        LocalTensor<float> acc = reduf.Get<float>(); // [0]=sum, [8]=sumsq, [16]=ReduceSum 结果暂存
        LocalTensor<float> wk = wbuf.Get<float>();
        Duplicate(acc, 0.0f, 24);
        PipeBarrier<PIPE_V>();
        for (int64_t off = 0; off < myElem; off += TILE) {
            int64_t cur = (myElem - off) < TILE ? (myElem - off) : TILE;
            LocalTensor<T1> xin = inQue.AllocTensor<T1>();
            DataCopyExtParams cp{1, (uint32_t)(cur * sizeof(T1)), 0, 0, 0};
            DataCopyPadExtParams<T1> pad{false, 0, 0, 0};
            DataCopyPad(xin, xGm[xBase + off], cp, pad);
            inQue.EnQue(xin);
            xin = inQue.DeQue<T1>();
            LocalTensor<float> xf = fbuf.Get<float>();
            LocalTensor<float> sq = rbuf.Get<float>();
            Cast(xf, xin, RoundMode::CAST_NONE, cur);
            inQue.FreeTensor(xin);
            PipeBarrier<PIPE_V>();
            Mul(sq, xf, xf, cur);
            PipeBarrier<PIPE_V>();
            ReduceSum(acc[16], xf, wk, cur);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, acc[16], 1);
            PipeBarrier<PIPE_V>();
            ReduceSum(acc[16], sq, wk, cur);
            PipeBarrier<PIPE_V>();
            Add(acc[8], acc[8], acc[16], 1);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(evV2S);
        WaitFlag<HardEvent::V_S>(evV2S);
        float sumLocal = acc.GetValue(0);
        float sqLocal = acc.GetValue(8);

        // partial -> workspace[blockIdx*2]
        LocalTensor<float> part = partBuf.Get<float>();
        part.SetValue(0, sumLocal);
        part.SetValue(1, sqLocal);
        SetFlag<HardEvent::S_MTE3>(evS2M3);
        WaitFlag<HardEvent::S_MTE3>(evS2M3);
        DataCopyExtParams cpp{1, (uint32_t)(2 * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm[blockIdx * 2], part, cpp);
    }

    // combine: 读回本组 coresPerGroup 个 partial,合成全局 sum/sumsq -> mean/rstd。
    __aicore__ inline void ComputeMeanRstd(float& mean, float& rstd, event_t evM2S)
    {
        int64_t groupCoreBase = groupGlobalIdx * coresPerGroup;
        LocalTensor<float> comb = combBuf.Get<float>();
        DataCopyExtParams cpr{1, (uint32_t)(coresPerGroup * 2 * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padr{false, 0, 0, 0};
        DataCopyPad(comb, wsGm[groupCoreBase * 2], cpr, padr);
        SetFlag<HardEvent::MTE2_S>(evM2S);
        WaitFlag<HardEvent::MTE2_S>(evM2S);
        float gSum = 0.0f, gSq = 0.0f;
        for (int64_t i = 0; i < coresPerGroup; i++) {
            gSum += comb.GetValue(i * 2);
            gSq += comb.GetValue(i * 2 + 1);
        }
        // mean/var 用主机预算 invCnt(避免设备标量浮点除法);值有界
        float invCnt = tiling->groupInvCnt;
        mean = gSum * invCnt;
        float var = gSq * invCnt - mean * mean;
        // rstd = 1/sqrt(var+eps):精确 Sqrt + Div(1/sqrt), 与 golden 的 1.0/np.sqrt(var+eps) 同序(先开方后取倒),
        // 正确舍入下逐位吻合;不用 Rsqrt(float 误差~2e-4 超万分之一, 见接口约束)。
        event_t evV2Ssq = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        LocalTensor<float> sbuf = rbuf.Get<float>();
        LocalTensor<float> onev = wbuf.Get<float>(); // reduce 后空闲, 作被除数 1.0
        Duplicate(sbuf, var + eps, 64);
        Duplicate(onev, 1.0f, 64);
        PipeBarrier<PIPE_V>();
        Sqrt(sbuf, sbuf, 64);
        PipeBarrier<PIPE_V>();
        Div(onev, onev, sbuf, 64); // onev = 1/sqrt(var+eps)
        SetFlag<HardEvent::V_S>(evV2Ssq);
        WaitFlag<HardEvent::V_S>(evV2Ssq);
        rstd = onev.GetValue(0);
    }

    // 预加载本核通道段的 gamma/beta 到 float UB(避免标量 bf16 转换)。
    __aicore__ inline void PreloadGammaBeta(event_t evM2V, event_t evV2S)
    {
        LocalTensor<float> gF = gFBuf.Get<float>();
        LocalTensor<float> bF = bFBuf.Get<float>();
        if (hasGamma) {
            LocalTensor<T2> gRaw = gRawBuf.Get<T2>();
            DataCopyExtParams cpg{1, (uint32_t)(myChan * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padg{false, 0, 0, 0};
            DataCopyPad(gRaw, gammaGm[gammaBase], cpg, padg);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(gF, gRaw, RoundMode::CAST_NONE, myChan);
            PipeBarrier<PIPE_V>();
        }
        if (hasBeta) {
            LocalTensor<T2> bRaw = bRawBuf.Get<T2>();
            DataCopyExtParams cpb{1, (uint32_t)(myChan * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padb{false, 0, 0, 0};
            DataCopyPad(bRaw, betaGm[gammaBase], cpb, padb);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(bF, bRaw, RoundMode::CAST_NONE, myChan);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(evV2S);
        WaitFlag<HardEvent::V_S>(evV2S);
    }

    // Pass2: 逐通道 normalize + silu + quant, 逐块落盘 int8。
    __aicore__ inline void DoPass2Normalize(float mean, float rstd)
    {
        LocalTensor<float> gF = gFBuf.Get<float>();
        LocalTensor<float> bF = bFBuf.Get<float>();
        for (int64_t ci = 0; ci < myChan; ci++) {
            float gVal = hasGamma ? gF.GetValue(ci) : 1.0f;
            float bVal = hasBeta ? bF.GetValue(ci) : 0.0f;
            float qsVal = perChannelScale ? quantScaleGm.GetValue(qsBase + ci) : quantScaleGm.GetValue(0);
            float invqs = 1.0f / qsVal;
            float scl = rstd * gVal;
            int64_t chOff = xBase + ci * hwNum;
            NormalizeChannelTiles(chOff, mean, scl, bVal, invqs);
        }
    }

    // 单通道: 逐 TILE 块 normalize + silu + quant, 落盘 int8。
    __aicore__ inline void NormalizeChannelTiles(int64_t chOff, float mean, float scl, float bVal, float invqs)
    {
        for (int64_t off = 0; off < hwNum; off += TILE) {
            int64_t cur = (hwNum - off) < TILE ? (hwNum - off) : TILE;
            LocalTensor<T1> xin = inQue.AllocTensor<T1>();
            DataCopyExtParams cp{1, (uint32_t)(cur * sizeof(T1)), 0, 0, 0};
            DataCopyPadExtParams<T1> pad{false, 0, 0, 0};
            DataCopyPad(xin, xGm[chOff + off], cp, pad);
            inQue.EnQue(xin);
            xin = inQue.DeQue<T1>();
            LocalTensor<float> yf = rbuf.Get<float>();
            Cast(yf, xin, RoundMode::CAST_NONE, cur);
            inQue.FreeTensor(xin);
            PipeBarrier<PIPE_V>();
            Adds(yf, yf, -mean, cur);
            PipeBarrier<PIPE_V>();
            Muls(yf, yf, scl, cur);
            PipeBarrier<PIPE_V>();
            Adds(yf, yf, bVal, cur);
            PipeBarrier<PIPE_V>();
            if (activateSilu) {
                LocalTensor<float> sg = fbuf.Get<float>();
                Sigmoid(sg, yf, cur);
                PipeBarrier<PIPE_V>();
                Mul(yf, yf, sg, cur);
                PipeBarrier<PIPE_V>();
            }
            Muls(yf, yf, invqs, cur);
            PipeBarrier<PIPE_V>();
            Maxs(yf, yf, -128.0f, cur);
            PipeBarrier<PIPE_V>();
            Mins(yf, yf, 127.0f, cur);
            PipeBarrier<PIPE_V>();
            // 量化收尾 fp32->int16(CAST_RINT)->fp16->int8(高精度idiom, 抄 dynamic_block_quant)。
            // 先 fp32->int16 把值正确舍入成整数(对齐 golden round(out/scale) 的 fp32 舍入),
            // 再 int16->fp16->int8 全精确(|y|<=127 的整数在 fp16 精确表示)。勿走
            // fp32->fp16->int8(大值fp16分辨率翻边界)。
            LocalTensor<int16_t> yi16 = fbuf.Get<int16_t>();
            Cast(yi16, yf, RoundMode::CAST_RINT, cur);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> yh = wbuf.Get<half>();
            Cast(yh, yi16, RoundMode::CAST_NONE, cur);
            PipeBarrier<PIPE_V>();
            LocalTensor<int8_t> yo = outQue.AllocTensor<int8_t>();
            Cast(yo, yh, RoundMode::CAST_RINT, cur);
            outQue.EnQue(yo);
            yo = outQue.DeQue<int8_t>();
            DataCopyExtParams cpo{1, (uint32_t)(cur * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(siluGm[chOff + off], yo, cpo);
            outQue.FreeTensor(yo);
        }
    }

private:
    // mean/rstd 各用完全独立的对齐缓冲(mean->gRawBuf, rstd->bRawBuf, 此刻均空闲),各自从对齐 [0] 写,
    // 杜绝 buffer 复用 race 与 DataCopyPad 源非对齐(UB 源须 32B 对齐)。
    // 复用现有 kernel 的 ProcessMeanAndRstd idiom(稳定):mean/rstd 各占 VL_FP32,SetValue 后 S->V,由 ProcessMeanAndRstd
    // 内部 Reg 窄化 + DIST_PACK_B32 掩码存,再 CopyMeanAndRstd2Gm(blockCount=1,copyLen=1)写 GM。
    __aicore__ inline void WriteMeanRstd(float mean, float rstd, event_t evS2V)
    {
        LocalTensor<float> meanT = mrFBuf.Get<float>();
        LocalTensor<float> rstdT = mrFBuf.Get<float>()[VL_FP32];
        meanT.SetValue(0, mean);
        rstdT.SetValue(0, rstd);
        if constexpr (sizeof(T2) == sizeof(float)) {
            // fp32 输出:SetValue(S) 后直接 DataCopyPad(MTE3), 需 S->MTE3
            event_t evS2M3o = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(evS2M3o);
            WaitFlag<HardEvent::S_MTE3>(evS2M3o);
            CopyMeanAndRstd2Gm<float>(meanT2Gm[groupGlobalIdx], rstdT2Gm[groupGlobalIdx], meanT, rstdT, 1, 1);
        } else {
            // T1(half/bf16)输出:ProcessMeanAndRstd 内部 Reg 读 UB(V), 需 S->V
            LocalTensor<T1> meanO = mrOBuf.Get<T1>();
            LocalTensor<T1> rstdO = mrOBuf.Get<T1>()[VL_FP32];
            SetFlag<HardEvent::S_V>(evS2V);
            WaitFlag<HardEvent::S_V>(evS2V);
            ProcessMeanAndRstd<T1>(meanT, meanO, meanGm, rstdT, rstdO, rstdGm, groupGlobalIdx, 1);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<> fbuf;    // Pass1 xf / Pass2 中间
    TBuf<> rbuf;    // Pass1 sq / Pass2 yf / rstd 计算 sbuf
    TBuf<> wbuf;    // ReduceSum sharedTmpBuffer
    TBuf<> reduf;   // 归约累加 acc([0]=sum [8]=sumsq [16]=tmp)
    TBuf<> partBuf; // partial(sum,sumsq)写 workspace 暂存
    TBuf<> combBuf; // combine 读回本组全部 partial
    TBuf<> gRawBuf; // gamma 原始(T2)
    TBuf<> bRawBuf; // beta 原始(T2)
    TBuf<> gFBuf;   // gamma float
    TBuf<> bFBuf;   // beta float
    TBuf<> mrFBuf;  // mean/rstd float 暂存(ProcessMeanAndRstd)
    TBuf<> mrOBuf;  // mean/rstd T1 输出暂存

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<int8_t> siluGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<T2> rstdT2Gm;
    GlobalTensor<float> wsGm;

    const GroupNormSiluQuantRegbaseTilingData* tiling;
    int64_t blockIdx;
    int64_t numGroups;
    int64_t hwNum;
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    int64_t coresPerGroup;
    int64_t groupGlobalIdx;
    int64_t subIdx;
    int64_t chanPerCore;
    int64_t chanStart;
    int64_t myChan;
    int64_t myElem;
    int64_t xBase;
    int64_t gammaBase;
    int64_t qsBase;
    float eps;
    bool perChannelScale{false};
    bool activateSilu{true};
    bool hasGamma{false};
    bool hasBeta{false};
};

} // namespace GroupNormSiluQuant

#endif // GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_
