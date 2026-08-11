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
 * \file instance_norm_grad_base.h
 * \brief Shared base for InstanceNormGrad arch35 kernels (GM binding, tiling parse, DataCopyPad
 *        helpers, per-task param loading, and the deterministic cross-N stage2 reduction).
 */
#ifndef INSTANCE_NORM_GRAD_BASE_H
#define INSTANCE_NORM_GRAD_BASE_H
#pragma once

#include "kernel_operator.h"
#include "instance_norm_grad_common.h"

namespace InstanceNormGrad {
using namespace AscendC;

template <typename T>
class InstanceNormGradBase {
public:
    __aicore__ inline InstanceNormGradBase() {}
    __aicore__ inline ~InstanceNormGradBase() {}

protected:
    __aicore__ inline void InitCommon(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean, GM_ADDR gamma,
                                      GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta, GM_ADDR workspace,
                                      const InstanceNormGradTilingData* __restrict tiling, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        blockIdx_ = GetBlockIdx();
        N_ = tiling->N;
        C_ = tiling->C;
        M_ = tiling->M;
        cTile_ = tiling->cTile;
        cTileNum_ = tiling->cTileNum;
        taskNum_ = tiling->taskNum;
        taskNumPerCore_ = tiling->taskNumPerCore;
        taskNumPerTailCore_ = tiling->taskNumPerTailCore;
        tailCore_ = tiling->tailCore;
        stage1CoreUsed_ = tiling->stage1CoreUsed;
        mUbTile_ = tiling->mUbTile;
        mUbIterNum_ = tiling->mUbIterNum;
        mUbTailNum_ = tiling->mUbTailNum;
        reduceNCnt_ = tiling->reduceNCnt;
        workSpaceSize_ = tiling->workSpaceSize;
        stage2CoreUsed_ = tiling->stage2CoreUsed;
        cBlockFactor_ = tiling->cBlockFactor;
        cTailBlockFactor_ = tiling->cTailBlockFactor;
        stage2SubCap_ = tiling->stage2SubCap;

        // this core's stage1 task range [startTask_, startTask_ + curCoreTaskNum_)
        if (blockIdx_ < tailCore_) {
            curCoreTaskNum_ = taskNumPerCore_;
            startTask_ = static_cast<int64_t>(taskNumPerCore_) * blockIdx_;
        } else {
            curCoreTaskNum_ = taskNumPerTailCore_;
            startTask_ = static_cast<int64_t>(taskNumPerCore_) * tailCore_ +
                         static_cast<int64_t>(taskNumPerTailCore_) * (blockIdx_ - tailCore_);
        }

        int64_t allEle = N_ * M_ * C_;
        dyGm_.SetGlobalBuffer((__gm__ T*)dy, allEle);
        xGm_.SetGlobalBuffer((__gm__ T*)x, allEle);
        pdxGm_.SetGlobalBuffer((__gm__ T*)pd_x, allEle);
        varGm_.SetGlobalBuffer((__gm__ T*)variance, N_ * C_);
        meanGm_.SetGlobalBuffer((__gm__ T*)mean, N_ * C_);
        gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, C_);
        pdGammaGm_.SetGlobalBuffer((__gm__ T*)pd_gamma, C_);
        pdBetaGm_.SetGlobalBuffer((__gm__ T*)pd_beta, C_);
        dgammaWs_.SetGlobalBuffer((__gm__ float*)workspace, workSpaceSize_);
        dbetaWs_.SetGlobalBuffer((__gm__ float*)workspace + workSpaceSize_, workSpaceSize_);

        cAlignF32_ = CeilAlign(static_cast<uint32_t>(cTile_), VL_FP32);
        rowStrideMaxT_ = CeilAlign(static_cast<uint32_t>(cTile_) * static_cast<uint32_t>(sizeof(T)), GetUbBlockSize()) /
                         sizeof(T);
    }

    __aicore__ inline void InitStage1Buffers()
    {
        uint32_t tileBytes = (mUbTile_ * rowStrideMaxT_ + VL_FP32) * sizeof(T);
        pipe_->InitBuffer(inQueX_, DOUBLE_BUFFER, tileBytes);
        pipe_->InitBuffer(inQueDy_, DOUBLE_BUFFER, tileBytes);
        pipe_->InitBuffer(outQuePdx_, DOUBLE_BUFFER, tileBytes);
        uint32_t paramBytes = cAlignF32_ * sizeof(float);
        pipe_->InitBuffer(varBuf_, paramBytes);
        pipe_->InitBuffer(meanBuf_, paramBytes);
        pipe_->InitBuffer(gammaBuf_, paramBytes);
        pipe_->InitBuffer(rstdBuf_, paramBytes);
        pipe_->InitBuffer(pdVarBuf_, paramBytes);
        pipe_->InitBuffer(pdMeanBuf_, paramBytes);
        pipe_->InitBuffer(accDgammaBuf_, paramBytes);
        pipe_->InitBuffer(accDbetaBuf_, paramBytes);
        pipe_->InitBuffer(cDgammaBuf_, paramBytes); // Kahan compensation, persisted across M-tiles
        pipe_->InitBuffer(cDbetaBuf_, paramBytes);
        pipe_->InitBuffer(cPdVarBuf_, paramBytes);  // Kahan compensation for pdVar
        pipe_->InitBuffer(cPdMeanBuf_, paramBytes); // Kahan compensation for pdMean
        pipe_->InitBuffer(tmpParamBuf_, cAlignF32_ * sizeof(T));
    }

    __aicore__ inline uint32_t RowStrideT(uint32_t cLen) const
    {
        return CeilAlign(cLen * static_cast<uint32_t>(sizeof(T)), GetUbBlockSize()) / sizeof(T);
    }

    // Split a global taskId into (n, cStart, cLen). cTiles tile [0, C) fully.
    __aicore__ inline void GetTaskCoords(int64_t taskId, int64_t& n, int64_t& cStart, uint32_t& cLen) const
    {
        n = taskId / cTileNum_;
        int64_t cIdx = taskId % cTileNum_;
        cStart = cIdx * cTile_;
        int64_t remain = C_ - cStart;
        cLen = static_cast<uint32_t>(remain < cTile_ ? remain : cTile_);
    }

    // Load variance/mean/gamma for [cStart, cStart+cLen), compute rstd, zero the four accumulators.
    __aicore__ inline void LoadTaskParams(int64_t n, int64_t cStart, uint32_t cLen)
    {
        // Cross-task hazard: with many instances per core (large N) this core runs ProcessTask
        // back-to-back. The previous task's ComputePdx (V) is still reading mean/rstd/gamma when the
        // MTE2 param loads below overwrite those buffers, so some tasks read corrupted params and
        // their pd_x / pd_gamma come out wrong (pd_beta uses only dy, so it stays correct). Wait for
        // the prior VEC reads to drain before the MTE2 overwrite. (The fp32 LoadOneParamToF32 path
        // only issues MTE2->V after the load, not V->MTE2 before it, so add the guard here.)
        SyncVToMte2();
        LoadOneParamToF32(varGm_, n * C_ + cStart, cLen, varBuf_.template Get<float>());
        LoadOneParamToF32(meanGm_, n * C_ + cStart, cLen, meanBuf_.template Get<float>());
        LoadOneParamToF32(gammaGm_, cStart, cLen, gammaBuf_.template Get<float>());
        ComputeRstd((__local_mem__ float*)varBuf_.template Get<float>().GetPhyAddr(),
                    (__local_mem__ float*)rstdBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)pdVarBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)pdMeanBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)accDgammaBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)accDbetaBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)cDgammaBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)cDbetaBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)cPdVarBuf_.template Get<float>().GetPhyAddr(), cLen);
        ZeroF32((__local_mem__ float*)cPdMeanBuf_.template Get<float>().GetPhyAddr(), cLen);
    }

    __aicore__ inline void LoadOneParamToF32(const GlobalTensor<T>& gm, int64_t gmOffset, uint32_t cLen,
                                             const LocalTensor<float>& fp32Buf)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(cLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(fp32Buf, gm[gmOffset], params, pad);
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
        } else {
            LocalTensor<T> tmp = tmpParamBuf_.template Get<T>();
            DataCopyPad(tmp, gm[gmOffset], params, pad);
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
            CastTToF32(tmp, fp32Buf, cLen);
            // tmpParamBuf_ is shared by the successive var/mean/gamma loads. The next param's
            // DataCopyPad (MTE2) must wait for this cast (V) to finish reading tmp, else the fp16
            // params race and overwrite each other mid-cast. (fp32 path copies straight to fp32Buf.)
            SyncVToMte2();
        }
    }

    __aicore__ inline void CastTToF32(const LocalTensor<T>& src, const LocalTensor<float>& dst, uint32_t cLen)
    {
        __local_mem__ T* s = (__local_mem__ T*)src.GetPhyAddr();
        __local_mem__ float* d = (__local_mem__ float*)dst.GetPhyAddr();
        uint16_t loopCnt = (cLen + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> v;
            MaskReg preg;
            uint32_t sreg = cLen;
            for (uint16_t i = 0; i < loopCnt; ++i) {
                preg = UpdateMask<float>(sreg);
                LoadTAsF32<T>(s, v, preg, i * VL_FP32);
                DataCopy(d + i * VL_FP32, v, preg);
            }
        }
    }

    // DataCopyPad a [rows, cLen] sub-block (GM row stride = C) into a UB tile (32B-aligned rows).
    __aicore__ inline void CopyInTile(const GlobalTensor<T>& gm, const LocalTensor<T>& local, int64_t n,
                                      uint32_t mStart, uint32_t rows, int64_t cStart, uint32_t cLen)
    {
        int64_t baseOff = (n * M_ + mStart) * C_ + cStart;
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T)),
                                 static_cast<uint32_t>((C_ - cLen) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, gm[baseOff], params, pad);
    }

    __aicore__ inline void CopyOutPdx(const LocalTensor<T>& local, int64_t n, uint32_t mStart, uint32_t rows,
                                      int64_t cStart, uint32_t cLen)
    {
        int64_t baseOff = (n * M_ + mStart) * C_ + cStart;
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T)), 0,
                                 static_cast<uint32_t>((C_ - cLen) * sizeof(T)), 0};
        DataCopyPad(pdxGm_[baseOff], local, params);
    }

    // After pass1: emit dgamma/dbeta. N==1 -> straight to output; else -> workspace row n (fp32).
    __aicore__ inline void WritePartialOrOutput(int64_t n, int64_t cStart, uint32_t cLen)
    {
        LocalTensor<float> accDg = accDgammaBuf_.template Get<float>();
        LocalTensor<float> accDb = accDbetaBuf_.template Get<float>();
        if (N_ <= 1) {
            LocalTensor<T> tmp = tmpParamBuf_.template Get<T>();
            CastF32ToT<T>((__local_mem__ float*)accDg.GetPhyAddr(), (__local_mem__ T*)tmp.GetPhyAddr(), cLen);
            SyncVToMte3();
            DataCopyExtParams p{1, static_cast<uint32_t>(cLen * sizeof(T)), 0, 0, 0};
            DataCopyPad(pdGammaGm_[cStart], tmp, p);
            SyncMte3ToV();
            CastF32ToT<T>((__local_mem__ float*)accDb.GetPhyAddr(), (__local_mem__ T*)tmp.GetPhyAddr(), cLen);
            SyncVToMte3();
            DataCopyPad(pdBetaGm_[cStart], tmp, p);
            SyncMte3ToV();
        } else {
            SyncVToMte3();
            DataCopyExtParams p{1, static_cast<uint32_t>(cLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(dgammaWs_[n * C_ + cStart], accDg, p);
            DataCopyPad(dbetaWs_[n * C_ + cStart], accDb, p);
            SyncMte3ToV();
        }
    }

    // ---- deterministic cross-N stage2 (fixed n order) --------------------------------------
    __aicore__ inline void Stage2Process()
    {
        if (N_ <= 1) {
            return;
        }
        pipe_->Reset();
        SyncAll();
        if (blockIdx_ >= stage2CoreUsed_) {
            return;
        }
        int64_t cStart2 = static_cast<int64_t>(blockIdx_) * cBlockFactor_;
        uint32_t cLen2 = (blockIdx_ == stage2CoreUsed_ - 1) ? static_cast<uint32_t>(cTailBlockFactor_) :
                                                              static_cast<uint32_t>(cBlockFactor_);
        // 每轮处理的通道数由 host 按 ubSize 算好下发(见 tiling 的 STAGE2_BUFFERS_F32),
        // 内核不再自带容量常量,避免改缓冲个数时两边失配导致 UB 超限。
        uint32_t cSubCap = stage2SubCap_; // 已由 host 按向量长度向下对齐,此处不可再向上取整
        if (cSubCap == 0) {
            cSubCap = VL_FP32;
        }
        pipe_->InitBuffer(s2InQue_, DOUBLE_BUFFER, cSubCap * sizeof(float));
        pipe_->InitBuffer(s2AccDgBuf_, cSubCap * sizeof(float));
        pipe_->InitBuffer(s2AccDbBuf_, cSubCap * sizeof(float));
        pipe_->InitBuffer(s2CDgBuf_, cSubCap * sizeof(float)); // Kahan compensation, cross-N merge
        pipe_->InitBuffer(s2CDbBuf_, cSubCap * sizeof(float));
        pipe_->InitBuffer(s2OutBuf_, cSubCap * sizeof(T));

        for (uint32_t cs = 0; cs < cLen2; cs += cSubCap) {
            uint32_t cw = (cs + cSubCap <= cLen2) ? cSubCap : (cLen2 - cs);
            Stage2ReduceSub(cStart2 + cs, cw);
        }
    }

    __aicore__ inline void Stage2ReduceSub(int64_t cStart, uint32_t cw)
    {
        LocalTensor<float> accDg = s2AccDgBuf_.template Get<float>();
        LocalTensor<float> accDb = s2AccDbBuf_.template Get<float>();
        LocalTensor<float> cDg = s2CDgBuf_.template Get<float>();
        LocalTensor<float> cDb = s2CDbBuf_.template Get<float>();
        ZeroF32((__local_mem__ float*)accDg.GetPhyAddr(), cw);
        ZeroF32((__local_mem__ float*)accDb.GetPhyAddr(), cw);
        ZeroF32((__local_mem__ float*)cDg.GetPhyAddr(), cw);
        ZeroF32((__local_mem__ float*)cDb.GetPhyAddr(), cw);
        for (int64_t n = 0; n < reduceNCnt_; ++n) {
            AddWsRow(dgammaWs_, n * C_ + cStart, cw, accDg, cDg);
            AddWsRow(dbetaWs_, n * C_ + cStart, cw, accDb, cDb);
        }
        LocalTensor<T> out = s2OutBuf_.template Get<T>();
        CastF32ToT<T>((__local_mem__ float*)accDg.GetPhyAddr(), (__local_mem__ T*)out.GetPhyAddr(), cw);
        SyncVToMte3();
        DataCopyExtParams p{1, static_cast<uint32_t>(cw * sizeof(T)), 0, 0, 0};
        DataCopyPad(pdGammaGm_[cStart], out, p);
        SyncMte3ToV();
        CastF32ToT<T>((__local_mem__ float*)accDb.GetPhyAddr(), (__local_mem__ T*)out.GetPhyAddr(), cw);
        SyncVToMte3();
        DataCopyPad(pdBetaGm_[cStart], out, p);
        SyncMte3ToV();
    }

    // 跨 N 合并 partial 行。Pass1 已在跨 M-tile 方向上了 Kahan,这里若用裸 fp32 累加,
    // N 越大误差越回吐(误差 ~ N*eps*sum|x| 对 ~2*eps*sum|x|),等于把 Pass1 省下的精度丢掉;
    // 故同样施加 Kahan 补偿(含与 Pass1 一致的 nan-guard:补偿量为 nan 时清零,让和保持 inf)。
    __aicore__ inline void AddWsRow(const GlobalTensor<float>& ws, int64_t off, uint32_t cw,
                                    const LocalTensor<float>& acc, const LocalTensor<float>& comp)
    {
        LocalTensor<float> in = s2InQue_.template AllocTensor<float>();
        DataCopyExtParams p{1, static_cast<uint32_t>(cw * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(in, ws[off], p, pad);
        s2InQue_.EnQue(in);
        in = s2InQue_.template DeQue<float>();
        __local_mem__ float* accUb = (__local_mem__ float*)acc.GetPhyAddr();
        __local_mem__ float* inUb = (__local_mem__ float*)in.GetPhyAddr();
        __local_mem__ float* compUb = (__local_mem__ float*)comp.GetPhyAddr();
        uint16_t loopCnt = (cw + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> a, b, c, kY, kT, kD, zeroReg;
            MaskReg preg;
            MaskReg nanMask;
            uint32_t sreg = cw;
            for (uint16_t i = 0; i < loopCnt; ++i) {
                preg = UpdateMask<float>(sreg);
                Duplicate(zeroReg, 0.0f, preg);
                DataCopy(a, accUb + i * VL_FP32);
                DataCopy(b, inUb + i * VL_FP32);
                DataCopy(c, compUb + i * VL_FP32);
                Sub(kY, b, c, preg);  // y = addend - compensation
                Add(kT, a, kY, preg); // t = sum + y
                Sub(kD, kT, a, preg); // d = t - sum
                Sub(c, kD, kY, preg); // c = d - y (lost low-order part)
                Compare<float, CMPMODE::EQ>(nanMask, c, c, preg);
                Select(c, c, zeroReg, nanMask); // nan -> 0, keep sum at inf
                Move(a, kT, preg);
                DataCopy(accUb + i * VL_FP32, a, preg);
                DataCopy(compUb + i * VL_FP32, c, preg);
            }
        }
        s2InQue_.FreeTensor(in);
    }

    __aicore__ inline void SyncVToMte3()
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
    }
    __aicore__ inline void SyncMte3ToV()
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(e);
        WaitFlag<HardEvent::MTE3_V>(e);
    }
    __aicore__ inline void SyncMte2ToV()
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void SyncVToMte2()
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        SetFlag<HardEvent::V_MTE2>(e);
        WaitFlag<HardEvent::V_MTE2>(e);
    }

protected:
    TPipe* pipe_ = nullptr;
    int32_t blockIdx_ = 0;

    GlobalTensor<T> dyGm_, xGm_, pdxGm_;
    GlobalTensor<T> varGm_, meanGm_, gammaGm_, pdGammaGm_, pdBetaGm_;
    GlobalTensor<float> dgammaWs_, dbetaWs_;

    TQue<QuePosition::VECIN, 2> inQueX_, inQueDy_;
    TQue<QuePosition::VECOUT, 2> outQuePdx_;
    TBuf<TPosition::VECCALC> varBuf_, meanBuf_, gammaBuf_, rstdBuf_;
    TBuf<TPosition::VECCALC> pdVarBuf_, pdMeanBuf_, accDgammaBuf_, accDbetaBuf_, tmpParamBuf_;
    TBuf<TPosition::VECCALC> cDgammaBuf_, cDbetaBuf_, cPdVarBuf_, cPdMeanBuf_;
    // stage2 buffers
    TQue<QuePosition::VECIN, 2> s2InQue_;
    TBuf<TPosition::VECCALC> s2AccDgBuf_, s2AccDbBuf_, s2CDgBuf_, s2CDbBuf_, s2OutBuf_;

    int64_t N_ = 0, C_ = 0, M_ = 1;
    int64_t cTile_ = 0, cTileNum_ = 1, taskNum_ = 0;
    uint32_t taskNumPerCore_ = 0, taskNumPerTailCore_ = 0, tailCore_ = 0, stage1CoreUsed_ = 0;
    uint32_t mUbTile_ = 0, mUbIterNum_ = 1, mUbTailNum_ = 0;
    int64_t reduceNCnt_ = 0, workSpaceSize_ = 0;
    uint32_t stage2CoreUsed_ = 0;
    uint32_t stage2SubCap_ = 0;
    int64_t cBlockFactor_ = 0, cTailBlockFactor_ = 0;
    int64_t startTask_ = 0;
    uint32_t curCoreTaskNum_ = 0;
    uint32_t cAlignF32_ = 0;
    uint32_t rowStrideMaxT_ = 0;
};
} // namespace InstanceNormGrad
#endif // INSTANCE_NORM_GRAD_BASE_H
