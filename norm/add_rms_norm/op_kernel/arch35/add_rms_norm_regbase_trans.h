/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_RMS_NORM_REGBASE_TRANS_H
#define ADD_RMS_NORM_REGBASE_TRANS_H
#include "add_rms_norm_regbase_common.h"
#include "../../rms_norm/rms_norm_base.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddRmsNorm {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

constexpr int32_t TRANS_INPUT_NUM = 2; // 算法固定处理 x1、x2 两路输入，不是硬件对齐参数。
constexpr int32_t TRANS_B16_PER_B32 = sizeof(float) / sizeof(uint16_t);
// TransDataTo5HD 的 src/dst 地址表长度由接口规定；FP32 每个元素占两个 B16 地址槽。
constexpr int32_t TRANS_ADDR_LIST_SIZE = NCHW_CONV_ADDR_LIST_SIZE;
constexpr int32_t TRANS_FP32_ADDR_GROUP_SIZE = TRANS_ADDR_LIST_SIZE / TRANS_B16_PER_B32;

/**
 * 小 R 转置模板：将 [A,R] 搬成 UB [R,A] 后沿 A 向量化，计算完成再转回；
 * 每核可循环多个 A tile，性能准入由 Host 决定。
 */
template <typename T>
class KernelAddRmsNormRegBaseTrans {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

public:
    __aicore__ inline explicit KernelAddRmsNormRegBaseTrans(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                                const AddRMSNormRegbaseTransTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        // key=4000 保留原有 TilingData 布局，避免 Host/Kernel 混合版本解码错位。
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        rAligned = tiling->rAligned;
        rTileBase = tiling->rTileBase;
        tileALen = tiling->tileALen;
        epsilon = tiling->epsilon;
        avgFactor = tiling->avgFactor;

        xGm1.SetGlobalBuffer((__gm__ T*)x1);
        xGm2.SetGlobalBuffer((__gm__ T*)x2);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd);
        xOutGm.SetGlobalBuffer((__gm__ T*)x);

        // 二维 buffer 按最大 A tile 分配，输入与转置输出各使用两个物理槽位。
        uint64_t xShapeLen = tileALen * rAligned;
        pPipe->InitBuffer(x1Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pPipe->InitBuffer(x2Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pPipe->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pPipe->InitBuffer(xOutQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        // gamma 每核只搬入并转 FP32 一次。
        pPipe->InitBuffer(gammaQueue_, 1, rAligned * sizeof(T));
        // 保留第一遍的 FP32 x1+x2，第二遍直接复用。
        pPipe->InitBuffer(xAddBuf_, xShapeLen * sizeof(float));
        pPipe->InitBuffer(tmpBuf_, rAligned * sizeof(float));
        // rstdQueue_ 管理跨 tile 的 V→MTE3 生命周期。
        pPipe->InitBuffer(rstdQueue_, BUFFER_NUM, tileALen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t usedCoreNum = GetBlockNum();
        // 商余法把连续 A 行均衡分核。
        int64_t baseRowsPerCore = static_cast<int64_t>(numRow) / usedCoreNum;
        int64_t extraRowCoreNum = static_cast<int64_t>(numRow) % usedCoreNum;
        int64_t beginRow = blockIdx * baseRowsPerCore + (blockIdx < extraRowCoreNum ? blockIdx : extraRowCoreNum);
        int64_t coreRowNum = baseRowsPerCore + (blockIdx < extraRowCoreNum ? 1 : 0);
        int64_t endRow = beginRow + coreRowNum;

        // gamma 在本核所有 A tile 间复用。
        CopyInGamma();
        LocalTensor<T> gammaLocal = gammaQueue_.DeQue<T>();

        LocalTensor<float> gammaFp32 = tmpBuf_.Get<float>();
        __ubuf__ T* gammaInUb = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ float* gammaFp32InUb = (__ubuf__ float*)gammaFp32.GetPhyAddr();
        uint32_t gammaMaskCount = static_cast<uint32_t>(numCol);
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg;
            MaskReg gammaMask = UpdateMask<float>(gammaMaskCount);
            LoadRegForDtype<T>(gammaInUb, gammaReg, gammaMask, 0);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaFp32InUb, gammaReg, gammaMask);
        }

        // 首 tile 预取，后续迭代复用双缓冲队列。
        uint32_t firstTileALen = static_cast<uint32_t>(
            coreRowNum < static_cast<int64_t>(tileALen) ? coreRowNum : static_cast<int64_t>(tileALen));
        int64_t firstXOffset = beginRow * numCol;
        // NDDMA 首次访问 GM 前每核刷新一次 DataCache，后续 tile 无需重复刷新。
        NdDmaDci();
        CopyInX(firstXOffset, firstTileALen);

        uint64_t coreTileCount = static_cast<uint64_t>(coreRowNum) / tileALen +
                                 static_cast<uint64_t>(static_cast<uint64_t>(coreRowNum) % tileALen != 0);
        for (uint64_t tileIdx = 0; tileIdx < coreTileCount; ++tileIdx) {
            int64_t curRow = beginRow + static_cast<int64_t>(tileIdx * tileALen);
            uint32_t curTileALen = static_cast<uint32_t>(
                endRow - curRow < static_cast<int64_t>(tileALen) ? endRow - curRow : static_cast<int64_t>(tileALen));
            int64_t nextRow = curRow + curTileALen;
            bool hasNextTile = nextRow < endRow;
            int64_t nextRowNum = endRow - nextRow;
            uint32_t nextTileALen = static_cast<uint32_t>(
                nextRowNum < static_cast<int64_t>(tileALen) ? nextRowNum : static_cast<int64_t>(tileALen));
            int64_t nextXOffset = nextRow * numCol;
            int64_t xOffset = curRow * numCol;
            Compute(curTileALen, hasNextTile, nextXOffset, nextTileALen);
            CopyOutY(xOffset, curTileALen);
            CopyOutX(xOffset, curTileALen);
            CopyOutRstd(curRow, curTileALen);
        }
        gammaQueue_.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        // DataCopyPad 仅搬真实 R，UB 对齐区不参与计算。
        LocalTensor<T> gammaLocal = gammaQueue_.AllocTensor<T>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = numCol * sizeof(T);
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        DataCopyPad(gammaLocal, gammaGm, params, padParams);
        gammaQueue_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, uint32_t curTileALen)
    {
        // NdDma 将 GM [A,R] 搬为 UB [R,A]。
        static constexpr NdDmaConfig config = {false};
        NdDmaLoopInfo<TRANS_INPUT_NUM> copyLoopInfo;
        copyLoopInfo.loopSrcStride[0] = 1;
        copyLoopInfo.loopSrcStride[1] = numCol;
        copyLoopInfo.loopDstStride[0] = tileALen;
        copyLoopInfo.loopDstStride[1] = 1;
        copyLoopInfo.loopSize[0] = numCol;
        copyLoopInfo.loopSize[1] = curTileALen;
        NdDmaParams<T, TRANS_INPUT_NUM> params = {copyLoopInfo, 0};

        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        DataCopy<T, TRANS_INPUT_NUM, config>(x1Local, xGm1[xGmOffset], params);
        x1Queue_.EnQue(x1Local);

        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        DataCopy<T, TRANS_INPUT_NUM, config>(x2Local, xGm2[xGmOffset], params);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t curTileALen, bool hasNextTile, int64_t nextXOffset, uint32_t nextTileALen)
    {
        // 第一遍保存 FP32 x1+x2 并计算 rstd，第二遍复用该中间结果生成 y。
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<float> xAddLocal = xAddBuf_.Get<float>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        LocalTensor<T> xOutLocal = xOutQueue_.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQueue_.AllocTensor<float>();
        LocalTensor<float> gammaFp32 = tmpBuf_.Get<float>();

        __ubuf__ T* x1InUb = (__ubuf__ T*)x1Local.GetPhyAddr();
        __ubuf__ T* x2InUb = (__ubuf__ T*)x2Local.GetPhyAddr();
        __ubuf__ float* xAddInUb = (__ubuf__ float*)xAddLocal.GetPhyAddr();
        __ubuf__ T* xOutInUb = (__ubuf__ T*)xOutLocal.GetPhyAddr();
        __ubuf__ T* yInUb = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdTmp = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ float* gammaFp32Tmp = (__ubuf__ float*)gammaFp32.GetPhyAddr();

        uint32_t sreg = curTileALen;
        uint16_t aChunks = (sreg + VL_FP32 - 1) / VL_FP32;
        uint16_t numRows = static_cast<uint16_t>(numCol);
        // UpdateMask 会递减 maskCount，必须跨 A chunk 持久保存。
        uint32_t maskCount = sreg;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> xSumReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            for (uint16_t a = 0; a < aChunks; ++a) {
                uint32_t aOff = a * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                Duplicate(sumReg, float(0.0), pregLoop);
                for (uint16_t r = 0; r < numRows; ++r) {
                    uint32_t rowOff = r * tileALen + aOff;
                    LoadRegForDtype<T>(x1InUb, x1Reg, pregLoop, rowOff);
                    LoadRegForDtype<T>(x2InUb, x2Reg, pregLoop, rowOff);
                    Add(xSumReg, x1Reg, x2Reg, pregLoop);
                    StoreAlign<float, StoreDist::DIST_NORM_B32>(xAddInUb + rowOff, xSumReg, pregLoop);
                    StoreRegForDtype<T>(xOutInUb, xSumReg, pregLoop, rowOff);
                    Mul(sqReg, xSumReg, xSumReg, pregLoop);
                    Add(sumReg, sumReg, sqReg, pregLoop);
                }
                Muls(sumReg, sumReg, avgFactor, pregLoop);
                NormCommon::ComputeRstdNewtonRaphsonReg<true>(sumReg, rstdReg, pregLoop, epsilon);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(rstdTmp + aOff, rstdReg, pregLoop);
            }
        }

        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
        // 输入槽位释放后预取下一 tile。
        if (hasNextTile) {
            CopyInX(nextXOffset, nextTileALen);
        }

        uint32_t outputMaskCount = sreg;
        __VEC_SCOPE__
        {
            RegTensor<float> xSumReg;
            RegTensor<float> gammaReg;
            RegTensor<float> yReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            for (uint16_t a = 0; a < aChunks; ++a) {
                uint32_t aOff = a * VL_FP32;
                pregLoop = UpdateMask<float>(outputMaskCount);
                LoadAlign(rstdReg, rstdTmp + aOff);
                for (uint16_t r = 0; r < numRows; ++r) {
                    uint32_t rowOff = r * tileALen + aOff;
                    LoadAlign(xSumReg, xAddInUb + rowOff);
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(gammaReg, gammaFp32Tmp + r);
                    Mul(yReg, xSumReg, gammaReg, pregLoop);
                    Mul(yReg, yReg, rstdReg, pregLoop);
                    StoreRegForDtype<T>(yInUb, yReg, pregLoop, rowOff);
                }
            }
        }

        yQueue_.EnQue<T>(yLocal);
        xOutQueue_.EnQue<T>(xOutLocal);
        rstdQueue_.EnQue<float>(rstdLocal);
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileALen)
    {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        // 将 y 从 [R,A] 转回 [A,R]，只搬真实 A/R。
        LocalTensor<T> yTrans = yQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            CalcTransposeFp32(yLocal, yTrans, curTileALen);
        } else {
            CalcTransposeFp16(yLocal, yTrans, curTileALen);
        }
        yQueue_.FreeTensor(yLocal);
        yQueue_.EnQue<T>(yTrans);
        LocalTensor<T> yTransOut = yQueue_.DeQue<T>();
        DataCopyExtParams params;
        params.blockCount = curTileALen;
        params.blockLen = numCol * sizeof(T);
        uint64_t srcAlignLen = (numCol * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        params.srcStride = (rAligned * sizeof(T) - srcAlignLen) / BLOCK_SIZE;
        params.dstStride = 0;
        DataCopyPad(yGm[yGmOffset], yTransOut, params);
        yQueue_.FreeTensor(yTransOut);
    }

    __aicore__ inline void CopyOutX(int64_t xGmOffset, uint32_t curTileALen)
    {
        LocalTensor<T> xOutLocal = xOutQueue_.DeQue<T>();
        // x 与 y 使用相同的恢复转置过程。
        LocalTensor<T> xTrans = xOutQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            CalcTransposeFp32(xOutLocal, xTrans, curTileALen);
        } else {
            CalcTransposeFp16(xOutLocal, xTrans, curTileALen);
        }
        xOutQueue_.FreeTensor(xOutLocal);
        xOutQueue_.EnQue<T>(xTrans);
        LocalTensor<T> xTransOut = xOutQueue_.DeQue<T>();
        DataCopyExtParams params;
        params.blockCount = curTileALen;
        params.blockLen = numCol * sizeof(T);
        uint64_t srcAlignLen = (numCol * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        params.srcStride = (rAligned * sizeof(T) - srcAlignLen) / BLOCK_SIZE;
        params.dstStride = 0;
        DataCopyPad(xOutGm[xGmOffset], xTransOut, params);
        xOutQueue_.FreeTensor(xTransOut);
    }

    // FP32 每个元素占两个 B16 地址槽。
    __aicore__ inline void CalcTransposeFp32(const LocalTensor<T>& src, LocalTensor<T>& dst, uint32_t curTileALen)
    {
        int rRepeatTimes = ops::CeilDiv(static_cast<int64_t>(rAligned),
                                        static_cast<int64_t>(TRANS_FP32_ADDR_GROUP_SIZE));
        for (int i = 0; i < rRepeatTimes; i++) {
            TransDataTo5HDParams params;
            LocalTensor<T> srcLocalList[TRANS_ADDR_LIST_SIZE];
            LocalTensor<T> dstLocalList[TRANS_ADDR_LIST_SIZE];

            uint32_t aRepeatTimes = ops::CeilDiv(curTileALen, uint32_t(TRANS_ADDR_LIST_SIZE));
            params.repeatTimes = aRepeatTimes;
            params.srcRepStride = aRepeatTimes == 1 ? 0 : TRANS_B16_PER_B32;
            params.dstRepStride = aRepeatTimes == 1 ? 0 : TRANS_ADDR_LIST_SIZE * rRepeatTimes;
            for (int j = 0; j < TRANS_FP32_ADDR_GROUP_SIZE; j++) {
                uint32_t offset = TRANS_FP32_ADDR_GROUP_SIZE * tileALen * i + tileALen * j;
                srcLocalList[j] = src[offset];
                srcLocalList[j + TRANS_FP32_ADDR_GROUP_SIZE] = src[offset + TRANS_FP32_ADDR_GROUP_SIZE];
            }
            for (int j = 0; j < TRANS_FP32_ADDR_GROUP_SIZE; j++) {
                uint32_t offset = TRANS_FP32_ADDR_GROUP_SIZE * i + TRANS_FP32_ADDR_GROUP_SIZE * rRepeatTimes * j;
                dstLocalList[j * TRANS_B16_PER_B32] = dst[offset];
                dstLocalList[j * TRANS_B16_PER_B32 +
                             1] = dst[offset + TRANS_FP32_ADDR_GROUP_SIZE * TRANS_FP32_ADDR_GROUP_SIZE * rRepeatTimes];
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTransposeFp16(const LocalTensor<T>& src, LocalTensor<T>& dst, uint32_t curTileALen)
    {
        // FP16/BF16 每个元素占一个 B16 地址槽。
        int rRepeatTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), static_cast<int64_t>(rTileBase));
        uint32_t aRepeatTimes = ops::CeilDiv(curTileALen, static_cast<uint32_t>(rTileBase));
        for (int i = 0; i < rRepeatTimes; i++) {
            TransDataTo5HDParams params;
            params.repeatTimes = aRepeatTimes;
            params.srcRepStride = aRepeatTimes == 1 ? 0 : 1;
            params.dstRepStride = aRepeatTimes == 1 ? 0 : (rTileBase * rRepeatTimes);
            LocalTensor<T> srcLocalList[TRANS_ADDR_LIST_SIZE];
            LocalTensor<T> dstLocalList[TRANS_ADDR_LIST_SIZE];
            for (int j = 0; j < TRANS_ADDR_LIST_SIZE; j++) {
                uint32_t offset = rTileBase * tileALen * i + tileALen * j;
                srcLocalList[j] = src[offset];
            }
            for (int j = 0; j < TRANS_ADDR_LIST_SIZE; j++) {
                uint32_t offset = rTileBase * i + rAligned * j;
                dstLocalList[j] = dst[offset];
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CopyOutRstd(int64_t rstdOffset, uint32_t curTileALen)
    {
        // rstd 已按 A 连续，直接搬真实行数。
        LocalTensor<float> rstdLocal = rstdQueue_.DeQue<float>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = curTileALen * sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(rstdGm[rstdOffset], rstdLocal, params);
        rstdQueue_.FreeTensor(rstdLocal);
    }

private:
    TPipe* pPipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQueue_;
    TBuf<TPosition::VECCALC> xAddBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xOutGm;
    GlobalTensor<float> rstdGm;

    uint64_t numRow;
    uint64_t numCol;
    uint64_t rAligned;
    uint64_t rTileBase;
    uint64_t tileALen;
    float epsilon;
    float avgFactor;
};
} // namespace AddRmsNorm
#endif // ADD_RMS_NORM_REGBASE_TRANS_H
