/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_RMS_NORM_REGBASE_SPLIT_AR_H
#define ADD_RMS_NORM_REGBASE_SPLIT_AR_H
#include <limits>
#include "add_rms_norm_regbase_common.h"
#include "../../rms_norm/rms_norm_base.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddRmsNorm {
using namespace AscendC;

// 每层 cache 占一个 UB block，层数覆盖 uint64_t 计数范围。
constexpr int64_t AR_RECOMPUTE_SUM_LEN = platform::GetUbBlockSize() / sizeof(float);
constexpr int64_t AR_VECTOR_LEN = platform::GetVRegSize() / sizeof(float);
constexpr uint16_t AR_CACHE_LEVEL_COUNT = static_cast<uint16_t>(std::numeric_limits<uint64_t>::digits);
// 保证 workspace 的 MTE3 写在屏障后的 MTE2 读之前完成。
constexpr SyncAllConfig SPLIT_AR_SYNC_CONFIG = {PIPE_MTE3, PIPE_MTE2};

// 商余式向上取整避免 value + divisor - 1 在大 R 下溢出整数范围。
__aicore__ inline uint64_t SplitARCeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

// 返回严格小于 value 的最大 2 的幂。
__aicore__ inline uint64_t GetReduceFoldPoint(uint64_t value)
{
    if (value <= 1) {
        return 0;
    }
    if (value <= 2) {
        return 1;
    }
    if (value <= 4) {
        return 2;
    }
    uint64_t reduced = value - 1;
    uint64_t power = std::numeric_limits<uint64_t>::digits - 1 - AscendC::ScalarCountLeadingZero(reduced);
    return static_cast<uint64_t>(1) << power;
}

// 返回二进制计数器插入第 idx+1 个 partial 时的进位层。
__aicore__ inline uint16_t GetCacheId(uint64_t idx)
{
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1);
}

// 在线合并被进位覆盖的低层 partial。
__aicore__ inline void UpdateCache(__ubuf__ float* cachePtr, __ubuf__ float* srcPtr, uint16_t cacheId)
{
    __VEC_SCOPE__
    {
        RegTensor<float> aReg;
        RegTensor<float> bReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(aReg, srcPtr);
        for (uint16_t j = 0; j < cacheId; ++j) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(bReg, cachePtr + j * AR_RECOMPUTE_SUM_LEN);
            AscendC::MicroAPI::Add(aReg, aReg, bReg, pregOne);
        }
        StoreAlign(cachePtr + cacheId * AR_RECOMPUTE_SUM_LEN, aReg, pregOne);
    }
}

__aicore__ inline void CopyPartial(__ubuf__ float* dstPtr, __ubuf__ float* srcPtr)
{
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        StoreAlign(dstPtr, srcReg, pregOne);
    }
}

__aicore__ inline void ClearPartialVector(__ubuf__ float* dstPtr, uint32_t count)
{
    uint16_t loopCount = static_cast<uint16_t>(SplitARCeilDiv(count, AR_VECTOR_LEN));
    uint32_t maskCount = count;
    __VEC_SCOPE__
    {
        RegTensor<float> zeroReg;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCount; ++i) {
            pregLoop = UpdateMask<float>(maskCount);
            Duplicate(zeroReg, 0.0f, pregLoop);
            StoreAlign(dstPtr + i * AR_VECTOR_LEN, zeroReg, pregLoop);
        }
    }
}

// srcPtr 按 block 对齐，dstPtr 允许只偏移一个 float。
__aicore__ inline void StorePartialElement(__ubuf__ float* dstPtr, __ubuf__ float* srcPtr)
{
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, srcReg, pregOne);
    }
}

/**
 * SplitAR 每核固定一个 R 分块并遍历全部 A 行。Pass1 写 [核,A] partial，SyncAll 后
 * 每核归并得到全部 rstd；Pass2 按 R tile 复用 gamma。ubFactor 循环支持任意大 R。
 */
template <typename T>
class KernelAddRmsNormRegBaseSplitAR {
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
    static constexpr uint32_t BLK_FP32 = BLOCK_SIZE / sizeof(float);

public:
    __aicore__ inline explicit KernelAddRmsNormRegBaseSplitAR(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                                GM_ADDR workspace, const AddRMSNormRegbaseSplitARTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        rBlockFactor = tiling->rBlockFactor;
        rBlockNum = tiling->rBlockNum;
        tailR = tiling->tailR;
        ubFactor = tiling->ubFactor;
        numRowAlign = tiling->numRowAlign;
        epsilon = tiling->epsilon;
        avgFactor = tiling->avgFactor;
        rBlockIdx = GetBlockIdx();
        curRStart = rBlockIdx * rBlockFactor;
        curRLen = (rBlockIdx == rBlockNum - 1) ? tailR : rBlockFactor;

        xGm1.SetGlobalBuffer((__gm__ T*)x1);
        xGm2.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        xOutGm.SetGlobalBuffer((__gm__ T*)x);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma + curRStart, curRLen);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd, numRow);
        wsGm.SetGlobalBuffer((__gm__ float*)workspace);

        // 准入保证每核跨 A 累计至少 8 次 tile 迭代，五个数据队列固定使用双缓冲。
        pPipe->InitBuffer(x1Queue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(x2Queue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(gammaQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(yQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(xOutQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        // ReduceSum<RA> 的源起址及可写尾部均按 VL 对齐。
        uint64_t combElementCount = SplitARCeilDiv(rBlockNum * numRowAlign, VL_FP32) * VL_FP32;
        pPipe->InitBuffer(combQueue, 1, combElementCount * sizeof(float));
        pPipe->InitBuffer(partQueue, 1, numRowAlign * sizeof(float));
        pPipe->InitBuffer(cacheBuf, AR_RECOMPUTE_SUM_LEN * AR_CACHE_LEVEL_COUNT * sizeof(float));
        pPipe->InitBuffer(workBuf, (ubFactor + VL_FP32) * sizeof(float));
        pPipe->InitBuffer(rstdQueue, 1, numRowAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        DoPass1PartialReduce();
        LocalTensor<float> part = partQueue.DeQue<float>();
        WritePartial(part);
        // 跨核 workspace 写后读同步。
        SyncAll<true, SPLIT_AR_SYNC_CONFIG>();
        ComputeCrossCoreRstd(part);
        LocalTensor<float> rstdLocal = rstdQueue.DeQue<float>();
        if (rBlockIdx == 0) {
            CopyOutRstd(rstdLocal);
        }
        DoPass2Output(rstdLocal);
        rstdQueue.FreeTensor(rstdLocal);
        partQueue.FreeTensor(part);
    }

private:
    __aicore__ inline void DoPass1PartialReduce()
    {
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> part = partQueue.AllocTensor<float>();
        LocalTensor<float> workLocal = workBuf.Get<float>();
        LocalTensor<float> cache = cacheBuf.Get<float>();
        __ubuf__ float* cachePtr = (__ubuf__ float*)cache.GetPhyAddr();
        __ubuf__ float* partPtr = (__ubuf__ float*)part.GetPhyAddr();
        __ubuf__ float* tilePartPtr = (__ubuf__ float*)workLocal.GetPhyAddr() + ubFactor;
        LocalTensor<float> tilePart = workLocal[ubFactor];
        ClearPartialVector(partPtr, static_cast<uint32_t>(numRowAlign));

        uint64_t tileCount = SplitARCeilDiv(curRLen, ubFactor);
        uint64_t paddedTileCount = tileCount;
        if ((tileCount & (tileCount - 1)) != 0) {
            paddedTileCount = GetReduceFoldPoint(tileCount) * 2;
        }

        // Pass1 先遍历 A，再遍历当前行的 R tile。
        for (uint64_t row = 0; row < numRow; ++row) {
            for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint64_t tileOffset = tileIdx * ubFactor;
                uint32_t curTileLen = static_cast<uint32_t>(curRLen - tileOffset < ubFactor ? curRLen - tileOffset :
                                                                                              ubFactor);
                uint64_t foldPoint = curTileLen == ubFactor ? ubFactor / 2 : GetReduceFoldPoint(curTileLen);
                ProcessPass1Tile(row, tileOffset, curTileLen, foldPoint, xFp32, tilePart, workLocal);
                UpdateCache(cachePtr, tilePartPtr, GetCacheId(tileIdx));
            }
            // 追加零 partial 到下一次幂，直接形成唯一根节点，避免额外的 cache 扫描归并。
            if (paddedTileCount != tileCount) {
                ClearPartialVector(tilePartPtr, static_cast<uint32_t>(AR_RECOMPUTE_SUM_LEN));
                for (uint64_t tileIdx = tileCount; tileIdx < paddedTileCount; ++tileIdx) {
                    UpdateCache(cachePtr, tilePartPtr, GetCacheId(tileIdx));
                }
            }
            uint64_t resultCacheOffset = GetCacheId(paddedTileCount - 1) * AR_RECOMPUTE_SUM_LEN;
            CopyPartial(tilePartPtr, cachePtr + resultCacheOffset);
            StorePartialElement(partPtr + row, tilePartPtr);
        }
        partQueue.EnQue(part);
    }

    __aicore__ inline void ProcessPass1Tile(uint64_t row, uint64_t tileOffset, uint32_t curTileLen, uint64_t foldPoint,
                                            LocalTensor<float>& xFp32, LocalTensor<float>& part,
                                            LocalTensor<float>& workLocal)
    {
        CopyInX(row, tileOffset, curTileLen);
        ComputeAdd(xFp32, curTileLen);
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(xFp32, part, workLocal, 1,
                                                                       static_cast<uint32_t>(ubFactor), curTileLen,
                                                                       static_cast<uint32_t>(foldPoint), BLK_FP32);
        if constexpr (IsSameType<T, float>::value) {
            CopyOutX(row, tileOffset, curTileLen);
        }
    }

    __aicore__ inline void WritePartial(const LocalTensor<float>& part)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(numRowAlign * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm[GetBlockIdx() * numRowAlign], part, params);
    }

    __aicore__ inline void CopyInX(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        uint64_t gmOffset = row * numCol + curRStart + tileOffset;
        DataCopyPad(x1Local, xGm1[gmOffset], params, pad);
        DataCopyPad(x2Local, xGm2[gmOffset], params, pad);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeAdd(LocalTensor<float>& xFp32, uint32_t curTileLen)
    {
        LocalTensor<T> x1Local = x1Queue.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.DeQue<T>();
        __ubuf__ T* x1Ptr = (__ubuf__ T*)x1Local.GetPhyAddr();
        __ubuf__ T* x2Ptr = (__ubuf__ T*)x2Local.GetPhyAddr();
        __ubuf__ float* xFp32Ptr = (__ubuf__ float*)xFp32.GetPhyAddr();
        LocalTensor<T> xOutLocal;
        __ubuf__ T* xOutPtr = nullptr;
        if constexpr (IsSameType<T, float>::value) {
            xOutLocal = xOutQueue.AllocTensor<T>();
            xOutPtr = (__ubuf__ T*)xOutLocal.GetPhyAddr();
        }
        uint16_t loopCount = static_cast<uint16_t>((curTileLen + VL_FP32 - 1) / VL_FP32);
        // UpdateMask 会递减 maskCount，必须在 VL 循环外持久保存。
        uint32_t maskCount = curTileLen;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> xSumReg;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                LoadRegForDtype<T>(x1Ptr, x1Reg, pregLoop, offset);
                LoadRegForDtype<T>(x2Ptr, x2Reg, pregLoop, offset);
                Add(xSumReg, x1Reg, x2Reg, pregLoop);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32Ptr + offset, xSumReg, pregLoop);
                if constexpr (IsSameType<T, float>::value) {
                    StoreRegForDtype<T>(xOutPtr, xSumReg, pregLoop, offset);
                }
            }
        }
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        if constexpr (IsSameType<T, float>::value) {
            xOutQueue.EnQue(xOutLocal);
        }
    }

    __aicore__ inline void CopyOutX(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        LocalTensor<T> xOutLocal = xOutQueue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        uint64_t gmOffset = row * numCol + curRStart + tileOffset;
        DataCopyPad(xOutGm[gmOffset], xOutLocal, params);
        xOutQueue.FreeTensor(xOutLocal);
    }

    __aicore__ inline void ComputeRstdFromSum(LocalTensor<float>& sumLocal)
    {
        LocalTensor<float> rstdLocal = rstdQueue.AllocTensor<float>();
        NormCommon::ComputeRstdNewtonRaphson<true, true>(sumLocal, rstdLocal, static_cast<uint32_t>(numRow), epsilon,
                                                         avgFactor, VL_FP32);
        rstdQueue.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyOutRstd(const LocalTensor<float>& rstdLocal)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(numRow * sizeof(float)), 0, 0, 0};
        DataCopyPad(rstdGm, rstdLocal, params);
    }

    __aicore__ inline void ComputeCrossCoreRstd(LocalTensor<float>& part)
    {
        // 每核读回 [rBlockNum,numRowAlign]，沿核维归并全部 A 行。
        LocalTensor<float> comb = combQueue.AllocTensor<float>();
        DataCopyExtParams cpr{1, static_cast<uint32_t>(rBlockNum * numRowAlign * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padr{false, 0, 0, 0};
        DataCopyPad(comb, wsGm, cpr, padr);

        combQueue.EnQue(comb);
        comb = combQueue.DeQue<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(rBlockNum), static_cast<uint32_t>(numRowAlign)};
        ReduceSum<float, Pattern::Reduce::RA, true>(part, comb, srcShape, false);
        combQueue.FreeTensor(comb);
        ComputeRstdFromSum(part);
    }

    __aicore__ inline void DoPass2Output(const LocalTensor<float>& rstdLocal)
    {
        __ubuf__ float* rstdPtr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        uint64_t tileCount = SplitARCeilDiv(curRLen, ubFactor);
        // Pass2 先遍历 R tile，再遍历 A，以跨行复用 gamma。
        for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            uint64_t tileOffset = tileIdx * ubFactor;
            uint32_t curTileLen = static_cast<uint32_t>(curRLen - tileOffset < ubFactor ? curRLen - tileOffset :
                                                                                          ubFactor);
            CopyInGamma(tileOffset, curTileLen);
            LocalTensor<T> gammaLocal = gammaQueue.DeQue<T>();
            for (uint64_t row = 0; row < numRow; ++row) {
                CopyInOutputData(row, tileOffset, curTileLen);
                LocalTensor<T> x1Local = x1Queue.DeQue<T>();
                LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
                if constexpr (IsSameType<T, float>::value) {
                    ComputeOutputFromXOut(x1Local, gammaLocal, yLocal, rstdPtr + row, curTileLen);
                } else {
                    LocalTensor<T> x2Local = x2Queue.DeQue<T>();
                    LocalTensor<T> xOutLocal = xOutQueue.AllocTensor<T>();
                    ComputeOutputFromInputs(x1Local, x2Local, gammaLocal, yLocal, xOutLocal, rstdPtr + row, curTileLen);
                    x2Queue.FreeTensor(x2Local);
                    xOutQueue.EnQue(xOutLocal);
                }
                x1Queue.FreeTensor(x1Local);
                yQueue.EnQue(yLocal);
                CopyOutY(row, tileOffset, curTileLen);
                if constexpr (!IsSameType<T, float>::value) {
                    CopyOutX(row, tileOffset, curTileLen);
                }
            }
            gammaQueue.FreeTensor(gammaLocal);
        }
    }

    __aicore__ inline void CopyInOutputData(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        if constexpr (IsSameType<T, float>::value) {
            LocalTensor<T> xLocal = x1Queue.AllocTensor<T>();
            DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            uint64_t gmOffset = row * numCol + curRStart + tileOffset;
            DataCopyPad(xLocal, xOutGm[gmOffset], params, pad);
            x1Queue.EnQue(xLocal);
        } else {
            // 16 位输出必须基于 FP32 add，且 Inplace 场景不能提前覆盖 x1。
            CopyInX(row, tileOffset, curTileLen);
        }
    }

    __aicore__ inline void CopyInGamma(uint64_t tileOffset, uint32_t curTileLen)
    {
        LocalTensor<T> gammaLocal = gammaQueue.AllocTensor<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(gammaLocal, gammaGm[tileOffset], params, pad);
        gammaQueue.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeOutputFromXOut(const LocalTensor<T>& xLocal, const LocalTensor<T>& gammaLocal,
                                                 LocalTensor<T>& yLocal, __ubuf__ float* rstdPtr, uint32_t curTileLen)
    {
        __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
        __ubuf__ T* gammaPtr = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();
        uint16_t loopCount = static_cast<uint16_t>((curTileLen + VL_FP32 - 1) / VL_FP32);
        // 将本核 rstd 广播后，以 FP32 计算并按 T 写回。
        uint32_t maskCount = curTileLen;

        __VEC_SCOPE__
        {
            RegTensor<float> xSumReg;
            RegTensor<float> gammaReg;
            RegTensor<float> yReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                LoadRegForDtype<T>(xPtr, xSumReg, pregLoop, offset);
                LoadRegForDtype<T>(gammaPtr, gammaReg, pregLoop, offset);
                Mul(yReg, xSumReg, rstdReg, pregLoop);
                Mul(yReg, yReg, gammaReg, pregLoop);
                StoreRegForDtype<T>(yPtr, yReg, pregLoop, offset);
            }
        }
    }

    __aicore__ inline void ComputeOutputFromInputs(const LocalTensor<T>& x1Local, const LocalTensor<T>& x2Local,
                                                   const LocalTensor<T>& gammaLocal, LocalTensor<T>& yLocal,
                                                   LocalTensor<T>& xOutLocal, __ubuf__ float* rstdPtr,
                                                   uint32_t curTileLen)
    {
        __ubuf__ T* x1Ptr = (__ubuf__ T*)x1Local.GetPhyAddr();
        __ubuf__ T* x2Ptr = (__ubuf__ T*)x2Local.GetPhyAddr();
        __ubuf__ T* gammaPtr = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ T* xOutPtr = (__ubuf__ T*)xOutLocal.GetPhyAddr();
        uint16_t loopCount = static_cast<uint16_t>((curTileLen + VL_FP32 - 1) / VL_FP32);
        uint32_t maskCount = curTileLen;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> xSumReg;
            RegTensor<float> gammaReg;
            RegTensor<float> yReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                LoadRegForDtype<T>(x1Ptr, x1Reg, pregLoop, offset);
                LoadRegForDtype<T>(x2Ptr, x2Reg, pregLoop, offset);
                Add(xSumReg, x1Reg, x2Reg, pregLoop);
                LoadRegForDtype<T>(gammaPtr, gammaReg, pregLoop, offset);
                Mul(yReg, xSumReg, rstdReg, pregLoop);
                Mul(yReg, yReg, gammaReg, pregLoop);
                StoreRegForDtype<T>(yPtr, yReg, pregLoop, offset);
                StoreRegForDtype<T>(xOutPtr, xSumReg, pregLoop, offset);
            }
        }
    }

    __aicore__ inline void CopyOutY(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        // 仅搬出当前 tile 的真实长度。
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        uint64_t gmOffset = row * numCol + curRStart + tileOffset;
        DataCopyPad(yGm[gmOffset], yLocal, params);
        yQueue.FreeTensor(yLocal);
    }

private:
    TPipe* pPipe;
    TQue<QuePosition::VECIN, 1> x1Queue;
    TQue<QuePosition::VECIN, 1> x2Queue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> xOutQueue;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> workBuf;
    TQue<QuePosition::VECOUT, 1> partQueue;
    TQue<QuePosition::VECIN, 1> combQueue;
    TBuf<TPosition::VECCALC> cacheBuf;
    TQue<QuePosition::VECOUT, 1> rstdQueue;

    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xOutGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> wsGm;

    uint64_t numRow;
    uint64_t numCol;
    uint64_t rBlockFactor;
    uint64_t rBlockNum;
    uint64_t tailR;
    uint64_t ubFactor;
    uint64_t numRowAlign;
    float epsilon;
    float avgFactor;

    uint64_t rBlockIdx;
    uint64_t curRStart;
    uint64_t curRLen;
};
} // namespace AddRmsNorm
#endif // ADD_RMS_NORM_REGBASE_SPLIT_AR_H
