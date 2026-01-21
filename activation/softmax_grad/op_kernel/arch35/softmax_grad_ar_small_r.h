/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file SOFTMAX_GRAD_AR_SMALL_R.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_AR_SMALL_R_H
#define SOFTMAX_GRAD_AR_SMALL_R_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "softmax_grad_base.h"

namespace SoftmaxGradOps
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T>
class SoftmaxGradARSmallR
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr int64_t DATA_BLOCK_COUNT = 16;
    static constexpr int64_t DATA_BLOCK_COUNT_HALF = 8;
    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline SoftmaxGradARSmallR(){};

    __aicore__ inline SoftmaxGradARSmallR(TPipe* pipeIn)
    {
        pipe_ = pipeIn;
    }

    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, const SoftmaxGradARSmallRTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;

        x0Gm_.SetGlobalBuffer((__gm__ T*)x0);
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        int64_t xShapeLen = tilingData_->tileA0Len * tilingData_->rAligned;
        pipe_->InitBuffer(x0Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));

        pipe_->InitBuffer(xTmpBuf_, BUFFER_NUM * xShapeLen * sizeof(float));
        pipe_->InitBuffer(xSumBuffer_, tilingData_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;
        tmpLocal_ = xTmpBuf_.Get<float>();

        int64_t xOffset = beginIdx * tilingData_->tileA0Len * tilingData_->totalRLen;
        CopyInAndTransPose(xOffset, (beginIdx == tilingData_->totalTiles - 1) ? tilingData_->tileA0Tail : tilingData_->tileA0Len, tilingData_->totalRLen);
        int curIdx;
        uint32_t curTileA0Len = tilingData_->tileA0Len;
        uint32_t nextTileA0Len = tilingData_->tileA0Len;
        for (curIdx = beginIdx; curIdx < endIdx - 1; curIdx++) {
            if (curIdx + 1 == (tilingData_->totalTiles - 1)) {
                nextTileA0Len = tilingData_->tileA0Tail;
            }
            int64_t xOffsetPreLoad = (curIdx + 1) * tilingData_->tileA0Len * tilingData_->totalRLen;
            xOffset = curIdx * tilingData_->tileA0Len * tilingData_->totalRLen;
            LocalTensor<T> x0Tensor = x0Queue_.DeQue<T>();
            LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();

            __local_mem__ T* x0Local = (__local_mem__ T*)x0Tensor.GetPhyAddr();
            __local_mem__ T* x1Local = (__local_mem__ T*)x1Tensor.GetPhyAddr();
            CalcReduceSum(x0Local , x1Local, curTileA0Len);
            CopyInAndTransPose(xOffsetPreLoad, nextTileA0Len, tilingData_->totalRLen);
            x1Queue_.FreeTensor(x1Tensor);
            CalcOutput(x0Local, curTileA0Len);
            CalcTranspose(curTileA0Len, tilingData_->rAligned);
            x0Queue_.FreeTensor(x0Tensor);

            CopyOutY(xOffset, curTileA0Len);
        }
        if (curIdx == (tilingData_->totalTiles - 1)) {
            curTileA0Len = tilingData_->tileA0Tail;
        }
        xOffset = curIdx * tilingData_->tileA0Len * tilingData_->totalRLen;
        LocalTensor<T> x0Tensor = x0Queue_.DeQue<T>();
        LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();
        __local_mem__ T* x0Local = (__local_mem__ T*)x0Tensor.GetPhyAddr();
        __local_mem__ T* x1Local = (__local_mem__ T*)x1Tensor.GetPhyAddr();
        CalcReduceSum(x0Local , x1Local, curTileA0Len);
        x1Queue_.FreeTensor(x1Tensor);
        CalcOutput(x0Local, curTileA0Len);
        CalcTranspose(curTileA0Len, tilingData_->rAligned);
        x0Queue_.FreeTensor(x0Tensor);
        CopyOutY(xOffset, curTileA0Len);
    }

private:
    __aicore__ inline void CalcReduceSum(const __local_mem__ T* x0Local, const __local_mem__ T* x1Local,
                                         uint32_t curTileA0Len)
    {
        __local_mem__ float* tmpAddr = (__local_mem__ float*)tmpLocal_.GetPhyAddr();
        __local_mem__ float* tmpAddr2 = (__local_mem__ float*)tmpLocal_[tilingData_->tileA0Len * tilingData_->rAligned].GetPhyAddr();
 
        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(tilingData_->totalRLen);
        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeT(x0Local, x0Reg, pregMask, xOffset);
                    LoadTensorForDtypeT(x1Local, x1Reg, pregMask, xOffset);

                    Mul(x0Reg, x0Reg, x1Reg, pregMask);
                    DataCopy(tmpAddr + xOffset, x0Reg, pregMask);
                    DataCopy(tmpAddr2 + xOffset, x0Reg, pregMask);
                }
            }
        }

        xSumTensor_ = xSumBuffer_.Get<float>();
        uint32_t srcShape[2] = {uint32_t(tilingData_->totalRLen), uint32_t(tilingData_->tileA0Len)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(xSumTensor_, tmpLocal_, srcShape, false);
    }

    __aicore__ inline void CalcOutput(const __local_mem__ T* x0Local, uint32_t curTileA0Len)
    {
        __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();
        __local_mem__ float* tmpAddr2 = (__local_mem__ float*)tmpLocal_[tilingData_->tileA0Len * tilingData_->rAligned].GetPhyAddr();
        tmpLocalTy_ = tmpLocal_.template ReinterpretCast<T>();
        __local_mem__ T* tmpAddrTy = (__local_mem__ T*)tmpLocalTy_.GetPhyAddr();

        uint16_t curTileRLenVl = static_cast<uint16_t>(tilingData_->totalRLen);
        uint16_t loopA0Num = static_cast<uint16_t>(ops::CeilDiv(curTileA0Len, VL_FP32));

        __VEC_SCOPE__
        {
            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;
            RegTensor<float> sumReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;
            uint32_t tileA0LenLocal = tilingData_->tileA0Len;
            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, (__local_mem__ float*)xSumLocal + k * VL_FP32);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0LenLocal + k * VL_FP32;
                    LoadTensorForDtypeT(x0Local, x0Reg, pregMask, xOffset);

                    DataCopy(x1Reg, tmpAddr2 + xOffset);
                    Mul(x0Reg, x0Reg, sumReg, pregMask);
                    Sub(x0Reg, x1Reg, x0Reg, pregMask);

                    if constexpr (xToFp32_) {
                        MicroAPI::DataCopy(tmpAddrTy + xOffset, x0Reg, pregMask);
                    } else {  // fp16、bf16
                        RegTensor<T> xFp16;
                        MicroAPI::Cast<T, float, castTraitFp32ToFp16>(xFp16, x0Reg, pregMask);
                        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(tmpAddrTy + xOffset, xFp16, pregMask);
                    }
                }
            }
        }
    }
    
    __aicore__ inline void CalcTransposeFp32(const LocalTensor<T>& yLocal, uint32_t curTileA0Len, uint32_t rAligned)
    {
        int rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), DATA_BLOCK_COUNT_HALF);
        for (int i = 0; i < rRepeartTimes; i++) {
            TransDataTo5HDParams params;
            LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];

            uint32_t aRepeartTimes = ops::CeilDiv(curTileA0Len, uint32_t(DATA_BLOCK_COUNT));
            params.repeatTimes = aRepeartTimes;
            params.srcRepStride = aRepeartTimes == 1 ? 0 : CONST_TWO;
            params.dstRepStride = aRepeartTimes == 1 ? 0 : DATA_BLOCK_COUNT * rRepeartTimes;
            for (int j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
                uint32_t offset = DATA_BLOCK_COUNT_HALF * tilingData_->tileA0Len * i + tilingData_->tileA0Len * j;
                srcLocalList[j] = tmpLocalTy_[offset];
                srcLocalList[j + DATA_BLOCK_COUNT_HALF] = tmpLocalTy_[offset + DATA_BLOCK_COUNT_HALF];
            }
            for (int j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
                uint32_t offset = DATA_BLOCK_COUNT_HALF * i + DATA_BLOCK_COUNT_HALF * rRepeartTimes * j;
                dstLocalList[j * CONST_TWO] = yLocal[offset];
                dstLocalList[j * CONST_TWO + 1] = yLocal[offset + DATA_BLOCK_COUNT_HALF * DATA_BLOCK_COUNT_HALF * rRepeartTimes];
            }

            AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTransposeFp16(LocalTensor<T> yLocal, uint32_t curTileA0Len, uint32_t rAligned)
    {
        int rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), tilingData_->rTileBase);
        for (int i = 0; i < rRepeartTimes; i++) {
            TransDataTo5HDParams params;
            int64_t aRepeartTimes = ops::CeilDiv(static_cast<int64_t>(curTileA0Len), int64_t(tilingData_->rTileBase));
            params.repeatTimes = aRepeartTimes;
            params.srcRepStride = aRepeartTimes == 1 ? 0 : 1;
            params.dstRepStride = aRepeartTimes == 1 ? 0 : (tilingData_->rTileBase * rRepeartTimes);
            LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];
            for (int j = 0; j < DATA_BLOCK_COUNT; j++) {
                uint32_t offset = tilingData_->rTileBase * tilingData_->tileA0Len * i + tilingData_->tileA0Len * j;
                srcLocalList[j] = tmpLocalTy_[offset];
            }
            for (int j = 0; j < DATA_BLOCK_COUNT; j++) {
                uint32_t offset = tilingData_->rTileBase * i + rAligned * j; 
                dstLocalList[j] = yLocal[offset];
            }

            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTranspose(uint32_t curTileA0Len, uint32_t rAligned)
    {
        LocalTensor<T> yLocal_ = yQueue_.AllocTensor<T>();

        if constexpr (xToFp32_) {
            CalcTransposeFp32(yLocal_, curTileA0Len, rAligned);
        } else {
            CalcTransposeFp16(yLocal_, curTileA0Len, rAligned);
        }

        yQueue_.EnQue(yLocal_);
    }

    __aicore__ inline void LoadTensorForDtypeT(const __local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                               uint32_t offset)
    {
        if constexpr (xToFp32_) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else {  // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
            Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyInAndTransPose(int64_t xGmOffset, uint32_t curTileA0Len, uint32_t totalRLen)
    {
        static constexpr MultiCopyConfig config = {false};
        MultiCopyLoopInfo<CONST_TWO> copyLoopInfo;
        copyLoopInfo.loopSrcStride[0] = 1;
        copyLoopInfo.loopSrcStride[1] = totalRLen;   
        copyLoopInfo.loopDstStride[0] = tilingData_->tileA0Len;  
        copyLoopInfo.loopDstStride[1] = 1;
        copyLoopInfo.loopSize[0] = totalRLen;    
        copyLoopInfo.loopSize[1] = curTileA0Len;   
        MultiCopyParams<T, CONST_TWO> params = {copyLoopInfo, 0};

        LocalTensor<T> x0Local_ = x0Queue_.AllocTensor<T>();
        DataCopy<T, CONST_TWO, config>(x0Local_, x0Gm_[xGmOffset], params);
        x0Queue_.EnQue(x0Local_);

        LocalTensor<T> x1Local_ = x1Queue_.AllocTensor<T>();
        DataCopy<T, CONST_TWO, config>(x1Local_, x1Gm_[xGmOffset], params);
        x1Queue_.EnQue(x1Local_);
    }

    __aicore__ inline void CopyOutFp32(const LocalTensor<T>& yLocal, int64_t yGmOffset, uint32_t curTileA0Len)
    {
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = curTileA0Len;
        copyOutParams.blockLen = tilingData_->totalRLen * sizeof(T);
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
        DataCopyPad(yGm_[yGmOffset], yLocal[0], copyOutParams);
    }

    __aicore__ inline void CopyOutFp16(const LocalTensor<T>& yLocal, int64_t yGmOffset, uint32_t curTileA0Len)
    {
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = curTileA0Len;
        copyOutParams.blockLen =  tilingData_->totalRLen * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm_[yGmOffset], yLocal[0], copyOutParams);
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileA0Len)
    {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();

        if constexpr (xToFp32_) {
            CopyOutFp32(yLocal, yGmOffset, curTileA0Len);
        } else {
            CopyOutFp16(yLocal, yGmOffset, curTileA0Len);
        }

        yQueue_.FreeTensor(yLocal);
    }

private:
    const SoftmaxGradARSmallRTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> x0Queue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;

    TBuf<TPosition::VECCALC> xSumBuffer_;
    TBuf<TPosition::VECCALC> xTmpBuf_;

    LocalTensor<float> xSumTensor_;

    LocalTensor<float> tmpLocal_;
    LocalTensor<T> tmpLocalTy_;
    LocalTensor<T> yLocal_;

    static constexpr bool xToFp32_ = IsSameType<T, float>::value;
};
}  // namespace SoftmaxGradOps

#endif