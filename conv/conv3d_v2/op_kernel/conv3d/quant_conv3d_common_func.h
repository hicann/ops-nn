/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_conv3d_common_func.h
 * \brief
 */

#ifndef QUANT_CONV3D_COMMON_FUNC_H
#define QUANT_CONV3D_COMMON_FUNC_H

#include "conv3d_common_init_func.h"
#include "conv3d_common_set_func.h"

namespace Conv3dFunc {
CONV_DECLARE_REG_IMPL(BeginSetCrossFlag);
CONV_DECLARE_REG_IMPL(EndWaitCrossFlag);
CONV_DECLARE_REG_IMPL(SetWorkspace);
CONV_DECLARE_REG_IMPL(SetVecScale);
CONV_DECLARE_REG_IMPL(SetSubBlockIdx);
CONV_DECLARE_REG_IMPL(LoadTotalCoreChannel);
CONV_DECLARE_REG_IMPL(FreeTotalCoreChannel);
CONV_DECLARE_REG_IMPL(VecCompute);

template <class Intf, uint32_t ImplType>
struct VecCompute {
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output)
    {
        uint64_t mSize, nSize;
        uint64_t ws_startoffset = self->ctx.workspaceDbFlag * self->ctx.conv3dTiling->mL0 * self->ctx.conv3dTiling->nL0;
        self->ctx.copyOutIns.GetL0CSize(mSize, nSize);
        uint64_t n16num = nSize / BLOCK_L0_N;
        if (n16num > 1) {
            self->ctx.outMoffset = 0;
            if (self->ctx.subblockIdx) {
                uint64_t halfnum = CeilDIV(n16num, 2) * BLOCK_L0_N;
                nSize = nSize - halfnum;
                self->ctx.channelOffset = halfnum;
                ws_startoffset += halfnum * mSize;
                self->ctx.outNoffset = halfnum;
            } else {
                nSize = CeilDIV(n16num, 2) * BLOCK_L0_N;
            }
        } else {
            self->ctx.channelOffset = 0;
            self->ctx.outNoffset = 0;
            if (self->ctx.subblockIdx) {
                if (mSize > 1) {
                    uint64_t halfnum = CeilDIV(mSize, 2);
                    mSize = mSize - halfnum;
                    ws_startoffset += BLOCK_L0_N * halfnum;
                    self->ctx.outMoffset = halfnum;
                } else {
                    CrossCoreWaitFlag(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);
                    CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);
                    return false;
                }
            } else {
                mSize = CeilDIV(mSize, 2);
            }
        }
        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            CopyScaleAndBias(self, 0, nSize);
        }
        CrossCoreWaitFlag(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);
        DequantCompute(self, mSize, nSize, ws_startoffset, output);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);

        return false;
    }

    static __aicore__ inline void DequantCompute(Intf *self, uint64_t mSize, uint64_t nSize, uint64_t ws_startoffset,
                                                 const GlobalTensor<typename Intf::OutputT> &output)
    {
        uint32_t maxnUBIter = CeilDIV(nSize, self->ctx.conv3dTiling->nUB);
        uint32_t maxmUBIter = CeilDIV(mSize, self->ctx.conv3dTiling->mUB);
        uint16_t totalSrcStride = (mSize - self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;
        for (uint32_t nIter = 0; nIter < maxnUBIter - 1; nIter++) {
            if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
                CopyScaleAndBias(self, nIter, self->ctx.conv3dTiling->nUB);
            }
            for (uint32_t mIter = 0; mIter < maxmUBIter - 1; mIter++) {
                uint64_t srcOffset = ws_startoffset + (nIter * self->ctx.conv3dTiling->nUB * mSize +
                                    mIter * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
                CopyIn(self, self->ctx.totalBlockCount, self->ctx.totalBlockLen, totalSrcStride, srcOffset);
                oneVecCompute(self, self->ctx.conv3dTiling->mUB, self->ctx.conv3dTiling->nUB, mIter, nIter, output);
            }
            uint16_t blockLen =  (mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;
            uint16_t srcStride = (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N * self->ctx.sizeOfL0c / 32 ;
            uint64_t srcOffset = ws_startoffset + (nIter * self->ctx.conv3dTiling->nUB * mSize +
                                (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
            CopyIn(self, self->ctx.totalBlockCount, blockLen, srcStride, srcOffset);
            oneVecCompute(self, mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB, self->ctx.conv3dTiling->nUB, maxmUBIter - 1, nIter, output);
            if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
                FreeScaleAndBias(self);
            }
        }
        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            CopyScaleAndBias(self, maxnUBIter - 1, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB);
        }
        uint16_t blockCount = (nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB) / BLOCK_L0_N;
        for (uint32_t mIter = 0; mIter < maxmUBIter - 1; mIter++) {
            uint64_t srcOffset = ws_startoffset + (maxnUBIter-1) * self->ctx.conv3dTiling->nUB * mSize +
                                                    mIter * self->ctx.conv3dTiling->mUB * BLOCK_L0_N ;
            CopyIn(self, blockCount, self->ctx.totalBlockLen, totalSrcStride, srcOffset);
            oneVecCompute(self, self->ctx.conv3dTiling->mUB, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB, mIter, maxnUBIter - 1, output);
        }
        uint16_t blockLen =  (mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;
        uint16_t srcStride = (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N * self->ctx.sizeOfL0c / 32 ;
        uint64_t srcOffset = ws_startoffset + ((maxnUBIter-1) * self->ctx.conv3dTiling->nUB * mSize +
                                                    (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
        CopyIn(self, blockCount, blockLen, srcStride, srcOffset);
        oneVecCompute(self, mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB,
                        maxmUBIter - 1, maxnUBIter - 1, output);
        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0) ||
            self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            FreeScaleAndBias(self);
        }
    }

    static __aicore__ inline void CopyIn(Intf *self, uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint64_t srcOffset)
    {
        self->ctx.ubin = self->ctx.queueUBin.template AllocTensor<typename Intf::L0cT>();
        DataCopyParams copyinParams;
        copyinParams.blockCount = blockCount;
        copyinParams.blockLen =  blockLen;
        copyinParams.srcStride = srcStride;
        copyinParams.dstStride = 0;
        DataCopy(self->ctx.ubin, self->ctx.workspacegm[srcOffset], copyinParams);
        self->ctx.queueUBin.EnQue(self->ctx.ubin);
    }

    static __aicore__ inline void  CopyScaleAndBias(Intf *self, uint16_t nIter, uint16_t num)
    {
        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            LocalTensor<typename Intf::BiasT> bias = self->ctx.queueUBbias.template AllocTensor<typename Intf::BiasT>();
            CopyInChannel<typename Intf::BiasT>(self, bias, self->ctx.biasgm[self->ctx.channelOffset], nIter, num);
            self->ctx.queueUBbias.EnQue(bias);
            LocalTensor<typename Intf::FP32T> scale = self->ctx.queueUBscale.template AllocTensor<typename Intf::FP32T>();
            CopyInChannel<typename Intf::FP32T>(self, scale, self->ctx.scalegm[self->ctx.channelOffset], nIter, num);
            self->ctx.queueUBscale.EnQue(scale);

            LocalTensor<typename Intf::BiasT> biasLocal = self->ctx.queueUBbias.template DeQue<typename Intf::BiasT>();

            if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
                self->ctx.ubbias = self->ctx.fp32BiasBuf.template Get<typename Intf::FP32T>();
                Cast(self->ctx.ubbias, biasLocal, RoundMode::CAST_NONE, num);
                self->ctx.queueUBbias.FreeTensor(biasLocal);

            } else {
                self->ctx.ubbias = biasLocal;
            }
            self->ctx.ubscale = self->ctx.queueUBscale.template DeQue<typename Intf::FP32T>();
        }

        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            LocalTensor<typename Intf::BiasT> bias = self->ctx.queueUBbias.template AllocTensor<typename Intf::BiasT>();
            CopyInChannel<typename Intf::BiasT>(self, bias, self->ctx.biasgm[self->ctx.channelOffset], 0, num);
            self->ctx.queueUBbias.EnQue(bias);
            LocalTensor<typename Intf::BiasT> biasLocal = self->ctx.queueUBbias.template DeQue<typename Intf::BiasT>();

            if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
                self->ctx.ubbias = self->ctx.fp32BiasBuf.template Get<typename Intf::FP32T>();
                Cast(self->ctx.ubbias, biasLocal, RoundMode::CAST_NONE, num);
                self->ctx.queueUBbias.FreeTensor(biasLocal);

            } else {
                self->ctx.ubbias = biasLocal;
            }

            LocalTensor<typename Intf::FP32T> scale = self->ctx.queueUBscale.template AllocTensor<typename Intf::FP32T>();
            CopyInChannel<typename Intf::FP32T>(self, scale, self->ctx.scalegm[self->ctx.channelOffset], 0, num);
            self->ctx.queueUBscale.EnQue(scale);
            self->ctx.ubscale = self->ctx.queueUBscale.template DeQue<typename Intf::FP32T>();
        }
    }

    static __aicore__ inline void  FreeScaleAndBias(Intf *self)
    {
        if constexpr (!(IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value)) {
            self->ctx.queueUBbias.FreeTensor(self->ctx.ubbias);
        }
        self->ctx.queueUBscale.FreeTensor(self->ctx.ubscale);
    }

    template <typename DataTypeT>
    static __aicore__ inline void CopyInChannel(Intf *self, const LocalTensor<DataTypeT>& dst, const GlobalTensor<DataTypeT> &src, uint16_t nIter, uint16_t num)
    {
        DataCopyParams copyinParams;
        copyinParams.blockCount = 1;
        copyinParams.blockLen =  num * sizeof(DataTypeT) / 32;
        uint64_t Offset = self->ctx.copyOutIns.GetChannelOffset(nIter * self->ctx.conv3dTiling->nUB);
        DataCopy(dst, src[Offset], copyinParams);
    }

    static __aicore__ inline void MulScaleAddBias(uint16_t m, uint16_t n, const LocalTensor<typename Intf::FP32T>& src,
                                                 const LocalTensor<typename Intf::FP32T>& bias, const LocalTensor<typename Intf::FP32T>& scale)
    {
        uint64_t mask = BLOCK_L0_N;
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = BLOCK_L0_N * sizeof(typename Intf::FP32T) / 32;
        repeatParams.src0RepStride = BLOCK_L0_N * sizeof(typename Intf::FP32T) / 32;
        repeatParams.src1RepStride = 0;

        uint16_t maxNiter = n / BLOCK_L0_N;
        uint16_t maxMiter = CeilDIV(m, 255);
        for(uint16_t niter = 0; niter < maxNiter; niter++) {
            for (uint16_t miter = 0; miter < maxMiter - 1; miter++)
            {
                uint64_t offset = m * niter * BLOCK_L0_N + 255 * miter * BLOCK_L0_N;
                Mul(src[offset], src[offset], scale[niter * BLOCK_L0_N], mask, 255, repeatParams);
            }

            uint64_t offset = m * niter * BLOCK_L0_N + 255 * (maxMiter - 1) * BLOCK_L0_N;
            Mul(src[offset], src[offset], scale[niter * BLOCK_L0_N], mask, m - (maxMiter - 1) * 255, repeatParams);
        }

        PipeBarrier<PIPE_V>();

        for(uint16_t niter = 0; niter < maxNiter; niter++) {
            for (uint16_t miter = 0; miter < maxMiter - 1; miter++)
            {
                uint64_t offset = m * niter * BLOCK_L0_N + 255 * miter * BLOCK_L0_N;
                Add(src[offset], src[offset], bias[niter * BLOCK_L0_N], mask, 255, repeatParams);
            }
            uint64_t offset = m * niter * BLOCK_L0_N + 255 * (maxMiter - 1) * BLOCK_L0_N;
            Add(src[offset], src[offset], bias[niter * BLOCK_L0_N], mask, m - (maxMiter - 1) * 255, repeatParams);
        }
    }

    static __aicore__ inline void oneVecCompute(Intf *self, uint16_t m, uint16_t n, uint16_t mIter, uint16_t nIter, const GlobalTensor<typename Intf::OutputT> &output)
    {
        LocalTensor<typename Intf::L0cT> localUBin = self->ctx.queueUBin.template DeQue<typename Intf::L0cT>();
        LocalTensor<typename Intf::FP32T> dstLocal = localUBin.template ReinterpretCast<typename Intf::FP32T>();

        Cast(dstLocal, localUBin, AscendC::RoundMode::CAST_RINT, m * n);

        LocalTensor<typename Intf::FP32T> bias;
        LocalTensor<typename Intf::FP32T> scale;
        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            bias = self->ctx.ubbias;
            scale = self->ctx.ubscale;
        }

        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            bias = self->ctx.ubbias[nIter * self->ctx.conv3dTiling->nUB];
            scale = self->ctx.ubscale[nIter * self->ctx.conv3dTiling->nUB];
        }

        if (self->ctx.conv3dTiling->scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE)) {
            uint64_t Offset = self->ctx.copyOutIns.GetChannelOffset(nIter * self->ctx.conv3dTiling->nUB) + self->ctx.channelOffset;
            bias = self->ctx.ubbias[Offset];
            scale = self->ctx.ubscale[Offset];
        }
        PipeBarrier<PIPE_V>();
        MulScaleAddBias(m, n, dstLocal, bias, scale);

        //cast to bf16 or fp16
        self->ctx.ubout = self->ctx.queueUBout.template AllocTensor<typename Intf::OutputT>();
        PipeBarrier<PIPE_V>();
        Cast(self->ctx.ubout, dstLocal, AscendC::RoundMode::CAST_RINT, m * n);
        self->ctx.queueUBin.FreeTensor(localUBin);
        self->ctx.queueUBout.EnQue(self->ctx.ubout);
        LocalTensor<typename Intf::OutputT> copyubout = self->ctx.queueUBout.template DeQue<typename Intf::OutputT>();

        self->ctx.copyOutIns.CopyUBOut(output, mIter, nIter, m, n, copyubout);
        self->ctx.queueUBout.FreeTensor(copyubout);
    }
};

}  // namespace Conv3dFunc

#endif
