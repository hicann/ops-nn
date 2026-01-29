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
 * \file conv_bp_sub_func_load_gm_to_l1.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_ADVANCE_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_ADVANCE_H

#include "conv_bp_sub_func_mix.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Dn2NzParams;
using AscendC::Nd2NzParams;

namespace Convolution3DBackpropFunc {
template <class Intf>
__aicore__ inline void InitZeroValue(Intf *self, const LocalTensor<typename Intf::SrcT> &buf)
{
    uint32_t len = buf.GetSize() * sizeof(typename Intf::SrcT);
    if constexpr(std::is_same<typename Intf::SrcT, hifloat8_t>::value ||
        std::is_same<typename Intf::SrcT, fp8_e4m3fn_t>::value) {
        InitConstValue(buf.template ReinterpretCast<uint16_t>(), {1, static_cast<uint16_t>(len >> 5), 0, 0});
    } else {
        AscendC::InitConstValueParams<typename Intf::SrcT> initConstValueParams;
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = len >> 5;  // 除以blockSize
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = (typename Intf::SrcT)(0);
        InitConstValue(buf, initConstValueParams);
    }
    PipeBarrier<PIPE_MTE2>();
}

template <class Intf>
__aicore__ inline void LoadToB1Dn2Nz(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCout1Cout0Size = AlignUp16(curCoutSize);
    if (curCout1Cout0Size != curCoutSize) {
        InitZeroValue(self, useB1Buf);
    }

    // DN (cout, cin, dk, hk, wk) -> NZ (cin1, hk, wk, cout1, cout0, cin0) = (cin1, hkwk, cout_align, cin0)
    // B: cout, D: cin, N: hkwk
    // DN (B, D, N) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;

    dn2NzParams.dnNum = curCoutSize; // B: cout
    dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->cinG * self->ctx.dkHkWk_; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = self->ctx.tiling_->c0; // dst B stride (unit: element)

    dn2NzParams.dValue = curCinSize; // D: cin
    dn2NzParams.srcDValue = self->ctx.dkHkWk_; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCout1Cout0Size; // dst D stride (unit: 32B)
    dn2NzParams.nValue = self->ctx.singleShapeHWk_; // N: hkwk
    dn2NzParams.dstNzNStride = curCout1Cout0Size; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2Nz(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCout1Cout0Size = AlignUp16(curCoutSize);
    if (curCout1Cout0Size != curCoutSize) {
        InitZeroValue(self, useB1Buf);
    }

    // ND (cout, dk, hk, wk, cin) -> NZ (cin1, hk, wk, cout1, cout0, cin0) = (cin1, hkwk, cout_align, cin0)
    // B: cout, D: cin, N: hkwk
    // ND (B, N, D) -> (D1, N, B, D0)
    Nd2NzParams nd2NzParams;

    nd2NzParams.ndNum = curCoutSize; // B: cout
    nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->cinG * self->ctx.dkHkWk_; // src B stride (unit: element)
    nd2NzParams.dstNzMatrixStride = self->ctx.tiling_->c0; // dst B stride (unit: element)

    nd2NzParams.dValue = curCinSize; // D: cin
    nd2NzParams.srcDValue = self->ctx.tiling_->cinG; // src N stride (unit: element)
    nd2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCout1Cout0Size; // dst D stride (unit: 32B)
    nd2NzParams.nValue = self->ctx.singleShapeHWk_; // N: hkwk
    nd2NzParams.dstNzNStride = curCout1Cout0Size; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1TransposeForDkHkWkIsOne(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }

    // DN(ND) (cout, cin) -> NZ (cout1, cin1, cin0, cout0) = (cout1, cin_align, cout0)
    // B: 1, D: cout, N: cin
    // DN (B, D, N) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1; // B: 1
    dn2NzParams.srcDnMatrixStride = 1; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = 1; // dst B stride (unit: element)

    dn2NzParams.dValue = curCoutSize; // D: cout
    dn2NzParams.srcDValue = self->ctx.tiling_->cinG; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = curCin1Cin0Size; // dst D stride (unit: 32B)
    dn2NzParams.nValue = curCinSize; // N: cin
    dn2NzParams.dstNzNStride = 1; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Dn2NzTranspose(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }

    // DN (cout, cin, dk, hk, wk) -> NZ (cout1, hk, wk, cin1, cin0, cout0) = (cout1, hkwk, cin_align, cout0)
    // B: cin, D: cout, N: hkwk
    // DN (B, D, N) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;

    dn2NzParams.dnNum = curCinSize; // B: cin
    dn2NzParams.srcDnMatrixStride = self->ctx.dkHkWk_; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = self->ctx.tiling_->c0; // dst B stride (unit: element)

    dn2NzParams.dValue = curCoutSize; // D: cout
    dn2NzParams.srcDValue = self->ctx.tiling_->cinG * self->ctx.dkHkWk_; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCin1Cin0Size; // dst D stride (unit: 32B)
    dn2NzParams.nValue = self->ctx.singleShapeHWk_; // N: hkwk
    dn2NzParams.dstNzNStride = curCin1Cin0Size; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Dn2NzTransposeForKernelSplitH(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }

    // DN (cout, cin, dk, hk, wk) -> NZ (cout1, splitHk, splitwk, cin1, cin0, cout0) = (cout1, splithkwk, cin_align, cout0)
    // B: cin, D: cout, N: hkwk
    // DN (B, D, N) -> (D1, N, B, D0)
    uint32_t cinOffset = self->ctx.tiling_->cinG;
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = curCinSize; // B: cin
    dn2NzParams.srcDnMatrixStride = self->ctx.dkHkWk_; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = 1 << self->ctx.tiling_->c0BitsB; // dst B stride (unit: element)

    dn2NzParams.dValue = curCoutSize; // D: cout
    dn2NzParams.srcDValue = cinOffset * self->ctx.dkHkWk_; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.splitHkWkList_[self->ctx.splitIndex_] * curCin1Cin0Size; // dst D stride (unit: 32B)

    dn2NzParams.nValue = self->ctx.splitWkList_[self->ctx.splitIndex_]; // N: splitWk
    dn2NzParams.dstNzNStride = curCin1Cin0Size; // dst N stride (unit: 32B)

    uint64_t srcGmOffset = 0;
    uint32_t dstB1Offset = 0;
    // 当cout不全载时，在load2B1阶段完成子kernel重排，splitHk在外部循环
    for (uint32_t i = 0; i < self->ctx.splitHkList_[self->ctx.splitIndex_]; i++) {
        DataCopy(useB1Buf[dstB1Offset], self->ctx.weightGlobal_[out2B1SrcAddrOffset + srcGmOffset], dn2NzParams);
        srcGmOffset += static_cast<uint64_t>(self->ctx.tiling_->wk) * self->ctx.tiling_->strideH;
        dstB1Offset += self->ctx.splitWkList_[self->ctx.splitIndex_] * (dn2NzParams.dstNzNStride << self->ctx.tiling_->c0BitsB);
    }
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2NzForDkHkWkIsOne(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCout1Cout0Size = AlignUp16(curCoutSize);
    if (curCout1Cout0Size != curCoutSize) {
        InitZeroValue(self, useB1Buf);
    }

    // ND (cout, cin, dk) -> NZ (cin1, cout1, cout0, cin0) = (cin1, cout_align, cin0)
    // B: not use, N: cout, D: cin
    // DN (B, D, N) -> (D1, N, B, D0)
    Nd2NzParams nd2NzParams;

    nd2NzParams.ndNum = 1; // not use
    nd2NzParams.srcNdMatrixStride = 0;
    nd2NzParams.dstNzMatrixStride = 0;

    nd2NzParams.nValue = curCoutSize; // N: cout
    nd2NzParams.srcDValue = self->ctx.tiling_->cinG * self->ctx.dkHkWk_; // src N stride (unit: element)
    nd2NzParams.dstNzNStride = 1; // dst N stride (unit: 32B)
    nd2NzParams.dValue = curCinSize; // D: cin
    nd2NzParams.dstNzC0Stride = curCout1Cout0Size; // dst D stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2NzTranspose(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }

    // ND (cout, dk, hk, wk, cin) -> NZ (cout1, hk, wk, cin1, cin0, cout0) = (cout1, hkwk, cin_align, cout0)
    // B: hkwk, D: cout, N: cin
    // ND (B, N, D) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = self->ctx.singleShapeHWk_; // B: hkwk
    dn2NzParams.dValue = curCoutSize; // D: cout
    if (self->ctx.tiling_->enableVecTrans) {
        dn2NzParams.srcDnMatrixStride = AlignUp16(self->ctx.tiling_->cinG); // Cout, Dk, Hk, Wk, Cin1, Cin0
        dn2NzParams.srcDValue = AlignUp16(self->ctx.tiling_->cinG) * self->ctx.dkHkWk_; // Cout, Dk, Hk, Wk, Cin1, Cin0
    } else {
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->cinG; // src B stride (unit: element)
        dn2NzParams.srcDValue = self->ctx.tiling_->cinG * self->ctx.dkHkWk_; // src D stride (unit: element)
    }
    dn2NzParams.dstNzMatrixStride = curCin1Cin0Size << self->ctx.tiling_->c0BitsB; // dst B stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCin1Cin0Size; // dst D stride (unit: 32B)

    dn2NzParams.nValue = curCinSize; // N: cin
    dn2NzParams.dstNzNStride = 1; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2NzForKernelSplit(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    // ND (cout, dk, hk, wk, cin) -> NZ (cout1, splithk, splitwk, cin1, cin0, cout0) = (cout1, splithkwk, cin_align, cout0)
    // B: splitwk, D: cout, N: cin
    // ND (B, N, D) -> (D1, N, B, D0)
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }
    uint32_t strideW = self->ctx.tiling_->strideW;
    uint32_t cinOffset = self->ctx.tiling_->cinG;
    if (self->ctx.tiling_->enableVecTrans) {
        cinOffset = AlignUp16(self->ctx.tiling_->cinG);
    }
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        out2B1SrcAddrOffset += (self->ctx.splitHIndex_ * self->ctx.tiling_->wk + self->ctx.splitWIndex_) * cinOffset;
    } else {
        strideW = 1;
    }

    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = self->ctx.splitWkList_[self->ctx.splitIndex_]; // B: hkwk
    dn2NzParams.srcDnMatrixStride = cinOffset * strideW; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = curCin1Cin0Size << self->ctx.tiling_->c0BitsB; // dst B stride (unit: element)

    dn2NzParams.dValue = curCoutSize; // D: cout
    dn2NzParams.srcDValue = cinOffset * self->ctx.dkHkWk_; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.splitHkWkList_[self->ctx.splitIndex_] * curCin1Cin0Size; // dst D stride (unit: 32B)

    dn2NzParams.nValue = curCinSize; // N: cin
    dn2NzParams.dstNzNStride = 1; // dst N stride (unit: 32B)

    uint64_t srcGmOffset = 0;
    uint32_t dstB1Offset = 0;
    // 当cout不全载时，在load2B1阶段完成子kernel重排，splitHk在外部循环
    for (uint32_t i = 0; i < self->ctx.splitHkList_[self->ctx.splitIndex_]; i++) {
        DataCopy(useB1Buf[dstB1Offset], self->ctx.weightGlobal_[out2B1SrcAddrOffset + srcGmOffset], dn2NzParams);
        srcGmOffset += static_cast<uint64_t>(self->ctx.tiling_->wk) * cinOffset * self->ctx.tiling_->strideH;
        dstB1Offset += self->ctx.splitWkList_[self->ctx.splitIndex_] * dn2NzParams.dstNzMatrixStride;
    }
}

template <class Intf>
__aicore__ inline void LoadGmDataToB1ForDn2Nz(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_TRANSPOSE_ONLY) {
        LoadToB1Nd2NzForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    } else if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY) {
        LoadToB1Dn2Nz(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    } else {
        if (self->ctx.dkHkWk_ == 1) {
            LoadToB1TransposeForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else {
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
                LoadToB1Dn2NzTransposeForKernelSplitH(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
            } else {
                LoadToB1Dn2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
            }
        }
    }
}

template <class Intf>
__aicore__ inline void LoadGmDataToB1ForNd2Nz(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_TRANSPOSE_ONLY) {
        LoadToB1Nd2NzForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    } else if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY) {
        LoadToB1Nd2Nz(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    } else {
        if (self->ctx.dkHkWk_ == 1) {
            LoadToB1TransposeForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else {
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                if (!self->ctx.kSCoutFullLoad_) { // B1若未全载，则在loadB1时完成子kernel重排
                    LoadToB1Nd2NzForKernelSplit(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
                } else {
                    LoadToB1Nd2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
                }
            } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
                LoadToB1Nd2NzForKernelSplit(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
            } else {
                LoadToB1Nd2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
            }
        }
    }
}

template <class Intf>
__aicore__ inline void LoadGmDataToB1DHWCN2NzTranspose(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue(self, useB1Buf);
    }

    // DHWCN (dk, hk, wk, cin，cout) -> NZ (cout1, hk, wk, cin1, cin0, cout0) = (cout1, hkwk, cin_align, cout0)
    // B: cin, D: cout, N: hkwk
    // DHWCN (B, N, D) -> (D1, N, B, D0)
    Nd2NzParams nd2NzParams;
    nd2NzParams.ndNum = self->ctx.singleShapeHWk_; // B: hkwk
    nd2NzParams.dValue = curCoutSize; // D: cout
    nd2NzParams.nValue = curCinSize; // N: cin

    nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->cinG * self->ctx.tiling_->cout; // src B stride (unit: element)
    nd2NzParams.srcDValue = self->ctx.tiling_->cout; // src N stride (unit: element)

    nd2NzParams.dstNzMatrixStride = curCin1Cin0Size << self->ctx.tiling_->c0BitsB; // dst B stride (unit: element)
    nd2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCin1Cin0Size; // dst D stride (unit: 32B)
    nd2NzParams.dstNzNStride = 1; // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf, class src1_T>
__aicore__ inline void LoadGmDataToB1(Intf *self, uint32_t kIdx, uint32_t curDkIdx)
{
    LocalTensor<typename Intf::SrcT> useB1Buf = self->ctx.inQueL1B_.template AllocTensor<typename Intf::SrcT>();

    uint32_t curCinIdx = self->ctx.curNL1Idx_ * self->ctx.tiling_->stepN * self->ctx.tiling_->baseN;
    uint32_t curCinSize = CalcCurCinSizeB1(self, curCinIdx);
    uint32_t curCoutIdx = 0;
    uint32_t curCoutSize = 0;
    CalcCoutIndexAndSizeB1(self, kIdx, curCoutIdx, curCoutSize);

    // 1982 kernel gm shape: (cout, cin, dk, hk, wk)
    // because l1 not cut hk and wk, so,
    // srcAddrOffset = coutIdx * coutStride + cinIdx * cinStride + dkIdx *  dkStride
    if (self->ctx.tiling_->enableVecTrans) {
        uint64_t out2B1SrcAddrOffset = static_cast<uint64_t>(curCoutIdx) * AlignUp16(self->ctx.tiling_->cinG) * self->ctx.dkHkWk_ +
            curCinIdx + curDkIdx * self->ctx.hkWk_ * AlignUp16(self->ctx.tiling_->cinG);
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
            LoadToB1Nd2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else {
            LoadToB1Nd2NzForKernelSplit(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        }
    } else {
        if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
            uint64_t out2B1SrcAddrOffset = static_cast<uint64_t>(curCoutIdx) * self->ctx.tiling_->cinG * self->ctx.dkHkWk_ +
                curCinIdx * self->ctx.dkHkWk_ + curDkIdx * self->ctx.hkWk_ + self->ctx.curHkIdx_ * self->ctx.tiling_->wk + self->ctx.curWkIdx_;
            LoadGmDataToB1ForDn2Nz(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::DHWCN) {
            uint64_t out2B1SrcAddrOffset = curCoutIdx + static_cast<uint64_t>(curCinIdx) * self->ctx.tiling_->cout +
                (curDkIdx * self->ctx.hkWk_ + self->ctx.curHkIdx_ * self->ctx.tiling_->wk + self->ctx.curWkIdx_) *
                self->ctx.tiling_->cinG * self->ctx.tiling_->cout;
            LoadGmDataToB1DHWCN2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else { // NDHWC
            uint64_t out2B1SrcAddrOffset = static_cast<uint64_t>(curCoutIdx) * self->ctx.tiling_->cinG * self->ctx.dkHkWk_ +
                curCinIdx + (curDkIdx * self->ctx.hkWk_ + self->ctx.curHkIdx_ * self->ctx.tiling_->wk + self->ctx.curWkIdx_) *
                self->ctx.tiling_->cinG;
            LoadGmDataToB1ForNd2Nz(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        }
    }

    self->ctx.inQueL1B_.EnQue(useB1Buf);
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf *self, uint64_t kIdx, uint32_t curDkIdx, bool loadFlag)
{
    if (!loadFlag || unlikely(kIdx >= self->ctx.kIter_ || (self->ctx.isB1FullLoadFlag_ && !self->ctx.isLoadB1_))) {
        return;
    }

    if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
        if ASCEND_IS_AIV {
            GroupTransdataWeight<Intf>(self, kIdx, curDkIdx);
        }
        self->ctx.groupIterIdx_ += 1;
    } else if constexpr (Intf::conv3dConfig.enableC04Flag) {
        if ASCEND_IS_AIV {
            C04TransdataWeight<Intf>(self, kIdx, curDkIdx);
        }
        self->ctx.c04LoadToB1IterIdx_ += 1;
    } else {
        if ASCEND_IS_AIC {
            LoadGmDataToB1<Intf, typename Intf::SrcT>(self, kIdx, curDkIdx);
        }
    }
}
}  // namespace Convolution3DBackpropFunc

#endif
