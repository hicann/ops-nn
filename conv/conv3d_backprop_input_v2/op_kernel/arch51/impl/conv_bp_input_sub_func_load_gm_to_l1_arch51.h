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
 * \file conv_bp_input_sub_func_load_gm_to_l1_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_ARCH51_H

#include "conv_bp_input_sub_func_utils_arch51.h"
#include "conv_bp_input_sub_func_index_calc_arch51.h"

using AscendC::Dn2NzParams;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void LoadToB1Nd2Nz(Intf* self, uint32_t curCinSize, uint32_t curCoutSize,
                                     uint64_t out2B1SrcAddrOffset, const LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    uint32_t curCout1Cout0Size = AlignUp16(curCoutSize);
    if (curCout1Cout0Size != curCoutSize) {
        InitZeroValue<Intf, typename Intf::SrcBT>(self, useB1Buf);
    }

    // ND (cout, dk, hk, wk, cin) -> NZ (cin1, hk, wk, cout1, cout0, cin0) = (cin1, hkwk, cout_align, cin0)
    // B: cout, D: cin, N: hkwk
    // ND (B, N, D) -> (D1, N, B, D0)
    Nd2NzParams nd2NzParams;

    nd2NzParams.ndNum = curCoutSize;                                                     // B: cout
    nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->cinG * self->ctx.tiling_->dkHkWk; // src B stride (unit: element)
    nd2NzParams.dstNzMatrixStride = self->ctx.tiling_->c0;                               // dst B stride (unit: element)

    nd2NzParams.dValue = curCinSize;                                           // D: cin
    nd2NzParams.srcDValue = self->ctx.tiling_->cinG;                           // src N stride (unit: element)
    nd2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCout1Cout0Size; // dst D stride (unit: 32B)
    nd2NzParams.nValue = self->ctx.singleShapeHWk_;                            // N: hkwk
    nd2NzParams.dstNzNStride = curCout1Cout0Size;                              // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1TransposeForDkHkWkIsOne(Intf* self, uint32_t curCinSize, uint32_t curCoutSize,
                                                       uint64_t out2B1SrcAddrOffset,
                                                       const LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue<Intf, typename Intf::SrcBT>(self, useB1Buf);
    }

    // DN(ND) (cout, cin) -> NZ (cout1, cin1, cin0, cout0) = (cout1, cin_align, cout0)
    // B: 1, D: cout, N: cin
    // DN (B, D, N) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;             // B: 1
    dn2NzParams.srcDnMatrixStride = 1; // src B stride (unit: element)
    dn2NzParams.dstNzMatrixStride = 1; // dst B stride (unit: element)

    dn2NzParams.dValue = curCoutSize;                // D: cout
    dn2NzParams.srcDValue = self->ctx.tiling_->cinG; // src D stride (unit: element)
    dn2NzParams.dstNzC0Stride = curCin1Cin0Size;     // dst D stride (unit: 32B)
    dn2NzParams.nValue = curCinSize;                 // N: cin
    dn2NzParams.dstNzNStride = 1;                    // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2NzForDkHkWkIsOne(Intf* self, uint32_t curCinSize, uint32_t curCoutSize,
                                                   uint64_t out2B1SrcAddrOffset,
                                                   const LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    uint32_t curCout1Cout0Size = AlignUp16(curCoutSize);
    if (curCout1Cout0Size != curCoutSize) {
        InitZeroValue<Intf, typename Intf::SrcBT>(self, useB1Buf);
    }

    // ND (cout, cin, dk) -> NZ (cin1, cout1, cout0, cin0) = (cin1, cout_align, cin0)
    // B: not use, N: cout, D: cin
    // DN (B, D, N) -> (D1, N, B, D0)
    Nd2NzParams nd2NzParams;

    nd2NzParams.ndNum = 1; // not use
    nd2NzParams.srcNdMatrixStride = 0;
    nd2NzParams.dstNzMatrixStride = 0;

    nd2NzParams.nValue = curCoutSize;                                            // N: cout
    nd2NzParams.srcDValue = self->ctx.tiling_->cinG * self->ctx.tiling_->dkHkWk; // src N stride (unit: element)
    nd2NzParams.dstNzNStride = 1;                                                // dst N stride (unit: 32B)
    nd2NzParams.dValue = curCinSize;                                             // D: cin
    nd2NzParams.dstNzC0Stride = curCout1Cout0Size;                               // dst D stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf>
__aicore__ inline void LoadToB1Nd2NzTranspose(Intf* self, uint32_t curCinSize, uint32_t curCoutSize,
                                              uint64_t out2B1SrcAddrOffset,
                                              const LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    if (curCin1Cin0Size != curCinSize) {
        InitZeroValue<Intf, typename Intf::SrcBT>(self, useB1Buf);
    }

    // ND (cout, dk, hk, wk, cin) -> NZ (cout1, hk, wk, cin1, cin0, cout0) = (cout1, hkwk, cin_align, cout0)
    // B: hkwk, D: cout, N: cin
    // ND (B, N, D) -> (D1, N, B, D0)
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = self->ctx.singleShapeHWk_; // B: hkwk
    dn2NzParams.dValue = curCoutSize;              // D: cout
    if (self->ctx.tiling_->enableVecTrans) {
        dn2NzParams.srcDnMatrixStride = AlignUp16(self->ctx.tiling_->cinG); // Cout, Dk, Hk, Wk, Cin1, Cin0
        dn2NzParams.srcDValue = AlignUp16(self->ctx.tiling_->cinG) *
                                self->ctx.tiling_->dkHkWk; // Cout, Dk, Hk, Wk, Cin1, Cin0
    } else {
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->cinG;                     // src B stride (unit: element)
        dn2NzParams.srcDValue = self->ctx.tiling_->cinG * self->ctx.tiling_->dkHkWk; // src D stride (unit: element)
    }
    dn2NzParams.dstNzMatrixStride = curCin1Cin0Size << self->ctx.tiling_->c0BitsB; // dst B stride (unit: element)
    dn2NzParams.dstNzC0Stride = self->ctx.singleShapeHWk_ * curCin1Cin0Size;       // dst D stride (unit: 32B)

    dn2NzParams.nValue = curCinSize; // N: cin
    dn2NzParams.dstNzNStride = 1;    // dst N stride (unit: 32B)

    DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf, class src1_T>
__aicore__ inline void LoadGmDataToB1(Intf* self, uint32_t kIdx, uint32_t curDkIdx)
{
    LocalTensor<typename Intf::SrcBT> useB1Buf = self->ctx.inQueL1B_.template AllocTensor<typename Intf::SrcBT>();

    uint32_t curCinIdx = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
    uint32_t curCinSize = CalcCurCinSizeB1(self, curCinIdx);
    uint32_t curCoutIdx = 0;
    uint32_t curCoutSize = 0;
    CalcCoutIndexAndSizeB1<Intf>(self, kIdx, curCoutIdx, curCoutSize);

    uint64_t out2B1SrcAddrOffset = static_cast<uint64_t>(curCoutIdx) * self->ctx.tiling_->cinG *
                                       self->ctx.tiling_->dkHkWk +
                                   curCinIdx +
                                   (curDkIdx * self->ctx.tiling_->hkWk + self->ctx.curHkIdx_ * self->ctx.tiling_->wk +
                                    self->ctx.curWkIdx_) *
                                       self->ctx.tiling_->cinG;
    // B1 布局须与 LoadToB2 读取方式（CalcLoadB2Condition）匹配，镜像上游 LoadGmDataToB1ForNd2Nz 的分发
    uint8_t cond = CalcLoadB2Condition<Intf>(self);
    if (unlikely(cond == B2_TRANSPOSE_ONLY)) { // dkHkWk == 1
        LoadToB1Nd2NzForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    } else if (unlikely(cond == B2_REVERSE_ONLY)) {
        if (self->ctx.tiling_->dkHkWk == 1) { // 按 CalcLoadB2Condition 逻辑此分支不可达，镜像上游防御性保留
            LoadToB1TransposeForDkHkWkIsOne(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        } else {
            LoadToB1Nd2NzTranspose(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
        }
    } else { // B2_TRANSPOSE_AND_REVERSE（含 B2_NO_TRANSPOSE_NO_REVERSE，同走 ND2NZ 布局）
        LoadToB1Nd2Nz(self, curCinSize, curCoutSize, out2B1SrcAddrOffset, useB1Buf);
    }

    self->ctx.inQueL1B_.EnQue(useB1Buf);
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf* self, uint64_t kIdx, uint32_t curDkIdx, bool loadFlag)
{
    if (!loadFlag || unlikely(kIdx >= self->ctx.kIter_ || (self->ctx.isB1FullLoadFlag_ && !self->ctx.isLoadB1_))) {
        return;
    }

    LoadGmDataToB1<Intf, typename Intf::SrcBT>(self, kIdx, curDkIdx);
}
} // namespace Convolution3DBackpropFunc

#endif
