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
 * \file conv_bp_filter_sub_func_load_a1.h
 * \brief A1 matrix load functions for conv3d backprop filter
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_LOAD_A1_H
#define CONV3D_BP_FILTER_SUB_FUNC_LOAD_A1_H
#include "conv_bp_filter_sub_func_load_common.h"

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void LoadToA1Fp32Nd2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    if (likely(!self->ctx.isSplitWo_)) {
        // fp32 时，使用nd2nz实现DN规格，主要是将D轴变为HoWo，N轴变为Cout
        /* B N D ==> D1 N B D0 */
        /* (HoWo), Cout ==> (HoWo)/C8, (Cout1, Cout0), C8 */
        nd2NzParams.ndNum = 1;
        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            nd2NzParams.nValue = self->ctx.tailM_;
        } else {
            nd2NzParams.nValue = self->ctx.tiling_->baseM; // N:Cout
        }

        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK
            nd2NzParams.dValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            nd2NzParams.dValue = self->ctx.kal1_; // D
        }

        nd2NzParams.srcDValue = self->ctx.dhwO_; // D_src_stride(loop1_src_stride)
        nd2NzParams.srcNdMatrixStride = 1;       // B_src_stride(loop4_src_stride)
        nd2NzParams.dstNzMatrixStride = 1;       // B_dst_stride(loop4_dst_stride)
        // D1_dst_stride(loop3_dst_stride)
        nd2NzParams.dstNzC0Stride =
            ShiftMulM0(ShiftCeilM0(nd2NzParams.nValue, self->ctx.tiling_->m0), self->ctx.tiling_->m0);
        nd2NzParams.dstNzNStride = 1; // N_dst_stride(loop2_dst_stride)
        return;
    }

    SplitWLoadToA1Nd2Nz(self, kaIdx, params, kaStepIdx, nd2NzParams);
}

template <class Intf>
static __aicore__ inline void SplitWLoadToA1Dn2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams)
{
    uint32_t hoStartIdx = kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_;
    uint32_t kal1Use = (self->ctx.stepKaRound == (kaStepIdx + 1)) ?
                           (self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - kaIdx * self->ctx.tiling_->baseK) :
                           (self->ctx.kal1_);
    uint32_t hoEndIdx = Ceil(kaIdx * self->ctx.tiling_->baseK + kal1Use, self->ctx.singleShapeWo_);
    uint32_t hoCopyLen = hoEndIdx - hoStartIdx;

    dn2NzParams.dnNum = hoCopyLen;
    dn2NzParams.nValue = self->ctx.singleShapeWo_; // NCDHW
    if (params.isLastMAL1) {
        // 最后一块mAL1，需要考虑tailM
        dn2NzParams.dValue = self->ctx.tailM_;
    } else {
        dn2NzParams.dValue = self->ctx.tiling_->baseM;
    }
    dn2NzParams.srcDValue = self->ctx.dhwO_;
    dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wo;

    // dstNzC0Stride需要用nValue * dnNum 对齐k0
    dn2NzParams.dstNzC0Stride =
        ShiftCeilM0(dn2NzParams.nValue * dn2NzParams.dnNum, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    dn2NzParams.dstNzNStride = 1;
    // NDC1HWC0排布，Matrix数量配置的是H轴，目的Matrix间隔为W*C0，即nValue * k0;
    dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;
    // 由于切wo存在多搬情况，需要修正alignedL1UseKa_,防止load2d时计算的src_stride错误
    self->ctx.alignedL1UseKa_ = dn2NzParams.dstNzC0Stride;
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp16Dn2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams)
{
    if (likely(!self->ctx.isSplitWo_)) {
        dn2NzParams.dnNum = 1;
        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            dn2NzParams.nValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            dn2NzParams.nValue = self->ctx.kal1_;
        }

        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            dn2NzParams.dValue = self->ctx.tailM_;
        } else {
            dn2NzParams.dValue = self->ctx.tiling_->baseM;
        }

        dn2NzParams.srcDValue = self->ctx.dhwO_;
        // 由于dnNum为1，可以不配置srcDnMatrixStride
        dn2NzParams.dstNzC0Stride =
            ShiftCeilChannelSize<Intf>(dn2NzParams.nValue, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;
        dn2NzParams.dstNzNStride = 1;
        // 由于dnNum为1，可以不配置// B_dst_stride(loop4_dst_stride)
        dn2NzParams.dstNzMatrixStride = 1;
        return;
    }

    SplitWLoadToA1Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp32Dn2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams)
{
    if (likely(!self->ctx.isSplitWo_)) {
        // fp32 时，使用dn2nz实现ND规格，主要是将D轴变为HoWo，N轴变为Cout
        /* B D N ==> D1 N B D0 */
        /* Cout, (HoWo), 1 ==> (HoWo)/C8, 1, (Cout1, Cout0), C8 */
        dn2NzParams.dnNum = 1;
        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            dn2NzParams.nValue = self->ctx.tailM_;
        } else {
            dn2NzParams.nValue = self->ctx.tiling_->baseM; // B:Cout
        }

        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            dn2NzParams.dValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            dn2NzParams.dValue = self->ctx.kal1_; // D
        }

        dn2NzParams.srcDValue = self->ctx.tiling_->cout; // D_src_stride(loop1_src_stride)
        dn2NzParams.srcDnMatrixStride = 1;               // B_src_stride(loop4_src_stride)
        dn2NzParams.dstNzMatrixStride = 1;               // B_dst_stride(loop4_dst_stride)
        // D1_dst_stride(loop3_dst_stride)
        dn2NzParams.dstNzC0Stride = ShiftCeilM0(dn2NzParams.nValue, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        dn2NzParams.dstNzNStride = 1; // N_dst_stride(loop2_dst_stride)
        return;
    }

    SplitWLoadToA1Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1Nd2NzNormal(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    nd2NzParams.ndNum = 1;
    if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
        // 最后一块kAL1，考虑tailK, 32表示32Byte
        nd2NzParams.nValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
    } else {
        nd2NzParams.nValue = self->ctx.kal1_;
    }

    if (params.isLastMAL1) {
        // 最后一块mAL1，需要考虑tailM
        nd2NzParams.dValue = self->ctx.tailM_;
    } else {
        nd2NzParams.dValue = self->ctx.tiling_->baseM;
    }

    nd2NzParams.srcDValue = self->ctx.tiling_->cout;
    nd2NzParams.dstNzC0Stride =
        ShiftCeilChannelSize<Intf>(nd2NzParams.nValue, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = 1;
}

template <class Intf>
static __aicore__ inline void SplitWLoadToA1Nd2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    uint32_t hoStartIdx = kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_;
    uint32_t kal1Use = (self->ctx.stepKaRound == (kaStepIdx + 1)) ?
                           (self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - kaIdx * self->ctx.tiling_->baseK) :
                           (self->ctx.kal1_);
    uint32_t hoEndIdx = Ceil(kaIdx * self->ctx.tiling_->baseK + kal1Use, self->ctx.singleShapeWo_);
    uint32_t hoCopyLen = hoEndIdx - hoStartIdx;

    nd2NzParams.ndNum = hoCopyLen;
    nd2NzParams.nValue = self->ctx.singleShapeWo_;
    if (params.isLastMAL1) {
        // 最后一块mAL1，需要考虑tailM
        nd2NzParams.dValue = self->ctx.tailM_;
    } else {
        nd2NzParams.dValue = self->ctx.tiling_->baseM;
    }
    nd2NzParams.srcDValue = self->ctx.tiling_->cout;
    nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->wo * self->ctx.tiling_->cout;

    // dstNzC0Stride需要用nValue * dnNum 对齐m0
    nd2NzParams.dstNzC0Stride =
        ShiftCeilM0(nd2NzParams.nValue * nd2NzParams.ndNum, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    nd2NzParams.dstNzNStride = 1;
    // NDC1HWC0排布，Matrix数量配置的是H轴，目的Matrix间隔为W*C0，即nValue * k0;
    nd2NzParams.dstNzMatrixStride = nd2NzParams.nValue * self->ctx.tiling_->k0;
    // 由于切wo存在多搬情况，需要修正alignedL1UseKa_,防止load2d时计算的src_stride错误
    self->ctx.alignedL1UseKa_ = nd2NzParams.dstNzC0Stride;
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp16Nd2Nz(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    if (likely(!self->ctx.isSplitWo_)) {
        LoadToA1Nd2NzNormal(self, kaIdx, params, kaStepIdx, nd2NzParams);
    } else {
        SplitWLoadToA1Nd2Nz(self, kaIdx, params, kaStepIdx, nd2NzParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadToA1Dn2NzBaseUseMEqualOne(
    Intf* self, const Out2L1ScalarParams& params, uint64_t kaStepIdx, uint32_t totalSize,
    LocalTensor<typename Intf::SrcT>& useA1Buf)
{
    uint32_t calCount = (totalSize * sizeof(typename Intf::SrcT) / 32) * (32 / sizeof(typename Intf::SrcT));
    uint32_t tailCount = totalSize - calCount;
    uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + kaStepIdx * self->ctx.kal1_;

    if (calCount > 0) {
        DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], calCount);
    }
    if (tailCount > 0) {
        Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = 1;
        dn2NzParams.nValue = 1;
        dn2NzParams.dValue = tailCount; // 元素个数
        dn2NzParams.srcDnMatrixStride = 1;
        dn2NzParams.srcDValue = 1;
        dn2NzParams.dstNzC0Stride = 1;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = 1;
        out2A1SrcAddrOffset = out2A1SrcAddrOffset + calCount;
        DataCopy(useA1Buf[calCount], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
    }
    return;
}

template <class Intf>
static __aicore__ inline void LoadToA1Nd2NzBaseUseMEqualOne(
    Intf* self, const Out2L1ScalarParams& params, uint64_t kaStepIdx, uint32_t totalSize,
    LocalTensor<typename Intf::SrcT>& useA1Buf)
{
    uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + kaStepIdx * self->ctx.kal1_ * self->ctx.tiling_->cout;
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;
    dn2NzParams.nValue = 1;
    dn2NzParams.dValue = totalSize; // 元素个数
    dn2NzParams.srcDnMatrixStride = 1;
    dn2NzParams.srcDValue = self->ctx.tiling_->cout;
    dn2NzParams.dstNzC0Stride = 1;
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 1;
    DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1BaseUseMEqualOne(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx,
    LocalTensor<typename Intf::SrcT>& useA1Buf)
{
    self->ctx.alignedL1UseM_ = params.isLastMAL1 ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
    uint32_t totalSize;
    if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
        // 最后一块kAL1，考虑tailK
        totalSize = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
    } else {
        totalSize = self->ctx.kal1_;
    }

    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        LoadToA1Dn2NzBaseUseMEqualOne(self, params, kaStepIdx, totalSize, useA1Buf);
    } else {
        LoadToA1Nd2NzBaseUseMEqualOne(self, params, kaStepIdx, totalSize, useA1Buf);
    }
}

template <class Intf>
static __aicore__ inline void LoadToA1ForTransFormat(
    Intf* self, uint64_t kaIdx, const Out2L1ScalarParams& params, uint64_t kaStepIdx,
    LocalTensor<typename Intf::SrcT>& useA1Buf)
{
    if (self->ctx.baseUseM_ == 1) {
        LoadToA1BaseUseMEqualOne(self, kaIdx, params, kaStepIdx, useA1Buf);
        return;
    }
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        uint64_t offset = kaStepIdx * self->ctx.kal1_;
        if (unlikely(self->ctx.isSplitWo_)) {
            offset = (kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_ * self->ctx.tiling_->wo);
        }
        uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + offset;
        if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
            if (likely(!self->ctx.isSplitWo_)) {
                // fp32 时，使用nd2nz实现DN规格，主要是将D轴变为HoWo，N轴变为Cout
                Nd2NzParams nd2NzParams;
                LoadToA1Fp32Nd2Nz<Intf>(self, kaIdx, params, kaStepIdx, nd2NzParams);
                DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
            } else {
                Dn2NzParams dn2NzParams;
                LoadToA1Fp32Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
                DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
            }
        } else {
            Dn2NzParams dn2NzParams;
            LoadToA1Fp16Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
            DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
        }
    } else {
        uint64_t offset = kaStepIdx * self->ctx.kal1_ * self->ctx.tiling_->cout;
        if (unlikely(self->ctx.isSplitWo_)) {
            offset = (kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_ * self->ctx.tiling_->wo) *
                        self->ctx.tiling_->cout;
        }
        uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + offset;
        if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
            if (likely(!self->ctx.isSplitWo_)) {
                // fp32 时，使用dn2nz实现ND规格，主要是将D轴变为HoWo，N轴变为Cout
                Dn2NzParams dn2NzParams;
                LoadToA1Fp32Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
                DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
            } else {
                Nd2NzParams nd2NzParams;
                LoadToA1Fp32Nd2Nz<Intf>(self, kaIdx, params, kaStepIdx, nd2NzParams);
                DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
            }

        } else {
            Nd2NzParams nd2NzParams;
            LoadToA1Fp16Nd2Nz<Intf>(self, kaIdx, params, kaStepIdx, nd2NzParams);
            DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
        }
    }
}

} // namespace ConvolutionBackpropFunc

#endif
