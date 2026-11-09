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
 * \file conv_bp_input_sub_func_vector_intrinsics.h
 * \brief Vector instruction wrappers for interleave, gather and polyphase transforms
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_VECTOR_INTRINSICS_H
#define CONV3D_BP_INPUT_SUB_FUNC_VECTOR_INTRINSICS_H

#include "conv_bp_input_sub_func_utils.h"

namespace Convolution3DBackpropFunc {

template <class ReDstT>
__simd_vf__ inline void InterleaveKernelSplitNormal(__ubuf__ ReDstT* src0Ptr, __ubuf__ ReDstT* src1Ptr,
                                                    __ubuf__ ReDstT* dst0Ptr, __ubuf__ ReDstT* dst1Ptr, uint32_t vfLen,
                                                    uint32_t doubleVfLen, uint16_t repeatTimes)
{
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<ReDstT, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<ReDstT> src0;
    AscendC::Reg::RegTensor<ReDstT> src1;
    AscendC::Reg::RegTensor<ReDstT> dst0;
    AscendC::Reg::RegTensor<ReDstT> dst1;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::LoadAlign(src0, src0Ptr + i * vfLen);
        AscendC::Reg::LoadAlign(src1, src1Ptr + i * vfLen);
        // Interleave指令不支持hif8，需要伪装成uint8
        AscendC::Reg::Interleave(dst0, dst1, src0, src1);
        AscendC::Reg::StoreAlign(dst0Ptr + i * doubleVfLen, dst0, preg);
        AscendC::Reg::StoreAlign(dst1Ptr + i * doubleVfLen, dst1, preg);
    }
}

template <class ReDstT>
__simd_vf__ inline void InterleaveKernelSplit1x1(__ubuf__ ReDstT* src0Ptr, __ubuf__ ReDstT* dst0Ptr,
                                                 __ubuf__ ReDstT* dst1Ptr, uint32_t vfLen, uint32_t doubleVfLen,
                                                 uint16_t repeatTimes)
{
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<ReDstT, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<ReDstT> src0;
    AscendC::Reg::RegTensor<ReDstT> src1;
    AscendC::Reg::RegTensor<ReDstT> dst0;
    AscendC::Reg::RegTensor<ReDstT> dst1;
    ReDstT scalarValue = 0;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::LoadAlign(src0, src0Ptr + i * vfLen);
        AscendC::Reg::Duplicate(src1, scalarValue);
        AscendC::Reg::Interleave(dst0, dst1, src0, src1);
        AscendC::Reg::StoreAlign(dst0Ptr + i * doubleVfLen, dst0, preg);
        AscendC::Reg::StoreAlign(dst1Ptr + i * doubleVfLen, dst1, preg);
    }
}

template <class IndexT>
__simd_vf__ inline void ExpandGatherIdxByStride(__ubuf__ IndexT* idxAddr, uint16_t repeatTimes, uint16_t numPerRepeat,
                                                uint16_t initialDstOffset, uint32_t mask, IndexT cinStride)
{
    AscendC::Reg::RegTensor<IndexT> idxReg;
    AscendC::Reg::LoadAlign<IndexT>(idxReg, idxAddr);
    AscendC::Reg::MaskReg maskReg = AscendC::Reg::UpdateMask<IndexT>(mask);

    uint16_t dstOffset = initialDstOffset;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        // cinG * hk * wk * [0, 1, ..., c0 - 1] + (i + 1) * hk * wk
        AscendC::Reg::Adds<IndexT, IndexT>(idxReg, idxReg, cinStride, maskReg);
        AscendC::Reg::StoreAlign<IndexT>(idxAddr + dstOffset, idxReg, maskReg);
        dstOffset += numPerRepeat;
    }
}

/* gen vselr gather/scatter index rows from the seed row (c0PerReg entries):
   vselrIdx[i][c]   = i + c * hkWk (gathers [c * hkWk + i] from source UB)
   vscatterIdx[i][c] = i * scatterStride + c (scatters to [i * scatterStride + c] in dest UB)
   rows are c0PerReg entries, not 32B aligned when c0PerReg < 16, so store unaligned */
template <class IndexT>
__simd_vf__ inline void ExpandVselrIdxByStride(__ubuf__ IndexT* vselrIdxAddr, __ubuf__ IndexT* vscatterIdxAddr,
                                               uint16_t repeatTimes, uint16_t numPerRepeat, uint16_t initialDstOffset,
                                               uint32_t mask, IndexT vselrKStride, IndexT vscatterKStride)
{
    AscendC::Reg::RegTensor<IndexT> idxVselrReg;
    AscendC::Reg::RegTensor<IndexT> idxVscatterReg;
    AscendC::Reg::DataCopy<IndexT>(idxVselrReg, vselrIdxAddr);
    AscendC::Reg::DataCopy<IndexT>(idxVscatterReg, vscatterIdxAddr);
    AscendC::Reg::MaskReg maskReg = AscendC::Reg::UpdateMask<IndexT>(mask);
    AscendC::Reg::UnalignReg ureg0;
    AscendC::Reg::UnalignReg ureg1;

    vselrIdxAddr += initialDstOffset;
    vscatterIdxAddr += initialDstOffset;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::Reg::Adds<IndexT, IndexT>(idxVselrReg, idxVselrReg, vselrKStride, maskReg);
        AscendC::Reg::DataCopyUnAlign(vselrIdxAddr, idxVselrReg, ureg0, numPerRepeat);
        AscendC::Reg::DataCopyUnAlignPost(vselrIdxAddr, ureg0, 0);
        AscendC::Reg::Adds<IndexT, IndexT>(idxVscatterReg, idxVscatterReg, vscatterKStride, maskReg);
        AscendC::Reg::DataCopyUnAlign(vscatterIdxAddr, idxVscatterReg, ureg1, numPerRepeat);
        AscendC::Reg::DataCopyUnAlignPost(vscatterIdxAddr, ureg1, 0);
    }
}

template <class SrcBT, class IndexT>
__simd_vf__ inline void GatherDn2Nz4Group(__ubuf__ IndexT* idxAddr, __ubuf__ SrcBT* srcAddr, __ubuf__ SrcBT* dstAddr,
                                          uint16_t cout1G, uint16_t hkWk, uint16_t cin1GIterMax,
                                          uint32_t srcCout1GStride, uint32_t srcCin1GStride, uint32_t dstCout1GStride,
                                          uint32_t dstKStride, uint32_t dstCin1GStride)
{
    AscendC::Reg::RegTensor<SrcBT> gatherReg;
    AscendC::Reg::RegTensor<IndexT> idxReg;
    AscendC::Reg::MaskReg maskReg = AscendC::Reg::CreateMask<SrcBT, AscendC::Reg::MaskPattern::ALL>();

    // copy index from ub to reg
    AscendC::Reg::LoadAlign<IndexT>(idxReg, idxAddr);

    for (uint16_t cout1GIdx = 0; cout1GIdx < cout1G; ++cout1GIdx) {
        for (uint16_t kIdx = 0; kIdx < hkWk; ++kIdx) {
            for (uint16_t cin1GIdx = 0; cin1GIdx < cin1GIterMax; ++cin1GIdx) {
                uint32_t srcOffset = cout1GIdx * srcCout1GStride + cin1GIdx * srcCin1GStride + kIdx;
                uint32_t dstOffset = cout1GIdx * dstCout1GStride + kIdx * dstKStride + cin1GIdx * dstCin1GStride;

                // gather data from ub to reg according to gather index
                AscendC::Reg::Gather<SrcBT, SrcBT, IndexT>(gatherReg, srcAddr + srcOffset, idxReg, maskReg);
                // copy gather output data from reg to ub
                AscendC::Reg::StoreAlign<SrcBT>(dstAddr + dstOffset, gatherReg, maskReg);
            }
        }
    }
}

/* vselr Dn2Nz: for each nIndex (cin0), stream-read the ND staging and gather/scatter
   to the NZ staging. Data flow: ndTensor -> ureg -> srcReg -> (GatherImpl) dstReg -> (DataCopyScatter) dst.
   Each pass consumes srcCo0Stride source elements (auto-updated srcAddr) and scatters
   mask = c0PerReg * hkWk lanes by the vscatter index table. */
template <class SrcT, class IndexT>
__simd_vf__ inline void GatherScatterDn2Nz4Group(__ubuf__ IndexT* vselrIndexAddr, __ubuf__ IndexT* vscatterIndexAddr,
                                                 __ubuf__ SrcT* srcAddr, __ubuf__ SrcT* dstAddr, uint16_t nLoopTimes,
                                                 uint16_t co1LoopTimes, uint8_t passCnt, uint32_t mask,
                                                 uint32_t dstNStride, uint32_t srcCo0Stride, uint32_t dstCo0Stride,
                                                 uint32_t dstCo1Stride)
{
    AscendC::Reg::RegTensor<SrcT> srcReg;
    AscendC::Reg::RegTensor<SrcT> dstReg;
    AscendC::Reg::RegTensor<IndexT> vselrIndex;
    AscendC::Reg::RegTensor<IndexT> vscatterIndex;
    AscendC::Reg::DataCopy<IndexT>(vselrIndex, vselrIndexAddr);
    AscendC::Reg::DataCopy<IndexT>(vscatterIndex, vscatterIndexAddr);
    AscendC::Reg::MaskReg maskReg = AscendC::Reg::UpdateMask<IndexT>(mask);
    AscendC::Reg::UnalignReg ureg;

    AscendC::Reg::DataCopyUnAlignPre(ureg, srcAddr);
    __ubuf__ SrcT* dst = dstAddr;
    for (uint16_t nIndex = 0; nIndex < nLoopTimes; ++nIndex) {
        dst = dstAddr + nIndex * dstNStride;
        for (uint16_t co1Index = 0; co1Index < co1LoopTimes; ++co1Index) {
            for (uint8_t passIdx = 0; passIdx < passCnt; ++passIdx) {
                // Data flow: ndTensor -> ureg -> srcReg. Auto update srcAddr: srcAddr += srcCo0Stride
                AscendC::Reg::DataCopyUnAlign(srcReg, ureg, srcAddr, srcCo0Stride);
                // vselr not support fp32, so here use API: GatherImpl
                AscendC::Reg::GatherImpl(dstReg, srcReg, vselrIndex);
                // scatter gather output data from reg to ub according to scatter index
                AscendC::Reg::DataCopyScatter<SrcT, IndexT>(dst, dstReg, vscatterIndex, maskReg);
                dst += dstCo0Stride;
            }
            dst += dstCo1Stride;
        }
    }
}

template <class SrcBT, class IndexT>
__simd_vf__ inline void GatherDn2Nz4C04(__ubuf__ IndexT* idxAddr, __ubuf__ SrcBT* srcAddr, __ubuf__ SrcBT* dstAddr,
                                        __ubuf__ uint32_t* maskAddr, uint16_t k1, uint16_t n1IterMax,
                                        uint32_t srcN1Stride, uint32_t srcK1Stride, uint32_t dstN1Stride,
                                        uint32_t dstK1Stride)
{
    AscendC::Reg::RegTensor<SrcBT> gatherReg;
    AscendC::Reg::RegTensor<IndexT> idxReg;
    AscendC::Reg::MaskReg maskReg = AscendC::Reg::CreateMask<SrcBT, AscendC::Reg::MaskPattern::ALL>();
    // copy index from ub to reg
    AscendC::Reg::LoadAlign<IndexT>(idxReg, idxAddr);

    for (uint16_t k1Idx = k1; k1Idx > 1; --k1Idx) {
        for (uint16_t n1Idx = 0; n1Idx < n1IterMax; ++n1Idx) {
            uint32_t srcOffset = n1Idx * srcN1Stride + (k1Idx - 1) * srcK1Stride;
            uint32_t dstOffset = n1Idx * dstN1Stride + (k1 - k1Idx) * dstK1Stride;

            AscendC::Reg::Gather<SrcBT, SrcBT, IndexT>(gatherReg, srcAddr + srcOffset, idxReg, maskReg);
            AscendC::Reg::StoreAlign<SrcBT>(dstAddr + dstOffset, gatherReg, maskReg);
        }
    }

    AscendC::Reg::MaskReg tailMaskReg = AscendC::Reg::CreateMask<SrcBT, AscendC::Reg::MaskPattern::ALL>();
    // copy mask from ub to reg
    AscendC::Reg::LoadAlign<uint32_t>(tailMaskReg, maskAddr);
    for (uint16_t n1Idx = 0; n1Idx < n1IterMax; ++n1Idx) {
        uint32_t srcOffset = n1Idx * srcN1Stride;
        uint32_t dstOffset = n1Idx * dstN1Stride + (k1 - 1) * dstK1Stride;

        AscendC::Reg::Gather<SrcBT, SrcBT, IndexT>(gatherReg, srcAddr + srcOffset, idxReg, tailMaskReg);
        AscendC::Reg::StoreAlign<SrcBT>(dstAddr + dstOffset, gatherReg, maskReg);
    }
}

} // namespace Convolution3DBackpropFunc

#endif
