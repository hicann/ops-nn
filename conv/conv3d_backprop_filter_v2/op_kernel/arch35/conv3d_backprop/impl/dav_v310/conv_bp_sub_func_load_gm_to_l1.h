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

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1_H

namespace ConvolutionBackpropFunc {
using AscendC::Dn2NzParams;
using AscendC::Nd2NzParams;

struct Out2L1ScalarParams {
    // to L1A
    bool isLoad2L1A{false};
    bool isLastMAL1{false};
    bool isFreeAL1{false};
    uint64_t out2A1SrcAddr{0};

    // to L1B
    bool isLoad2L1B{false};
    bool isFreeBL1{false};
    uint64_t out2B1SrcAddr{0};
    uint32_t singleShapeHi{0};
    uint32_t singleShapeWi{0};
};

__aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b)
{
    if (b == 0) {
        // 处理除数为0的情况，直接返回a
        return a;
    }

    return (a + b - 1) / b;
}

template <class Intf>
__aicore__ inline void InitZeroValue(Intf *self, const LocalTensor<typename Intf::SrcT> &buf)
{
    uint32_t len = buf.GetSize() * sizeof(typename Intf::SrcT);
    constexpr int32_t SHIFT_BITS_5 = 5;
    if constexpr(std::is_same<typename Intf::SrcT, hifloat8_t>::value ||
        std::is_same<typename Intf::SrcT, fp8_e4m3fn_t>::value) {
        InitConstValue(buf.template ReinterpretCast<uint16_t>(), {1, static_cast<uint16_t>(len >> SHIFT_BITS_5), 0, 0});
    } else {
        AscendC::InitConstValueParams<typename Intf::SrcT> initConstValueParams;
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = len >> SHIFT_BITS_5;  // 除以blockSize
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = (typename Intf::SrcT)(0);
        InitConstValue(buf, initConstValueParams);
    }
    PipeBarrier<PIPE_MTE2>();
}

template <class Intf>
static __aicore__ inline void CalOut2B1ParamsBaseNUndivided(Intf *self, Out2L1ScalarParams& params)
{
    // 计算L1上cin起始idx, 去掉cin1HkWkCin里的HkWk
    uint64_t b1SrCin = (self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
        (self->ctx.hwK_ * self->ctx.curSingleCoreDk_ * self->ctx.tiling_->n0);
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.tiling_->n0 * self->ctx.dhwI_ +
            (self->ctx.curNL1Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwI_;
    } else {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.tiling_->n0 +
            (self->ctx.curNL1Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwI_ * self->ctx.tiling_->cin;
    }

    uint64_t bL1N = ShiftCeilM0(self->ctx.curStepN_ * self->ctx.baseUseN_ / self->ctx.curSingleCoreDk_,
        self->ctx.tiling_->n0);
    uint32_t bL1cin1CopyLen = Ceil(bL1N, self->ctx.hwK_);
    if (self->ctx.singleShapeCin_ > 16) {
        // 当singleShapeCin_小于16时，搬运的bL1cin1CopyLen最大为1，无需多搬
        if (self->ctx.hwK_ > bL1N) {
            // 需判断过hwK_对bL1N取余不等于0了
            if (self->ctx.hwK_ % bL1N != 0) {
                ++bL1cin1CopyLen; // bL1cin1CopyLen本身就向上取整，也就是说会多搬一行，再多搬一行也就是说覆盖了跨三行的情况
            }
        } else if (2 * bL1N % self->ctx.hwK_ != 0) {
            // 如果2 * bL1N % self->ctx.hwK_ == 0，仅需搬两行，不会出现跨三行的情况
            ++bL1cin1CopyLen;
        }
    }
    uint32_t cin1RemainLen = ShiftCeilM0(self->ctx.singleShapeCin_, self->ctx.tiling_->n0) - b1SrCin;
    // L1上不能多载入数据，否则会导致underflow或者overflow问题
    self->ctx.bL1cin1CopyLen = cin1RemainLen > bL1cin1CopyLen ? (bL1cin1CopyLen * self->ctx.tiling_->n0) :
        (self->ctx.singleShapeCin_ - b1SrCin * self->ctx.tiling_->n0);
}

template <class Intf>
static __aicore__ inline void CalOut2B1Params(Intf *self, Out2L1ScalarParams& params)
{
    // 计算L1上cin起始idx, 去掉cin1HkWkCin里的HkWk
    uint64_t b1SrCin = (self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
        (self->ctx.hwK_ * self->ctx.curSingleCoreDk_);
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.dhwI_ +
            (self->ctx.curNL1Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ * self->ctx.hwI_;
    } else {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) + (self->ctx.curNL1Idx_ / self->ctx.cinHkWkLoop_) *
            self->ctx.curSingleCoreDk_ * self->ctx.hwI_ * self->ctx.tiling_->cin;
    }

    uint64_t bL1N = self->ctx.curStepN_ * self->ctx.baseUseN_ / self->ctx.curSingleCoreDk_;
    uint32_t bL1cin1CopyLen = Ceil(bL1N, self->ctx.hwK_);
    uint32_t cin1RemainLen = self->ctx.singleShapeCin_ - b1SrCin;
    self->ctx.bL1cin1CopyLen = cin1RemainLen > bL1cin1CopyLen ? bL1cin1CopyLen: cin1RemainLen;
}

template <class Intf>
static __aicore__ inline void CalOut2L1ScalarParams(Intf *self, Out2L1ScalarParams& params)
{
    // to L1A
    if (params.isLoad2L1A) {
        if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            params.out2A1SrcAddr = static_cast<uint64_t>(self->ctx.curML1Idx_) * self->ctx.tiling_->baseM *
                self->ctx.tiling_->dout * self->ctx.hwO_;
        } else {
            params.out2A1SrcAddr = static_cast<uint64_t>(self->ctx.curML1Idx_) * self->ctx.tiling_->baseM;
        }
        params.isLastMAL1 = DivStepM((self->ctx.mIter_ - 1), self->ctx.tiling_->stepM) ==
            DivStepM(self->ctx.curML0Idx_, self->ctx.tiling_->stepM);
    }

    // to L1B
    if (params.isLoad2L1B) {
        if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4 用来替换除法运算
            CalOut2B1ParamsBaseNUndivided(self, params);
        } else {
            CalOut2B1Params(self, params);
        }
        uint64_t singleShapeHi = self->ctx.singleShapeHo_* self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
        params.singleShapeHi = self->ctx.tiling_->hi > singleShapeHi ? singleShapeHi : self->ctx.tiling_->hi;
    }
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp32Nd2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    if (!self->ctx.isSplitWo_) {
        // fp32 时，使用nd2nz实现DN规格，主要是将D轴变为HoWo，N轴变为Cout
        /* B N D ==> D1 N B D0 */
        /* (HoWo), Cout ==> (HoWo)/C8, (Cout1, Cout0), C8 */
        nd2NzParams.ndNum = 1;
        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            nd2NzParams.nValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
        } else {
            nd2NzParams.nValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM; // N:Cout
        }

        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK
            nd2NzParams.dValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            nd2NzParams.dValue = self->ctx.kal1_; // D
        }

        nd2NzParams.srcDValue = self->ctx.dhwO_; // D_src_stride(loop1_src_stride)
        nd2NzParams.srcNdMatrixStride = 1; // B_src_stride(loop4_src_stride)
        nd2NzParams.dstNzMatrixStride = 1; // B_dst_stride(loop4_dst_stride)
        // D1_dst_stride(loop3_dst_stride)
        nd2NzParams.dstNzC0Stride = ShiftMulM0(ShiftCeilM0(nd2NzParams.nValue, self->ctx.tiling_->m0),  self->ctx.tiling_->m0);
        nd2NzParams.dstNzNStride = 1; // N_dst_stride(loop2_dst_stride)
        return;
    }

    SplitWLoadToA1Nd2Nz(self, kaIdx, params, kaStepIdx, nd2NzParams);
}

template <class Intf>
static __aicore__ inline void SplitWLoadToA1Dn2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams) {
    uint32_t hoStartIdx = kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_;
    uint32_t kal1Use = (self->ctx.stepKaRound == (kaStepIdx + 1)) ?
        (self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - kaIdx * self->ctx.tiling_->baseK) : (self->ctx.kal1_);
    uint32_t hoEndIdx = Ceil(kaIdx * self->ctx.tiling_->baseK + kal1Use, self->ctx.singleShapeWo_);
    uint32_t hoCopyLen = hoEndIdx - hoStartIdx;

    dn2NzParams.dnNum = hoCopyLen;
    dn2NzParams.nValue = self->ctx.singleShapeWo_;   //NCDHW
    if (params.isLastMAL1) {
        // 最后一块mAL1，需要考虑tailM
        dn2NzParams.dValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
    } else {
        dn2NzParams.dValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM;
    }
    dn2NzParams.srcDValue = self->ctx.dhwO_;
    dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wo;

    //dstNzC0Stride需要用nValue * dnNum 对齐k0
    dn2NzParams.dstNzC0Stride = ShiftCeilM0(dn2NzParams.nValue * dn2NzParams.dnNum, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    dn2NzParams.dstNzNStride = 1;
    // NDC1HWC0排布，Matrix数量配置的是H轴，目的Matrix间隔为W*C0，即nValue * k0;
    dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;
    //由于切wo存在多搬情况，需要修正alignedL1UseKa_,防止load2d时计算的src_stride错误
    self->ctx.alignedL1UseKa_ = dn2NzParams.dstNzC0Stride;
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp16Dn2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams)
{
    if (!self->ctx.isSplitWo_) {
        dn2NzParams.dnNum = 1;
        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            dn2NzParams.nValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            dn2NzParams.nValue = self->ctx.kal1_;
        }

        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            dn2NzParams.dValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
        } else {
            dn2NzParams.dValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM;
        }

        dn2NzParams.srcDValue = self->ctx.dhwO_;
        //由于dnNum为1，可以不配置srcDnMatrixStride
        dn2NzParams.dstNzC0Stride = ShiftCeilChannelSize<Intf>(dn2NzParams.nValue, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;
        dn2NzParams.dstNzNStride = 1;
        // 由于dnNum为1，可以不配置// B_dst_stride(loop4_dst_stride)
        dn2NzParams.dstNzMatrixStride = 1;
        return ;
    }

    SplitWLoadToA1Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp32Dn2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Dn2NzParams& dn2NzParams)
{
    if (!self->ctx.isSplitWo_) {
        // fp32 时，使用dn2nz实现ND规格，主要是将D轴变为HoWo，N轴变为Cout
        /* B D N ==> D1 N B D0 */
        /* Cout, (HoWo), 1 ==> (HoWo)/C8, 1, (Cout1, Cout0), C8 */
        dn2NzParams.dnNum = 1;
        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            dn2NzParams.nValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
        } else {
            dn2NzParams.nValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM; // B:Cout
        }

        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            dn2NzParams.dValue = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK;
        } else {
            dn2NzParams.dValue = self->ctx.kal1_; // D
        }

        dn2NzParams.srcDValue = self->ctx.tiling_->cout; // D_src_stride(loop1_src_stride)
        dn2NzParams.srcDnMatrixStride = 1;  // B_src_stride(loop4_src_stride)
        dn2NzParams.dstNzMatrixStride = 1; // B_dst_stride(loop4_dst_stride)
        // D1_dst_stride(loop3_dst_stride)
        dn2NzParams.dstNzC0Stride = ShiftCeilM0(dn2NzParams.nValue, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        dn2NzParams.dstNzNStride = 1; // N_dst_stride(loop2_dst_stride)
        return;
    }

    SplitWLoadToA1Dn2Nz<Intf>(self, kaIdx, params, kaStepIdx, dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1Nd2NzNormal(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
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
        nd2NzParams.dValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
    } else {
        nd2NzParams.dValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM;
    }

    nd2NzParams.srcDValue = self->ctx.tiling_->cout;
    nd2NzParams.dstNzC0Stride = ShiftCeilChannelSize<Intf>(nd2NzParams.nValue, self->ctx.tiling_->k0) *
        self->ctx.tiling_->k0;
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = 1;
}

template <class Intf>
static __aicore__ inline void SplitWLoadToA1Nd2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams) {
    uint32_t hoStartIdx = kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_;
    uint32_t kal1Use = (self->ctx.stepKaRound == (kaStepIdx + 1)) ?
        (self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - kaIdx * self->ctx.tiling_->baseK) : (self->ctx.kal1_);
    uint32_t hoEndIdx = Ceil(kaIdx * self->ctx.tiling_->baseK + kal1Use, self->ctx.singleShapeWo_);
    uint32_t hoCopyLen = hoEndIdx - hoStartIdx;

    nd2NzParams.ndNum = hoCopyLen;
    nd2NzParams.nValue = self->ctx.singleShapeWo_;
    if (params.isLastMAL1) {
        // 最后一块mAL1，需要考虑tailM
        nd2NzParams.dValue = (self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_;
    } else {
        nd2NzParams.dValue = self->ctx.curStepM_ * self->ctx.tiling_->baseM;
    }
    nd2NzParams.srcDValue = self->ctx.tiling_->cout;
    nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->wo * self->ctx.tiling_->cout;

    // dstNzC0Stride需要用nValue * dnNum 对齐m0
    nd2NzParams.dstNzC0Stride = ShiftCeilM0(nd2NzParams.nValue * nd2NzParams.ndNum, self->ctx.tiling_->m0) * 
                            self->ctx.tiling_->m0;
    nd2NzParams.dstNzNStride = 1;
    // NDC1HWC0排布，Matrix数量配置的是H轴，目的Matrix间隔为W*C0，即nValue * k0;
    nd2NzParams.dstNzMatrixStride = nd2NzParams.nValue * self->ctx.tiling_->k0;
    // 由于切wo存在多搬情况，需要修正alignedL1UseKa_,防止load2d时计算的src_stride错误
    self->ctx.alignedL1UseKa_ = nd2NzParams.dstNzC0Stride;
}

template <class Intf>
static __aicore__ inline void LoadToA1Fp16Nd2Nz(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, Nd2NzParams& nd2NzParams)
{
    if (!self->ctx.isSplitWo_) {
        LoadToA1Nd2NzNormal(self, kaIdx, params, kaStepIdx, nd2NzParams);
    } else {
        SplitWLoadToA1Nd2Nz(self, kaIdx, params, kaStepIdx, nd2NzParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadToA1Dn2NzBaseUseMEqualOne(Intf *self, const Out2L1ScalarParams& params,
    uint64_t kaStepIdx, uint32_t totalSize, LocalTensor<typename Intf::SrcT> &useA1Buf)
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
        dn2NzParams.dValue = tailCount; //元素个数
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
static __aicore__ inline void LoadToA1Nd2NzBaseUseMEqualOne(Intf *self, const Out2L1ScalarParams& params,
    uint64_t kaStepIdx, uint32_t totalSize, LocalTensor<typename Intf::SrcT> &useA1Buf)
{
    uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + kaStepIdx * self->ctx.kal1_ * self->ctx.tiling_->cout;
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;
    dn2NzParams.nValue = 1;
    dn2NzParams.dValue = totalSize; //元素个数
    dn2NzParams.srcDnMatrixStride = 1;
    dn2NzParams.srcDValue = self->ctx.tiling_->cout;
    dn2NzParams.dstNzC0Stride = 1;
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 1;
    DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToA1BaseUseMEqualOne(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, LocalTensor<typename Intf::SrcT> &useA1Buf)
{
    self->ctx.alignedL1UseM_ = params.isLastMAL1 ?
        ((self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_) :
        (self->ctx.curStepM_ * self->ctx.tiling_->baseM);
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
static __aicore__ inline void LoadToA1ForTransFormat(Intf *self, uint64_t kaIdx,
    const Out2L1ScalarParams& params, uint64_t kaStepIdx, LocalTensor<typename Intf::SrcT> &useA1Buf)
{
    if (self->ctx.baseUseM_ == 1) {
        LoadToA1BaseUseMEqualOne(self, kaIdx, params, kaStepIdx, useA1Buf);
    } else {
        if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            uint64_t offset = kaStepIdx * self->ctx.kal1_;
            if (self->ctx.isSplitWo_) {
                offset = (kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_ * self->ctx.tiling_->wo);
            }
            uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + offset;
            if constexpr (IsSameType<typename Intf::SrcT, float>::value ||
                IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
                if (!self->ctx.isSplitWo_) {
                    // fp32 时，使用nd2nz实现DN规格，主要是将D轴变为HoWo，N轴变为Cout
                    Nd2NzParams nd2NzParams;
                    LoadToA1Fp32Nd2Nz<Intf>(self, kaIdx, params, kaStepIdx, nd2NzParams);
                    DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
                } else { //hifloat8_t 类型暂不支持,在tiling侧已拦截，一定走的是不切W分支
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
            if (self->ctx.isSplitWo_) {
                offset = (kaStepIdx * self->ctx.kal1_ / self->ctx.singleShapeWo_ * self->ctx.tiling_->wo) * 
                            self->ctx.tiling_->cout;
            }
            uint64_t out2A1SrcAddrOffset = params.out2A1SrcAddr + offset;
            if constexpr (IsSameType<typename Intf::SrcT, float>::value ||
                IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
                if (!self->ctx.isSplitWo_) {
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
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf *self, bool cachePosA1, uint64_t kaIdx, const Out2L1ScalarParams& params,
                                bool isLoadA1, uint64_t kaStepIdx)
{
    if (!isLoadA1) {
        return;
    }

    auto l1UseKa_ = (kaStepIdx + 1 == DivCeil(self->ctx.kIter_, self->ctx.tiling_->stepKa)) ?
        self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - kaStepIdx * self->ctx.tiling_->stepKa *
        self->ctx.tiling_->baseK : self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
    self->ctx.alignedL1UseKa_ = ShiftCeilChannelSize<Intf>(l1UseKa_, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;

    if constexpr (IsSameType<typename Intf::SrcT, float>::value || IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
        if (self->ctx.baseUseM_ != 1) {
            auto l1UseM = params.isLastMAL1 ?
                ((self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_) :
                (self->ctx.curStepM_ * self->ctx.tiling_->baseM);
            self->ctx.alignedL1UseM_ = ShiftCeilM0(l1UseM, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        }
    }

    if (params.isLoad2L1A) {
        LocalTensor<typename Intf::SrcT> useA1Buf;
        if (cachePosA1) {
            useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useA1Buf = self->ctx.a1Pong_.template AllocTensor<typename Intf::SrcT>();
        }

        LoadToA1ForTransFormat(self, kaIdx, params, kaStepIdx, useA1Buf);

        if (cachePosA1) {
            self->ctx.a1Ping_.EnQue(useA1Buf);
        } else {
            self->ctx.a1Pong_.EnQue(useA1Buf);
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadToB1Dn2Nz(Intf *self, const uint32_t hiCopyLen,
    const uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams &params, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    if (!self->ctx.isSplitWo_) {
        Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = self->ctx.curSingleCoreDk_;
        // 若curStepN_大于cinHkWkLoop_，则说明StepN上包含dk，因此在加载L1时，需要加载stepN/cinHkWkLoop_个dk
        if (self->ctx.curStepN_ > self->ctx.cinHkWkLoop_) {
            dn2NzParams.dnNum *= (self->ctx.curStepN_ / self->ctx.cinHkWkLoop_);
        }

        dn2NzParams.nValue = static_cast<uint64_t>(hiCopyLen) * self->ctx.tiling_->wi;
        dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        dn2NzParams.srcDValue = self->ctx.dhwI_;
        dn2NzParams.srcDnMatrixStride = self->ctx.hwI_;

        dn2NzParams.dstNzC0Stride = dn2NzParams.nValue;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = ShiftCeilChannelSize<Intf>(dn2NzParams.dValue, self->ctx.tiling_->k0) *
            dn2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);
    } else {    //切分出来的singleshapedi=1才可以，否则需要通过多条ND2NZ指令进行拼接
        Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = hiCopyLen;
        //需要考虑左右边pad的情况，不能直接等于singleshapewi
        dn2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
        dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        dn2NzParams.srcDValue = self->ctx.dhwI_;
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wi;

        dn2NzParams.dstNzC0Stride = dn2NzParams.nValue * dn2NzParams.dnNum;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadToB1Nd2Nz(Intf *self, const uint32_t hiCopyLen,
    const uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams &params, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    if (!self->ctx.isSplitWo_) {
        Nd2NzParams nd2NzParams;
        nd2NzParams.ndNum = self->ctx.curSingleCoreDk_;
        // 若curStepN_大于cinHkWkLoop_，则说明StepN上包含dk，因此在加载L1时，需要加载stepN/cinHkWkLoop_个dk
        if (self->ctx.curStepN_ > self->ctx.cinHkWkLoop_) {
            nd2NzParams.ndNum *= (self->ctx.curStepN_ / self->ctx.cinHkWkLoop_);
        }
        nd2NzParams.srcNdMatrixStride = self->ctx.hwI_ * self->ctx.tiling_->cin;

        nd2NzParams.nValue = static_cast<uint64_t>(hiCopyLen) * self->ctx.tiling_->wi;
        nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;

        nd2NzParams.srcDValue = self->ctx.tiling_->cin;
        nd2NzParams.dstNzC0Stride = nd2NzParams.nValue;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = ShiftCeilChannelSize<Intf>(nd2NzParams.dValue, self->ctx.tiling_->k0) *
            nd2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);
    } else {
        Nd2NzParams nd2NzParams;
        nd2NzParams.ndNum = hiCopyLen;
        //需要考虑左右边pad的情况，不能直接等于singleshapewi
        nd2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
        nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        nd2NzParams.srcDValue = self->ctx.tiling_->cin;
        nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->wi * self->ctx.tiling_->cin;

        nd2NzParams.dstNzC0Stride = nd2NzParams.nValue * nd2NzParams.ndNum;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = nd2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);        
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf *self, bool cachePosB1, const Out2L1ScalarParams& params,
                                bool isLoadB1, uint64_t kbStepIdx, bool &skipCurrentHiCompute)
{
    if (!isLoadB1) {
        return;
    }
    skipCurrentHiCompute = false;
    // 需要载入BL1的条件为，被计算的BL0块是BL1上的第一块数据，一次载入完整BL1大小
    // 此时满足以下条件之一需要载入BL1：
    // 1.BL1上无db，并且K方向需要多于一个buffer，每次都需要载入；BL1开db，并且K方向buffer数量小于等于2
    // 2.singleShapeK / stepKb > 2, 优先循环k方向，BL1上数据无法复用
    // 3.order_M时，L1上驻留AL1, BL1数据不复用
    // 4.order_N时，BL1驻留在L1上，且K <=
    // 2，即L1上可以栽下全部Kb，此时遍历M方向，BL1数据上数据不会被覆盖，只在M方向循环第一次时载入BL1
    if (params.isLoad2L1B) {
        // L0shape到orgShape的对应关系，L0和L1是16对齐的，orgShape是Wi对齐的,先算Wo对齐再算Wi对齐
        // 先算L0B所在BL1块的起始地址，16对齐的
        uint64_t b1SrcKAlign = kbStepIdx * self->ctx.kbl1_;
        // load3d必须有完整Wo，做Wo对齐，计算起始地址所在的Ho
        uint32_t b1SrcHo = b1SrcKAlign / self->ctx.singleShapeWo_;
        uint32_t b1SrcHoGm = b1SrcHo + self->ctx.hoStartIdx_;
        // 计算Ho对应的Hi，根据卷积原理
        int64_t b1SrcHiGm = static_cast<uint64_t>(b1SrcHoGm) * self->ctx.tiling_->strideH - self->ctx.tiling_->padUp;
        uint32_t b1SrcHi = 0;
        if (b1SrcHiGm > 0 && self->ctx.hiStartIdx_ > 0) {
            b1SrcHi = b1SrcHiGm - self->ctx.hiStartIdx_;
        } else if (b1SrcHiGm > 0) {
            b1SrcHi = b1SrcHiGm;
        }

        uint32_t kbl1 = self->ctx.kbl1_;
        if (self->ctx.stepKbRound == (kbStepIdx + 1)) {
            kbl1 = self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - b1SrcKAlign;
        }
        uint32_t ho = CalRows2Copy(kbl1, self->ctx.singleShapeWo_);
        uint32_t hiCopyLen = ho * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
        if ((b1SrcHiGm + hiCopyLen <= 0) || (b1SrcHiGm >= self->ctx.tiling_->hi)) {
            skipCurrentHiCompute = true;
            return ;
        }

        uint32_t padUp = 0;
        if (b1SrcHiGm < 0) {
            hiCopyLen = hiCopyLen + b1SrcHiGm;
            padUp = -b1SrcHiGm;
        }
        if (b1SrcHiGm + hiCopyLen > self->ctx.tiling_->hi) {
            hiCopyLen = self->ctx.tiling_->hi - b1SrcHiGm;
        }

        uint32_t hiRemainLen = params.singleShapeHi - b1SrcHi;
        hiCopyLen = hiRemainLen > hiCopyLen ? hiCopyLen: hiRemainLen;
        if (hiCopyLen == 0) {
            skipCurrentHiCompute = true;
            return ;
        }
        LocalTensor<typename Intf::SrcT> useB1Buf;
        if (cachePosB1) {
            useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useB1Buf = self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();
        }
        // 得到gm的偏移量
        uint64_t out2B1SrcAddrOffset = 0;
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            out2B1SrcAddrOffset = params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi;
        } else {
            out2B1SrcAddrOffset = params.out2B1SrcAddr +
                static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi * self->ctx.tiling_->cin;
        }

        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            LoadToB1Dn2Nz(self, hiCopyLen, out2B1SrcAddrOffset, params, useB1Buf);
        } else {
            LoadToB1Nd2Nz(self, hiCopyLen, out2B1SrcAddrOffset, params, useB1Buf);
        }

        if (cachePosB1) {
            self->ctx.bL1HiCopyLenPing = hiCopyLen;
            self->ctx.bL1PadUpPing = padUp;
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.bL1HiCopyLenPong = hiCopyLen;
            self->ctx.bL1PadUpPong = padUp;
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

template <class Intf>
__aicore__ inline void UpdateSrcAddrBaseOnBatchDoutIdx(Intf *self, uint64_t curLoopBatchDoutIdx,
    Out2L1ScalarParams& params, bool &skipCurrentDinCompute)
{
    int32_t curBatchIdx = curLoopBatchDoutIdx / self->ctx.tiling_->dout;
    int32_t curDoutIdx = curLoopBatchDoutIdx - curBatchIdx * self->ctx.tiling_->dout;
    uint64_t curDinIdx = static_cast<uint64_t>(curDoutIdx) *
        self->ctx.tiling_->strideD + self->ctx.dkStartIdx_ * self->ctx.tiling_->dilationD;
    // 在拆分dk的情况下，如果涉及到的dinIdx在padding部分，则跳过Compute计算
    bool invalidDinIdx = curDinIdx < self->ctx.tiling_->padFront ||
        curDinIdx >= (self->ctx.tiling_->di + self->ctx.tiling_->padFront);
    if (self->ctx.seperateDk_ && invalidDinIdx) {
        // dinIdx当前在padding部分，因此跳过本轮mmad的计算。
        // 由于每次A, B矩阵的偏移计算依赖于前一次BatchDoutIdx迭代中的偏移，因此不能直接return跳过接下来偏移的计算
        skipCurrentDinCompute = true;
    }
    if (curLoopBatchDoutIdx == self->ctx.batchDoutStartIdx_) { // 当batchLoopBatchDoutIdx为起始Idx时，无需更新地址
        return;
    }
    // 更新batchIdx和doutIdx，重新计算offsetA和offsetB的值，优先循环dout方向；offsetC不需要更新
    int32_t preBatchIdx = curBatchIdx;
    int32_t preDoutIdx = curDoutIdx - 1;
    if (curDoutIdx == 0) {
        preBatchIdx--;
        preDoutIdx = (curLoopBatchDoutIdx - 1) - preBatchIdx * self->ctx.tiling_->dout;
    }

    int64_t batchOffsetAIncre = (curBatchIdx - preBatchIdx) *
        self->ctx.tiling_->cout * self->ctx.tiling_->dout * self->ctx.tiling_->ho * self->ctx.tiling_->wo;
    int64_t doutOffsetAIncre = (curDoutIdx - preDoutIdx) *
        static_cast<int64_t>(self->ctx.tiling_->ho * self->ctx.tiling_->wo);
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        doutOffsetAIncre = doutOffsetAIncre * self->ctx.tiling_->cout;
    }
    params.out2A1SrcAddr = params.out2A1SrcAddr + (batchOffsetAIncre + doutOffsetAIncre);

    int64_t batchOffsetBIncre = static_cast<int64_t>(curBatchIdx - preBatchIdx) *
        self->ctx.tiling_->cin * self->ctx.tiling_->di * self->ctx.tiling_->hi * self->ctx.tiling_->wi;
    int64_t doutOffsetBIncre = static_cast<int64_t>(curDoutIdx - preDoutIdx) *
        self->ctx.tiling_->strideD * self->ctx.tiling_->hi * self->ctx.tiling_->wi;
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        doutOffsetBIncre = doutOffsetBIncre * self->ctx.tiling_->cin;
    }
    params.out2B1SrcAddr = params.out2B1SrcAddr + (batchOffsetBIncre + doutOffsetBIncre);
}
}  // namespace ConvolutionBackpropFunc

#endif