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
 * \file swi_glu_grad_move_align.h
 * \brief
 */

#ifndef SWI_GLU_GRAD_UB_REARRANGE_H
#define SWI_GLU_GRAD_UB_REARRANGE_H

#include "swi_glu_grad_base.h"

namespace SwiGluGrad {
using namespace AscendC;

constexpr int32_t BLOCK = 32;

template <typename T> struct VciTypeGet;
template <>
struct VciTypeGet<uint32_t> {
    using T = int32_t;
};
template <>
struct VciTypeGet<uint16_t> {
    using T = int16_t;
};

template <typename T, typename U>
class SwiGluGradUbRearrangeKernel : public SwiGluGradBaseKernel<T> {
public:
    __aicore__ inline SwiGluGradUbRearrangeKernel(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR out, const GluBaseTilingData& tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void RearrangeUbIn(uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void RearrangeUbOut(uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void GenerateGatherIndex(uint32_t colLen);
    __aicore__ inline void RearrangeUbInByGather(uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void RearrangeUbOutByScatter(uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void ProcessNotGather();
    __aicore__ inline void ProcessWithGather();
    __aicore__ inline void ComputePerLoop(uint32_t dataCount);

protected:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueX_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueGrad_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueXGrad_;

    TBuf<QuePosition::VECCALC> Apart_;
    TBuf<QuePosition::VECCALC> Bpart_;
    TBuf<QuePosition::VECCALC> OutApart_;
    TBuf<QuePosition::VECCALC> OutBpart_;
    TBuf<QuePosition::VECCALC> indexBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<T> outGm_;

    // mode
    bool needGather_{false};

    // ub内循环处理大小
    uint32_t rowUbNorm_{0};
    uint32_t rowUbTail_{0};
    int64_t rowLoopByUb_{0};

    uint32_t vfLen_ = platform::GetVRegSize() / sizeof(T);
    const uint32_t EXTRA_BUFFER_SIZE{256};
    static constexpr int32_t BLOCK_NUM = BLOCK / sizeof(T);

    // gather var
    uint32_t perVfColNum_{0};
    uint32_t gatherNormNum_{0};
};

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::Init(
    GM_ADDR grad, GM_ADDR x, GM_ADDR out, const GluBaseTilingData& tilingData)
{
    if (!this->InitBase(tilingData)) {
        return;
    }

    uint64_t existNodeSize = tilingData.ubSize / BUFFER_NUMBER / DOUBLE_BUFFER;
    uint64_t tBufferSize = RoundDownToVL(existNodeSize / SPLIT_HALF);
    existNodeSize = tBufferSize * SPLIT_HALF;

    rowUbNorm_ = existNodeSize / (this->colTotal_ * sizeof(T)); // 单次搬入的数据量，向下取整，正常块大小
    if (rowUbNorm_ > this->rowProcess_) {
        rowUbNorm_ = this->rowProcess_;
    }

    rowLoopByUb_ = (this->rowProcess_ + rowUbNorm_ - 1) / rowUbNorm_; // 总行数 / 单次处理行数 = 循环次数。 向上取整
    rowUbTail_ = this->rowProcess_ -
                 rowUbNorm_ * (rowLoopByUb_ - 1); // 最后一次的尾块，总行数 - 单次处理行数 * (总次数 - 1) < 单次处理行数

    if (this->colProcess_ * sizeof(T) < BLOCK) {
        needGather_ = true;
        perVfColNum_ = vfLen_ / this->colProcess_;
        gatherNormNum_ = vfLen_ / this->colProcess_ * this->colProcess_;
    }

    // init params
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gradYGm_.SetGlobalBuffer((__gm__ T*)grad);
    outGm_.SetGlobalBuffer((__gm__ T*)out);

    pipe_.InitBuffer(inQueX_, DOUBLE_BUFFER, existNodeSize);
    pipe_.InitBuffer(inQueGrad_, DOUBLE_BUFFER, tBufferSize);
    pipe_.InitBuffer(outQueXGrad_, DOUBLE_BUFFER, existNodeSize);

    pipe_.InitBuffer(Apart_, tBufferSize);
    pipe_.InitBuffer(Bpart_, tBufferSize);
    pipe_.InitBuffer(OutApart_, tBufferSize);
    pipe_.InitBuffer(OutBpart_, tBufferSize);
    pipe_.InitBuffer(indexBuf_, EXTRA_BUFFER_SIZE);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::CopyIn(int64_t offset, uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> xTensor = inQueX_.AllocTensor<T>();
    LocalTensor<T> gradTensor = inQueGrad_.AllocTensor<T>();

    int64_t inputX = this->partAStart_ + offset;
    int64_t grad = this->gradStart_ + offset / SPLIT_HALF;

    uint32_t xTotalNum = rowLen * this->colTotal_;
    uint32_t gradTotalNum = rowLen * colLen;

    uint8_t rightPadNum = (xTotalNum + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM - xTotalNum;
    uint8_t rightPadNumGrad = (gradTotalNum + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM - gradTotalNum;

    DataCopyExtParams copyParamForX{
        static_cast<uint16_t>(1), static_cast<uint32_t>(xTotalNum * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams{
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadNum), static_cast<uint8_t>(0)};

    DataCopyPad(xTensor, xGm_[inputX], copyParamForX, padParams);

    DataCopyExtParams copyParamForGrad{
        static_cast<uint16_t>(1), static_cast<uint32_t>(gradTotalNum * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParamsForGrad{
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadNumGrad), static_cast<uint8_t>(0)};
    DataCopyPad(gradTensor, gradYGm_[grad], copyParamForGrad, padParamsForGrad);

    inQueX_.EnQue<T>(xTensor);
    inQueGrad_.EnQue<T>(gradTensor);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::CopyOut(int64_t offset, uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> outTensor = outQueXGrad_.DeQue<T>();
    uint64_t partA = this->partAStart_ + offset;

    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(rowLen * colLen * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(outGm_[partA], outTensor, copyParams);

    outQueXGrad_.FreeTensor(outTensor);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::RearrangeUbIn(uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> xLocal = inQueX_.DeQue<T>();
    LocalTensor<T> aLocal = Apart_.Get<T>();
    LocalTensor<T> bLocal = Bpart_.Get<T>();
    auto ubSrcAddrA = (__ubuf__ T*)xLocal.GetPhyAddr();
    auto ubSrcAddrB = (__ubuf__ T*)xLocal.GetPhyAddr(colLen);
    auto ubDstAddrA = (__ubuf__ T*)aLocal.GetPhyAddr();
    auto ubDstAddrB = (__ubuf__ T*)bLocal.GetPhyAddr();

    uint16_t size0 = rowLen;
    uint16_t size1 = colLen / vfLen_;
    uint32_t main = vfLen_;
    uint32_t tail = colLen - (vfLen_ * size1);
    uint32_t rowStride = this->colTotal_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregA;
        MicroAPI::RegTensor<T> vregB;
        MicroAPI::UnalignReg uSrcA;
        MicroAPI::UnalignReg uSrcB;
        MicroAPI::UnalignReg uDstA;
        MicroAPI::UnalignReg uDstB;

        for (uint16_t row = 0; row < size0; ++row) {
            auto curASrcAddr = ubSrcAddrA + row * rowStride;
            auto curBSrcAddr = ubSrcAddrB + row * rowStride;
            auto curADstAddr = ubDstAddrA + row * colLen;
            auto curBDstAddr = ubDstAddrB + row * colLen;

            MicroAPI::DataCopyUnAlignPre(uSrcA, curASrcAddr);
            MicroAPI::DataCopyUnAlignPre(uSrcB, curBSrcAddr);

            for (uint16_t i = 0; i < size1; ++i) {
                MicroAPI::DataCopyUnAlign(vregA, uSrcA, curASrcAddr, main);
                MicroAPI::DataCopyUnAlign(vregB, uSrcB, curBSrcAddr, main);
                MicroAPI::DataCopyUnAlign(curADstAddr, vregA, uDstA, main);
                MicroAPI::DataCopyUnAlign(curBDstAddr, vregB, uDstB, main);
            }

            MicroAPI::DataCopyUnAlign(vregA, uSrcA, curASrcAddr, tail);
            MicroAPI::DataCopyUnAlign(curADstAddr, vregA, uDstA, tail);
            MicroAPI::DataCopyUnAlignPost(curADstAddr, uDstA, 0);

            MicroAPI::DataCopyUnAlign(vregB, uSrcB, curBSrcAddr, tail);
            MicroAPI::DataCopyUnAlign(curBDstAddr, vregB, uDstB, tail);
            MicroAPI::DataCopyUnAlignPost(curBDstAddr, uDstB, 0);
        }
    }

    inQueX_.FreeTensor(xLocal);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::RearrangeUbOut(uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> outLocal = outQueXGrad_.AllocTensor<T>();
    LocalTensor<T> calcALocal = OutApart_.Get<T>();
    LocalTensor<T> calcBLocal = OutBpart_.Get<T>();
    auto ubDstAddrA = (__ubuf__ T*)outLocal.GetPhyAddr();
    auto ubDstAddrB = (__ubuf__ T*)outLocal.GetPhyAddr(colLen);
    auto ubSrcAddrA = (__ubuf__ T*)calcALocal.GetPhyAddr();
    auto ubSrcAddrB = (__ubuf__ T*)calcBLocal.GetPhyAddr();

    uint16_t size0 = rowLen;
    uint16_t size1 = colLen / vfLen_;
    uint32_t main = vfLen_;
    uint32_t tail = colLen - (vfLen_ * size1);
    uint32_t dstPerRowLen = this->colTotal_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregA;
        MicroAPI::RegTensor<T> vregB;
        MicroAPI::UnalignReg uSrcA;
        MicroAPI::UnalignReg uSrcB;
        MicroAPI::UnalignReg uDstA;
        MicroAPI::UnalignReg uDstB;

        for (uint16_t row = 0; row < size0; ++row) {
            auto curASrcAddr = ubSrcAddrA + row * colLen;
            auto curBSrcAddr = ubSrcAddrB + row * colLen;
            auto curADstAddr = ubDstAddrA + row * dstPerRowLen;
            auto curBDstAddr = ubDstAddrB + row * dstPerRowLen;

            MicroAPI::DataCopyUnAlignPre(uSrcA, curASrcAddr);
            MicroAPI::DataCopyUnAlignPre(uSrcB, curBSrcAddr);

            for (uint16_t i = 0; i < size1; ++i) {
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregA, uSrcA, curASrcAddr, main);
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregB, uSrcB, curBSrcAddr, main);
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curADstAddr, vregA, uDstA, main);
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curBDstAddr, vregB, uDstB, main);
            }

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregA, uSrcA, curASrcAddr, tail);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curADstAddr, vregA, uDstA, tail);
            MicroAPI::DataCopyUnAlignPost(curADstAddr, uDstA, 0);

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregB, uSrcB, curBSrcAddr, tail);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curBDstAddr, vregB, uDstB, tail);
            MicroAPI::DataCopyUnAlignPost(curBDstAddr, uDstB, 0);
        }
    }

    outQueXGrad_.EnQue<T>(outLocal);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::GenerateGatherIndex(uint32_t colLen)
{
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint32_t colTotalLen = colLen * SPLIT_HALF;

    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::T;
        MicroAPI::RegTensor<regType> index;
        MicroAPI::RegTensor<U> indexCast;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;

        MicroAPI::RegTensor<T> vregA;
        MicroAPI::RegTensor<T> vregB;

        MicroAPI::MaskReg pIndex = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(index, 0); // a = 0 ~ 254/sizeof(T)
        if constexpr (sizeof(U) == sizeof(uint32_t) || sizeof(U) == sizeof(uint16_t)) {
            indexCast = (MicroAPI::RegTensor<U>&)index;
        }
        Duplicate(v0, (U)colLen, pIndex);

        Div(v1, indexCast, v0, pIndex);       // b = a / (cut / SPLIT_HALF)
        Muls(v1, v1, (U)colTotalLen, pIndex); // c = b * cut

        Div(v2, indexCast, v0,
            pIndex); // d = a % (cut/2) = index % (cut / SPLIT_HALF) = index - index / (cut/2) * (cut/2)
        Mul(v2, v2, v0, pIndex);
        Sub(v2, indexCast, v2, pIndex);

        Add(indexCast, v1, v2, pIndex); // e = c + d

        MicroAPI::DataCopy(indexAddr, indexCast, pIndex);
    }
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::RearrangeUbInByGather(uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> xLocal = inQueX_.DeQue<T>();
    LocalTensor<T> aLocal = Apart_.Get<T>();
    LocalTensor<T> bLocal = Bpart_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();

    auto ubSrcAddrA = (__ubuf__ T*)xLocal.GetPhyAddr();
    auto ubSrcAddrB = (__ubuf__ T*)xLocal.GetPhyAddr(colLen);
    auto ubDstAddrA = (__ubuf__ T*)aLocal.GetPhyAddr();
    auto ubDstAddrB = (__ubuf__ T*)bLocal.GetPhyAddr();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    uint32_t main = gatherNormNum_;
    uint16_t size0 = rowLen / perVfColNum_;
    uint32_t stride = main;
    uint32_t tail = rowLen * colLen - size0 * main;
    uint32_t strideTail = tail;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> index;
        MicroAPI::RegTensor<U> indexUpd;

        MicroAPI::UnalignReg uDstA;
        MicroAPI::UnalignReg uDstB;

        MicroAPI::RegTensor<T> vregA;
        MicroAPI::RegTensor<T> vregB;

        MicroAPI::DataCopy(index, indexAddr);

        MicroAPI::MaskReg pMain = MicroAPI::UpdateMask<T>(main);
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<T>(tail);

        for (uint16_t i = 0; i < size0; ++i) {
            MicroAPI::Adds(indexUpd, index, (U)(i * stride * SPLIT_HALF), pMain);

            MicroAPI::DataCopyGather(vregA, (__local_mem__ T*)(ubSrcAddrA), indexUpd, pMain);
            MicroAPI::DataCopyGather(vregB, (__local_mem__ T*)(ubSrcAddrB), indexUpd, pMain);

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrA, vregA, uDstA, stride);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrB, vregB, uDstB, stride);
        }

        MicroAPI::DataCopyUnAlignPost(ubDstAddrA, uDstA, 0);
        MicroAPI::DataCopyUnAlignPost(ubDstAddrB, uDstB, 0);

        MicroAPI::Adds(indexUpd, index, (U)(size0 * stride * SPLIT_HALF), pTail);

        MicroAPI::DataCopyGather(vregA, (__local_mem__ T*)(ubSrcAddrA), indexUpd, pTail);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrA, vregA, uDstA, strideTail);
        MicroAPI::DataCopyUnAlignPost(ubDstAddrA, uDstA, 0);

        MicroAPI::DataCopyGather(vregB, (__local_mem__ T*)(ubSrcAddrB), indexUpd, pTail);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrB, vregB, uDstB, strideTail);
        MicroAPI::DataCopyUnAlignPost(ubDstAddrB, uDstB, 0);
    }

    inQueX_.FreeTensor(xLocal);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::RearrangeUbOutByScatter(uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> outLocal = outQueXGrad_.AllocTensor<T>();
    LocalTensor<T> calcALocal = OutApart_.Get<T>();
    LocalTensor<T> calcBLocal = OutBpart_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();

    auto ubDstAddrA = (__ubuf__ T*)outLocal.GetPhyAddr();
    auto ubDstAddrB = (__ubuf__ T*)outLocal.GetPhyAddr(this->colProcess_);
    auto ubSrcAddrA = (__ubuf__ T*)calcALocal.GetPhyAddr();
    auto ubSrcAddrB = (__ubuf__ T*)calcBLocal.GetPhyAddr();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    uint16_t size0 = rowLen / perVfColNum_;
    uint32_t main = gatherNormNum_;
    uint32_t tail = rowLen * colLen - size0 * main;

    uint32_t stride = main;
    uint32_t strideTail = tail;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> index;
        MicroAPI::RegTensor<U> indexUpd;

        MicroAPI::RegTensor<T> vregA;
        MicroAPI::RegTensor<T> vregB;

        MicroAPI::UnalignReg uSrcA;
        MicroAPI::UnalignReg uSrcB;

        MicroAPI::DataCopy(index, indexAddr);

        MicroAPI::MaskReg pMain = MicroAPI::UpdateMask<T>(main);
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<T>(tail);

        MicroAPI::DataCopyUnAlignPre(uSrcA, ubSrcAddrA);
        MicroAPI::DataCopyUnAlignPre(uSrcB, ubSrcAddrB);
        for (uint16_t i = 0; i < size0; ++i) {
            MicroAPI::Adds(indexUpd, index, (U)(i * stride * SPLIT_HALF), pMain);

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregA, uSrcA, ubSrcAddrA, stride);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregB, uSrcB, ubSrcAddrB, stride);

            MicroAPI::DataCopyScatter(ubDstAddrA, vregA, indexUpd, pMain);
            MicroAPI::DataCopyScatter(ubDstAddrB, vregB, indexUpd, pMain);
        }

        MicroAPI::Adds(indexUpd, index, (U)(size0 * stride * SPLIT_HALF), pTail);

        MicroAPI::DataCopyUnAlignPre(uSrcA, ubSrcAddrA);
        MicroAPI::DataCopyUnAlignPre(uSrcB, ubSrcAddrB);

        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregA, uSrcA, ubSrcAddrA, strideTail);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregB, uSrcB, ubSrcAddrB, strideTail);

        MicroAPI::DataCopyScatter((__local_mem__ T*)(ubDstAddrA), vregA, indexUpd, pTail);
        MicroAPI::DataCopyScatter((__local_mem__ T*)(ubDstAddrB), vregB, indexUpd, pTail);
    }

    outQueXGrad_.EnQue<T>(outLocal);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::ProcessNotGather()
{
    int64_t offset = 0;

    uint32_t dataCount = this->colProcess_ * rowUbNorm_;
    uint32_t loopStride = dataCount * SPLIT_HALF;

    for (int64_t i = 0; i < rowLoopByUb_ - 1; i++) {
        CopyIn(offset, rowUbNorm_, this->colProcess_);
        RearrangeUbIn(rowUbNorm_, this->colProcess_);
        ComputePerLoop(dataCount);
        RearrangeUbOut(rowUbNorm_, this->colProcess_);
        CopyOut(offset, rowUbNorm_, this->colTotal_);
        offset = offset + loopStride;
    }

    dataCount = this->colProcess_ * rowUbTail_;
    CopyIn(offset, rowUbTail_, this->colProcess_);
    RearrangeUbIn(rowUbTail_, this->colProcess_);
    ComputePerLoop(dataCount);
    RearrangeUbOut(rowUbTail_, this->colProcess_);
    CopyOut(offset, rowUbTail_, this->colTotal_);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::ProcessWithGather()
{
    int64_t offset = 0;
    GenerateGatherIndex(this->colProcess_);

    uint32_t dataCount = this->colProcess_ * rowUbNorm_;
    uint32_t loopStride = dataCount * SPLIT_HALF;

    for (int64_t i = 0; i < rowLoopByUb_ - 1; i++) {
        CopyIn(offset, rowUbNorm_, this->colProcess_);
        RearrangeUbInByGather(rowUbNorm_, this->colProcess_);
        ComputePerLoop(dataCount);
        RearrangeUbOutByScatter(rowUbNorm_, this->colProcess_);
        CopyOut(offset, rowUbNorm_, this->colTotal_);
        offset = offset + loopStride;
    }

    dataCount = this->colProcess_ * rowUbTail_;
    CopyIn(offset, rowUbTail_, this->colProcess_);
    RearrangeUbInByGather(rowUbTail_, this->colProcess_);
    ComputePerLoop(dataCount);
    RearrangeUbOutByScatter(rowUbTail_, this->colProcess_);
    CopyOut(offset, rowUbTail_, this->colTotal_);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::ComputePerLoop(uint32_t dataCount)
{
    LocalTensor<T> gradTensor = inQueGrad_.DeQue<T>();
    LocalTensor<T> aLocal = Apart_.Get<T>();
    LocalTensor<T> bLocal = Bpart_.Get<T>();
    LocalTensor<T> calcTempA = OutApart_.Get<T>();
    LocalTensor<T> calcTempB = OutBpart_.Get<T>();

    this->Compute(aLocal, bLocal, gradTensor, calcTempA, calcTempB, dataCount);
    inQueGrad_.FreeTensor(gradTensor);
}

template <typename T, typename U>
__aicore__ inline void SwiGluGradUbRearrangeKernel<T, U>::Process()
{
    if (!needGather_) {
        ProcessNotGather();
    } else {
        ProcessWithGather();
    }
}

} // namespace SwiGluGrad

#endif