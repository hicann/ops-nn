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
 * \file swi_glu_grad_base.h
 * \brief
 */

#ifndef SWI_GLU_GRAD_BASE_H
#define SWI_GLU_GRAD_BASE_H
#include "kernel_operator.h"
#include "../inc/platform.h"


namespace SwiGluGrad {
using namespace AscendC;

const int32_t DOUBLE_BUFFER = 2;
const int32_t BUFFER_NUMBER = 5;
const uint32_t SPLIT_HALF = 2;

static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t ONE_VL_SIZE = platform::GetVRegSize();

__aicore__ inline uint32_t RoundDownToVL(uint32_t x)
{
    return x / ONE_VL_SIZE * ONE_VL_SIZE;
}

static constexpr MicroAPI::CastTrait castTraitB162B32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };

static constexpr MicroAPI::CastTrait castTraitB322B16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

template <typename T>
class SwiGluGradBaseKernel {
public:
    __aicore__ inline SwiGluGradBaseKernel() = default;
    __aicore__ inline bool InitBase(const GluBaseTilingData& tilingData);

protected:
    __aicore__ inline void LoadOneTensor(const __local_mem__ void *input, MicroAPI::RegTensor<float> &dst, MicroAPI::MaskReg &preg, uint32_t offset);
    __aicore__ inline void StoreOneTensor(const __local_mem__ void *output, MicroAPI::RegTensor<float> &src, MicroAPI::MaskReg &preg, uint32_t offset);
    __aicore__ inline void Compute(LocalTensor<T> &xATensor, LocalTensor<T> &xBTensor, LocalTensor<T> &gradTensor,
                                            LocalTensor<T> &outATensor, LocalTensor<T> &outBTensor, int64_t dataCount);

protected:
    // tiling Data
    int64_t rowTotalA_ {0}; // A部分的总行宽
    int64_t colTotalA_ {0}; // A部分的总列宽
    int64_t colTotal_ {0};  // A + B总列宽
    int64_t rowBase_ {0};   // 正常核处理行大小
    int64_t colBase_ {0};
    int64_t rowTail_ {0};   // 尾核处理行大小
    int64_t colTail_ {0};
    uint64_t ubSize_ {0};
    uint32_t rowTileNum_ {0};   // A部分行方向切分后个数
    uint32_t colTileNum_ {0};   // A部分列方向切分后个数
    uint32_t usedCoreNum_ {0};

    // vars
    int64_t rowProcess_ {0};
    int64_t colProcess_ {0};
    int64_t partAStart_ {0};
    int64_t partBStart_ {0};
    int64_t gradStart_ {0};
    int32_t blockIdx_ {0};

    float beta_ = -1.0;
    static constexpr uint16_t vfFp32Block_ = platform::GetVRegSize() / sizeof(float);
    static constexpr uint16_t vfBlock_ = platform::GetVRegSize() / sizeof(T);
};

template <typename T>
__aicore__ inline bool SwiGluGradBaseKernel<T>::InitBase(const GluBaseTilingData& tilingData)
{
    blockIdx_ = GetBlockIdx();
    usedCoreNum_ = tilingData.usedCoreNum;
    if (blockIdx_ >= usedCoreNum_) {
        return false;
    }

    rowTotalA_ = tilingData.rowTotal;
    colTotalA_ = tilingData.colTotal;
    rowBase_ = tilingData.rowBase;
    colBase_ = tilingData.colBase;
    rowTail_ = tilingData.rowTail;
    colTail_ = tilingData.colTail;
    rowTileNum_ = tilingData.rowTileNum;
    colTileNum_ = tilingData.colTileNum;
    ubSize_ = tilingData.ubSize;

    colTotal_ = colTotalA_ * SPLIT_HALF;

    // 本核实际处理数据量
    colProcess_ = (blockIdx_ + 1) % colTileNum_ != 0 ? colBase_ : colTail_;
    rowProcess_ = (blockIdx_ + 1) / colTileNum_ < (rowTileNum_ - 1) ? rowBase_ : rowTail_;

    // 本核所处位置，优先列方向分核
    int64_t colOffset = blockIdx_ % colTileNum_;
    int64_t rowOffset = blockIdx_ / colTileNum_;

    partAStart_ = rowBase_ * rowOffset * colTotal_ + colBase_ * colOffset;
    partBStart_ = partAStart_ + colTotalA_;
    gradStart_ = rowBase_ * rowOffset * colTotalA_ + colBase_ * colOffset;

    return true;
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::LoadOneTensor(const __local_mem__ void *input, MicroAPI::RegTensor<float> &dst,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half *)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16, (__local_mem__ bfloat16_t *)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, (__local_mem__ float *)(input) + offset);
    }
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::StoreOneTensor(const __local_mem__ void *output, MicroAPI::RegTensor<float> &src,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>((__local_mem__ half *)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>((__local_mem__ bfloat16_t *)(output) + offset, xBf16,
            preg);
    } else {
        DataCopy((__local_mem__ float *)(output) + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::Compute(LocalTensor<T> &xATensor, LocalTensor<T> &xBTensor, LocalTensor<T> &gradTensor,
                                        LocalTensor<T> &outATensor, LocalTensor<T> &outBTensor, int64_t dataCount)
{
    __local_mem__ T *ubSrcAddrA = (__local_mem__ T *)xATensor.GetPhyAddr();
    __local_mem__ T *ubSrcAddrB = (__local_mem__ T *)xBTensor.GetPhyAddr();
    __local_mem__ T *ubGradAddr = (__local_mem__ T *)gradTensor.GetPhyAddr();
    __local_mem__ T *ubDstAddrA = (__local_mem__ T *)outATensor.GetPhyAddr();
    __local_mem__ T *ubDstAddrB = (__local_mem__ T *)outBTensor.GetPhyAddr();

    uint16_t repeatTimes = (dataCount + vfFp32Block_ - 1) / vfFp32Block_;

    float beta = beta_;
    uint32_t calcCount = dataCount;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> srcA;
        MicroAPI::RegTensor<float> srcB;
        MicroAPI::RegTensor<float> grad;
        MicroAPI::RegTensor<float> dstA;
        MicroAPI::RegTensor<float> dstB;
        MicroAPI::RegTensor<float> tempReg;

        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(calcCount);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            uint32_t offset = i * vfFp32Block_;
            LoadOneTensor(ubSrcAddrA, srcA, preg, offset);
            LoadOneTensor(ubSrcAddrB, srcB, preg, offset);
            LoadOneTensor(ubGradAddr, grad, preg, offset);

            // sigmoid
            Muls(dstB, srcA, beta, preg);
            Exp(dstB, dstB, preg);
            Adds(dstB, dstB, -beta, preg);
            Duplicate(dstA, -beta, preg);
            Div(dstB, dstA, dstB, preg);

            // outA part
            Muls(dstA, dstB, beta, preg);   // -sigmoid
            Adds(dstA, dstA, -beta, preg);  // 1 - sigmoid
            Mul(dstA, dstA, srcA, preg); // (1 - sigmoid) * A
            Adds(dstA, dstA, -beta, preg); // 1 + (1 - sigmoid) * A
            Mul(tempReg, grad, dstB, preg); // grad * sigmoid
            Mul(dstA, dstA, tempReg, preg); // grad * sigmoid * (1 + (1 - sigmoid) * A)
            Mul(dstA, dstA, srcB, preg); // grad * sigmoid * (1 + (1 - sigmoid) * A) * B

            // outB part
            Mul(dstB, dstB, grad, preg); // grad * sigmoid
            Mul(dstB, dstB, srcA, preg);   // grad * sigmoid * A

            StoreOneTensor(ubDstAddrA, dstA, preg, offset);
            StoreOneTensor(ubDstAddrB, dstB, preg, offset);
        }
    }
}

}

#endif