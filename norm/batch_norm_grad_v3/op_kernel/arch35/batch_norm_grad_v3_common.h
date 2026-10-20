/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_v3_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_V3_COMMON_H__
#define __BATCH_NORM_GRAD_V3_COMMON_H__

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormGradV3 {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

static constexpr uint32_t ONE = 1;
static constexpr uint32_t TWO = 2;
static constexpr uint32_t THREE = 3;
static constexpr uint32_t FOUR = 4;

static constexpr int64_t OUTPUT_MASK_NONE = 0;
static constexpr int64_t OUTPUT_MASK_DX = 1;
static constexpr int64_t OUTPUT_MASK_DGAMMA = OUTPUT_MASK_DX << ONE;
static constexpr int64_t OUTPUT_MASK_DBETA = OUTPUT_MASK_DGAMMA << ONE;
static constexpr int64_t OUTPUT_MASK_ALL = OUTPUT_MASK_DX | OUTPUT_MASK_DGAMMA | OUTPUT_MASK_DBETA;
static constexpr int64_t MERGED_QUE_NODE_NUM = 2;
static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
static constexpr uint32_t TWO_BLK_SIZE = ONE_BLK_SIZE * TWO;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint64_t MOVE_ALIGN_LOOP_SIZE_SHIFT = 21;
static constexpr uint64_t MOVE_ALIGN_LOOP_STRIDE_SHIFT = 40;
static constexpr uint64_t MOVE_ALIGN_LOOP_ONCE = 2097153ULL;
static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t RoundUpOneBlock(uint32_t x) { return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE; }

__aicore__ inline uint32_t RoundUpTwoBlock(uint32_t x) { return (x + TWO_BLK_SIZE - 1) / TWO_BLK_SIZE * TWO_BLK_SIZE; }

static constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTraitB322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadOneTensor(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                     uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, (__ubuf__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, (__ubuf__ float*)(input) + offset);
    }
}

template <typename T>
__aicore__ inline void LoadsTensorForDtypeT(const __ubuf__ void* src, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                            uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, (__ubuf__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(xFp16, ((__ubuf__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void StoreOneTensor(const __ubuf__ void* output, Reg::RegTensor<float>& src, Reg::MaskReg& preg,
                                      uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ bfloat16_t*)(output) + offset, xBf16, preg);
    } else {
        StoreAlign((__ubuf__ float*)(output) + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void LoadOneElement(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                      uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        LoadAlign<half, Reg::LoadDist::DIST_BRC_B16>(xFp16, (__ubuf__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dst, ((__ubuf__ float*)(input)) + offset);
    }
}

template <typename T>
__aicore__ inline void StoreOneElement(const __ubuf__ void* output, Reg::RegTensor<float>& src, Reg::MaskReg& preg,
                                       uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<half, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        StoreAlign<bfloat16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ bfloat16_t*)(output) + offset, xBf16,
                                                                       preg);
    } else {
        StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)output) + offset, src, preg);
    }
}

__aicore__ inline void LoadTwoTensorSum(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                        uint32_t offset0, uint32_t offset1)
{
    Reg::RegTensor<float> a, b;
    LoadAlign(a, (__ubuf__ float*)(input) + offset0);
    LoadAlign(b, (__ubuf__ float*)(input) + offset1);
    Add<float, Reg::MaskMergeMode::ZEROING>(dst, a, b, preg);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t n)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t n)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyInNormParam(LocalTensor<T>& localTensor, GlobalTensor<T>& globalTensor, uint64_t offset,
                                       uint64_t a)
{
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = a * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(localTensor, globalTensor[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutNormParam(LocalTensor<T>& localTensor, GlobalTensor<T>& globalTensor, uint64_t offset,
                                        uint64_t a)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = a * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(globalTensor[offset], localTensor, copyParams);
}

template <typename T2, typename T3, int32_t bufferDepth>
__aicore__ inline void CopyInGammaVarCommon(TQue<QuePosition::VECIN, bufferDepth>& gammaQueue,
                                            TQue<QuePosition::VECIN, bufferDepth>& runningVarQueue,
                                            GlobalTensor<T2>& gammaGm, GlobalTensor<T3>& runningVarGm, bool needCopy,
                                            int64_t offset, int64_t curTileALen)
{
    LocalTensor<T2> gammaLocal = gammaQueue.template AllocTensor<T2>();
    LocalTensor<T3> varLocal = runningVarQueue.template AllocTensor<T3>();

    if (needCopy) {
        DataCopyExtParams extParam;
        extParam.blockCount = 1;

        extParam.blockLen = curTileALen * sizeof(T2);
        DataCopyPadExtParams<T2> gammaPadParam;
        gammaPadParam.isPad = false;
        DataCopyPad(gammaLocal, gammaGm[offset], extParam, gammaPadParam);

        extParam.blockLen = curTileALen * sizeof(T3);
        DataCopyPadExtParams<T3> varPadParam;
        varPadParam.isPad = false;
        DataCopyPad(varLocal, runningVarGm[offset], extParam, varPadParam);
    }

    gammaQueue.EnQue(gammaLocal);
    runningVarQueue.EnQue(varLocal);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t rowSize,
                               int64_t colSize, int64_t dstStride, int64_t srcStride)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.dstStride = dstStride * sizeof(T) - params.blockLen;
    params.srcStride = (srcStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t rowSize,
                              int64_t colSize, int64_t dstStride, int64_t srcStride)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.srcStride = srcStride * sizeof(T) - params.blockLen;
    params.dstStride = (dstStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

// RC packed data movement and channel folding shared by the RA templates.
template <typename T>
__aicore__ inline void PackedCopyChannel(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t channels,
                                         uint32_t k)
{
    NdDmaLoopInfo<2> loop;
    loop.loopSize[0] = channels;
    loop.loopSrcStride[0] = 1;
    loop.loopDstStride[0] = 1;
    loop.loopSize[1] = k;
    loop.loopSrcStride[1] = 0;
    loop.loopDstStride[1] = channels;
    static constexpr NdDmaConfig config = {false};
    NdDmaParams<T, 2> params = {loop, static_cast<T>(0)};
    DataCopy<T, 2, config>(dst, src, params);
}

template <typename T>
__aicore__ inline void PackedCopyInput(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t rows,
                                       uint32_t channels, uint32_t k, bool dense = false)
{
    if (dense) {
        DataCopyExtParams copy{};
        copy.blockCount = 1;
        copy.blockLen = rows * channels * sizeof(T);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, copy, pad);
        return;
    }
    uint32_t lanes = k * channels;
    uint32_t groups = rows / k;
    uint32_t tail = rows % k * channels;
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyExtParams copy{};
    copy.blockCount = groups;
    copy.blockLen = lanes * sizeof(T);
    copy.dstStride = (VL_FP32 - lanes) * sizeof(T) / ONE_BLK_SIZE;
    if (groups != 0) {
        DataCopyPad(dst, src, copy, pad);
    }
    if (tail != 0) {
        copy.blockCount = 1;
        copy.blockLen = tail * sizeof(T);
        copy.dstStride = 0;
        DataCopyPad(dst[groups * VL_FP32], src[groups * lanes], copy, pad);
    }
}

template <typename T>
__aicore__ inline void PackedCopyOutput(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t rows,
                                        uint32_t channels, uint32_t k)
{
    uint32_t lanes = k * channels;
    uint32_t groups = rows / k;
    uint32_t tail = rows % k * channels;
    DataCopyExtParams copy{};
    copy.blockCount = groups;
    copy.blockLen = lanes * sizeof(T);
    copy.srcStride = (VL_FP32 - lanes) * sizeof(T) / ONE_BLK_SIZE;
    if (groups != 0) {
        DataCopyPad<T, PaddingMode::Normal>(dst, src, copy);
    }
    if (tail != 0) {
        copy.blockCount = 1;
        copy.blockLen = tail * sizeof(T);
        copy.srcStride = 0;
        DataCopyPad<T, PaddingMode::Normal>(dst[groups * lanes], src[groups * VL_FP32], copy);
    }
}

// K is a power of two. Each lane keeps its channel while the K rows are folded.
__aicore__ inline void PackedFoldChannels(Reg::RegTensor<float>& value, uint32_t channels, uint16_t k)
{
    Reg::RegTensor<int32_t> seq;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> right;
    Reg::Arange(seq, 0);
    for (uint16_t half = k / 2; half != 0; half /= 2) {
        uint32_t count = half * channels;
        Reg::MaskReg valid = Reg::UpdateMask<float>(count);
        Reg::Adds<uint32_t, uint32_t, Reg::MaskMergeMode::ZEROING>(index, (Reg::RegTensor<uint32_t>&)seq,
                                                                   half * channels, valid);
        Reg::Gather(right, value, index);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(value, value, right, valid);
    }
}

// Reverse the fold's layout to broadcast the final C gradients to all K rows.
__aicore__ inline void PackedBroadcastChannels(Reg::RegTensor<float>& value, uint32_t channels, uint16_t k)
{
    Reg::RegTensor<int32_t> seq, index, shifted;
    Reg::RegTensor<float> next;
    Reg::Arange(seq, 0);
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t copies = 1; copies < k; copies *= 2) {
        uint32_t count = copies * channels;
        Reg::MaskReg low = Reg::UpdateMask<float>(count);
        Reg::Adds(index, seq, -static_cast<int32_t>(copies * channels), all);
        Reg::Select(shifted, seq, index, low);
        // Selected indices are nonnegative and always below VL_FP32.
        Reg::Gather(next, value, (Reg::RegTensor<uint32_t>&)shifted);
        Reg::Move(value, next, all);
    }
}

__aicore__ inline int64_t GetCacheID(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }

} // namespace BatchNormGradV3
#endif
