/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file dynamic_quant_regbase_moe_full_load.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_RECOMPUTE_H
#define DYNAMIC_QUANT_REGBASE_RECOMPUTE_H

#include "dynamic_quant_regbase_perchannel_base.h"
#include "dynamic_quant_regbase_base.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"


namespace DynamicQuantPerChannel {
using namespace AscendC;

constexpr uint16_t FP16_UB_ALIGNED_NUM = 32 / sizeof(half);
constexpr uint16_t INT8_UB_ALIGNED_NUM = 32;
constexpr uint16_t INT4_UB_ALIGNED_NUM = 64;
constexpr uint16_t FP32_VF_ALIGNED_NUM = 256 / sizeof(float);

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
class DynamicQuantRegbasePerChannnelRecompute {
    // 如果输出的数据类型是INT4，用INT8处理，其余的输出类型不变
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;

public:
    __aicore__ inline DynamicQuantRegbasePerChannnelRecompute(TPipe* pipe)
    {
        pPipe = pipe;
    }

    // 相比V1新增了group_index输入
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantTilingData* __restrict tilingData)
    {
        DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
        coreIndex_ = GetBlockIdx();
        ParseTilingData(*tilingData);
        if (coreIndex_ >= coreNum_)
            return;
        SetMaxValue<yDtype>(maxValue_, offsetValue_, offsetDivValue_);
        // 计算参数
        currBlockNumOnCore_ = coreIndex_ < headCoreNum_ ? blockPerHead_ : blockPerTail_;
        blockStartIndex_ = coreIndex_ < headCoreNum_ ?
                               coreIndex_ * blockPerHead_ :
                               headCoreNum_ * blockPerHead_ + (coreIndex_ - headCoreNum_) * blockPerTail_;
        blockEndIndex_ = blockStartIndex_ + currBlockNumOnCore_;
        InitAndSetBuffer(x, smooth_scales, y, scale, offset);
    }

    __aicore__ inline void Process()
    {
        if (coreIndex_ >= coreNum_) {
            return;
        }

        for (uint32_t blockIdx = blockStartIndex_; blockIdx < blockEndIndex_; blockIdx++) {
            // 当前block所在的batch的编号
            uint32_t batchIndex = blockIdx / nBlockNum_;
            // 当前切块在对应batch中的局部编号
            uint32_t localBatchIndex = blockIdx % nBlockNum_;
            // 当前切块对应的列数
            uint32_t currColNumForBlock = (localBatchIndex + 1) == nBlockNum_ ? nTailBlockSize_ : nBlockSize_;
            LoopProcess(currColNumForBlock, batchIndex, localBatchIndex);
        }
    }

private:
    TPipe* pPipe = nullptr;
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData& tilingData);
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset);

    /** @brief 处理每个mLen * nBaseSize大小的块 */
    __aicore__ inline void LoopProcess(uint32_t blockColNum, uint32_t batchIndex, uint32_t localBatchIndex)
    {
        uint32_t newColNum = (localBatchIndex + 1) == nBlockNum_ ? nTailBlockSizeAligned_ : nBlockSize_;
        uint32_t newColNumB8 = (localBatchIndex + 1) == nBlockNum_ ? nTailBlockSizeAlignedB8_ : nBlockSize_;
        uint64_t localBatchOffset = localBatchIndex * nBlockSize_;
        uint64_t xStartOffset = batchIndex * mLen_ * nLen_ + localBatchOffset;
        uint64_t scaleOffset = batchIndex * nLen_ + localBatchOffset; // scale和offset共用一个地址偏移量

        LocalTensor<float> colMaxLocal = colMaxQueue_.AllocTensor<float>();
        LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        LocalTensor<float> colMinLocal;
        LocalTensor<float> offsetLocal;
        if constexpr (isSymmetrical == false) {
            colMinLocal = colMinQueue_.AllocTensor<float>();
            offsetLocal = offsetQueue_.AllocTensor<float>();
        }

        uint32_t xShift = mBlockSize_ * nLen_;
        uint32_t smoothShift = mBlockSize_;
        uint64_t smoothOffset = 0;
        uint64_t xOffset = xStartOffset;

        for (uint32_t mBaseIndex = 0; mBaseIndex < mBlockNum_; mBaseIndex++) {
            uint32_t currRowNum = mBaseIndex + 1 == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;
            // copyIn使用未padding时的列宽
            // inQueue申请并入队、smoothQueue申请并入队
            CopyIn(currRowNum, blockColNum, xOffset, smoothOffset);
            // 计算max和scale
            // inQueue出队、smoothQueue出队，smoothQueue释放，inQueue释放
            ComputeInner(colMaxLocal, colMinLocal, scaleLocal, offsetLocal, mBaseIndex, currRowNum, newColNum);
            xOffset += xShift;
            smoothOffset += smoothShift;
        }

        smoothOffset = 0;
        xOffset = xStartOffset;
        for (uint32_t mBaseIndex = 0; mBaseIndex < mBlockNum_; mBaseIndex++) {
            uint32_t currRowNum = mBaseIndex + 1 == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;
            // copyIn使用未padding时的列宽
            // inQueue申请并入队、smoothQueue申请并入队
            CopyIn(currRowNum, blockColNum, xOffset, smoothOffset);
            // 计算y和offset
            // inQueue出队、smoothQueue出队，outQueue申请和入队，smoothQueue释放，inQueue释放
            ComputeOut(
                colMaxLocal, colMinLocal, scaleLocal, offsetLocal, mBaseIndex, currRowNum, newColNum, newColNumB8);
            // outQueue出队并释放
            CopyOutY(currRowNum, blockColNum, xOffset);
            xOffset += xShift;
            smoothOffset += smoothShift;
        }
        scaleQueue_.EnQue(scaleLocal);
        if constexpr (isSymmetrical == false) {
            offsetQueue_.EnQue(offsetLocal);
        }
        CopyOutScaleAndOffset(blockColNum, scaleOffset);
        if constexpr (isSymmetrical == false) {
            colMinQueue_.FreeTensor(colMinLocal);
        }
        colMaxQueue_.FreeTensor(colMaxLocal);
    }

    __aicore__ inline void CopyIn(uint32_t blockRowNum, uint32_t blockColNum, uint64_t xOffset, uint64_t smoothOffset);

    __aicore__ inline void ComputeInner(
        LocalTensor<float>& colMaxLocal, LocalTensor<float>& colMinLocal, LocalTensor<float>& scaleLocal,
        LocalTensor<float>& offsetLocal, uint32_t mBlockIndex, uint32_t blockRowNum, uint32_t blockColNum);

    __aicore__ inline void ComputeOut(
        LocalTensor<float>& colMaxLocal, LocalTensor<float>& colMinLocal, LocalTensor<float>& scaleLocal,
        LocalTensor<float>& offsetLocal, uint32_t mBlockIndex, uint32_t blockRowNum, uint32_t blockColNumForX,
        uint32_t blockColNumForY);

    template <bool isLastBlock>
    __aicore__ inline void ComputeMaxVFforSymmetric(
        __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
        __local_mem__ float* colMaxLocalAddr, uint32_t blockRowNum, uint32_t blockColNum);

    __aicore__ inline void ComputeVFMinMaxforNoSymmetric(
        __ubuf__ xDtype* inAddr, __ubuf__ xDtype* smoothAddr, __ubuf__ float* scaleAddr,
        __ubuf__ float* colMaxLocalAddr, __ubuf__ float* colMinLocalAddr, __ubuf__ float* offsetAddr, uint32_t curBaseM,
        uint32_t curBlockSize);

    __aicore__ inline void ComputeVFforSymmetric(
        __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
        __local_mem__ float* colMaxLocalAddr, __local_mem__ yCopyDtype* yAddr, uint32_t blockRowNum,
        uint32_t blockColNumForX, uint32_t blockColNumForY);

    __aicore__ inline void ComputeVFforNoSymmetric(
        __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
        __local_mem__ float* colMaxLocalAddr, __local_mem__ float* colMinLocalAddr, __local_mem__ yCopyDtype* yAddr,
        __local_mem__ float* offsetAddr, uint32_t blockRowNum, uint32_t blockColNumForX, uint32_t blockColNumForY);

    __aicore__ inline void CopyOutScaleAndOffset(uint32_t blockColNum, uint64_t scaleOffset);

    __aicore__ inline void CopyOutY(uint32_t blockRowNum, uint32_t blockColNum, uint64_t xOffset);

    /* tiling data */
    DynamicQuantTilingData tilingData_;

    /* ascendc variable */
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> smoothQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> colMaxQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> colMinQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> offsetQueue_;

    /* global memory address */
    GlobalTensor<xDtype> inGm_, smoothGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<yCopyDtype> outGm_;

    /* variable */
    uint32_t coreIndex_ = 0; // 当前核的索引
    uint32_t coreNum_ = 0;
    uint32_t hasSmooth_ = 0;
    uint32_t blockStartIndex_ = 0; // 当前核上切块的起始索引
    uint32_t blockEndIndex_ = 0;   // 当前核上切块的结束索引（不包含该索引对应的块）
    uint32_t headCoreNum_ = 0;
    uint32_t totalBatchLen_ = 0;
    uint32_t mLen_ = 0;
    uint32_t mBlockSize_ = 0;
    uint32_t mTailBlockSize_ = 0;
    uint32_t mBlockNum_ = 0;
    uint32_t nLen_ = 0;
    uint32_t nBlockSize_ = 0;
    uint32_t nBaseSize_ = 0;
    uint32_t nTailBlockSize_ = 0;
    uint32_t nTailBlockSizeAligned_ = 0;
    uint32_t nTailBlockSizeAlignedB8_ = 0;
    uint32_t nTailBlockSizePadding_ = 0;
    uint32_t nBlockNum_ = 0;
    uint32_t nBaseLoopNum_ = 0;
    uint32_t blockPerHead_ = 0;
    uint32_t blockPerTail_ = 0;
    uint32_t totalBlockNum_ = 0;
    uint32_t currBlockNumOnCore_ = 0;

    float maxValue_ = 0.0;
    float offsetValue_ = 0.0;
    float offsetDivValue_ = 0.0;
    uint32_t outBufferSize_ = 0;
};

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ParseTilingData(
    const DynamicQuantTilingData& tilingData)
{
    coreNum_ = tilingData.coreNum;
    headCoreNum_ = tilingData.headCoreNum;
    hasSmooth_ = tilingData.hasSmooth;
    totalBatchLen_ = tilingData.totalBatchLen;   // batch轴合轴后的总大小
    mLen_ = tilingData.mLen;                     // m轴大小
    mBlockSize_ = tilingData.mBlockSize;         // 单次切分，UB内可放的M轴大小
    mTailBlockSize_ = tilingData.mTailBlockSize; // 按mBlockSize切分M轴时，尾块的M轴大小
    mBlockNum_ = tilingData.mBlockNum;           // 对每个batch，M轴切分的循环次数
    nLen_ = tilingData.nLen;                     // n轴大小
    nBlockSize_ = tilingData.nBlockSize; // 单次切分，UB内可放的N轴大小，为nBaseSize的整数倍(BaseN)
    nTailBlockSize_ = tilingData.nTailBlockSize; // 按nBlockSize切分N轴时，尾块的N轴大小
    nTailBlockSizeAligned_ = ops::CeilAlign(nTailBlockSize_, static_cast<uint32_t>(FP16_UB_ALIGNED_NUM));
    uint32_t alignB8Num = INT8_UB_ALIGNED_NUM;
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        alignB8Num = INT4_UB_ALIGNED_NUM;
    }
    nTailBlockSizeAlignedB8_ = ops::CeilAlign(nTailBlockSize_, alignB8Num);
    nTailBlockSizePadding_ = nTailBlockSizeAligned_ - nTailBlockSize_;
    nBlockNum_ = tilingData.nBlockNum;         // 对每个batch，N轴切分的循环次数
    nBaseSize_ = tilingData.nBaseSize;         // 对N轴切分，最小并行的元素个数(256/2 = 128)
    nBaseLoopNum_ = tilingData.nBaseLoopNum;   // vf内N轴的循环次数，nBaseLoopNum = nBlockSize / nBaseSize
    blockPerHead_ = tilingData.blockPerHead;   // 大核处理的块数
    blockPerTail_ = tilingData.blockPerTail;   // 小核处理的块数
    totalBlockNum_ = tilingData.totalBlockNum; // 总共需要处理几个块
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        outBufferSize_ = mBlockSize_ * nBlockSize_ / 2;
    } else {
        outBufferSize_ = mBlockSize_ * nBlockSize_;
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::InitAndSetBuffer(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
{
    inGm_.SetGlobalBuffer((__gm__ xDtype*)x);
    outGm_.SetGlobalBuffer((__gm__ yCopyDtype*)y);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

    if constexpr (hasSmooth) {
        smoothGm_.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
        // smooth_scale is col-wise
        pPipe->InitBuffer(smoothQueue_, USE_BUFFER_NUM, mBlockSize_ * sizeof(xDtype));
    }
    pPipe->InitBuffer(inQueue_, USE_BUFFER_NUM, mBlockSize_ * nBlockSize_ * sizeof(xDtype));
    pPipe->InitBuffer(colMaxQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    pPipe->InitBuffer(outQueue_, USE_BUFFER_NUM, outBufferSize_ * sizeof(yCopyDtype));
    pPipe->InitBuffer(scaleQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    if constexpr (isSymmetrical == false) {
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset);
        pPipe->InitBuffer(offsetQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
        pPipe->InitBuffer(colMinQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyIn(
    uint32_t blockRowNum, uint32_t blockColNum, uint64_t xOffset, uint64_t smoothOffset)
{
    uint32_t copyPadNumPerRow = blockColNum == nBlockSize_ ? 0 : nTailBlockSizePadding_;
    LocalTensor<xDtype> inLocal = inQueue_.AllocTensor<xDtype>();
    // 修改stride，srcStride改成(nLen - blockColNum) * sizeof(xDtype)
    uint32_t srcStrideNum = nLen_ - blockColNum;
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(blockRowNum), static_cast<uint32_t>(blockColNum * sizeof(xDtype)),
        static_cast<int64_t>(srcStrideNum * sizeof(xDtype)), 0, 0};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(copyPadNumPerRow), 0};
    DataCopyPad(inLocal, inGm_[xOffset], copyParams, padParams);
    inQueue_.EnQue(inLocal);

    if constexpr (hasSmooth) {
        LocalTensor<xDtype> smoothLocal = smoothQueue_.AllocTensor<xDtype>();
        DataCopyExtParams smoothCopyParams = {1, static_cast<uint32_t>(blockRowNum * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> smoothPadParams{true, 0, 0, 0};
        DataCopyPad(smoothLocal, smoothGm_[smoothOffset], smoothCopyParams, smoothPadParams);
        smoothQueue_.EnQue(smoothLocal);
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeInner(
    LocalTensor<float>& colMaxLocal, LocalTensor<float>& colMinLocal, LocalTensor<float>& scaleLocal,
    LocalTensor<float>& offsetLocal, uint32_t mBlockIndex, uint32_t blockRowNum, uint32_t blockColNum)
{
    __ubuf__ float* colMaxLocalAddr = (__ubuf__ float*)colMaxLocal.GetPhyAddr();
    __ubuf__ float* colMinLocalAddr = (__ubuf__ float*)colMinLocal.GetPhyAddr();
    __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __ubuf__ float* offsetAddr = (__ubuf__ float*)offsetLocal.GetPhyAddr();

    LocalTensor<xDtype> inLocal = inQueue_.DeQue<xDtype>();
    __ubuf__ xDtype* inAddr = (__ubuf__ xDtype*)inLocal.GetPhyAddr();
    if (mBlockIndex == 0) {
        // 当读取的是第一个块，则将局部max值初始化为float范围下的最小值
        Duplicate<float>(colMaxLocal, NEG_INFINITY, blockColNum);
        // 与max同理，但要初始化为最大值
        if constexpr (isSymmetrical == false)
            Duplicate<float>(colMinLocal, POS_INFINITY, blockColNum);
    }
    LocalTensor<xDtype> smoothLocal;
    __ubuf__ xDtype* smoothAddr{nullptr};
    if constexpr (hasSmooth) {
        smoothLocal = smoothQueue_.DeQue<xDtype>();
        smoothAddr = (__ubuf__ xDtype*)smoothLocal.GetPhyAddr();
    }

    // VF
    if constexpr (isSymmetrical) {
        if (mBlockIndex + 1 == mBlockNum_) {
            ComputeMaxVFforSymmetric<true>(inAddr, smoothAddr, scaleAddr, colMaxLocalAddr, blockRowNum, blockColNum);
        } else {
            ComputeMaxVFforSymmetric<false>(inAddr, smoothAddr, scaleAddr, colMaxLocalAddr, blockRowNum, blockColNum);
        }
    } else {
        ComputeVFMinMaxforNoSymmetric(
            inAddr, smoothAddr, scaleAddr, colMaxLocalAddr, colMinLocalAddr, offsetAddr, blockRowNum, blockColNum);
    }

    if constexpr (hasSmooth) {
        smoothQueue_.FreeTensor(smoothLocal);
    }
    inQueue_.FreeTensor(inLocal);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeOut(
    LocalTensor<float>& colMaxLocal, LocalTensor<float>& colMinLocal, LocalTensor<float>& scaleLocal,
    LocalTensor<float>& offsetLocal, uint32_t mBlockIndex, uint32_t blockRowNum, uint32_t blockColNumForX,
    uint32_t blockColNumForY)
{
    __ubuf__ float* colMaxLocalAddr = (__ubuf__ float*)colMaxLocal.GetPhyAddr();
    __ubuf__ float* colMinLocalAddr = (__ubuf__ float*)colMinLocal.GetPhyAddr();
    __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __ubuf__ float* offsetAddr = (__ubuf__ float*)offsetLocal.GetPhyAddr();

    LocalTensor<xDtype> inLocal = inQueue_.DeQue<xDtype>();
    __ubuf__ xDtype* inAddr = (__ubuf__ xDtype*)inLocal.GetPhyAddr();

    LocalTensor<xDtype> smoothLocal;
    __ubuf__ xDtype* smoothAddr{nullptr};
    if constexpr (hasSmooth) {
        smoothLocal = smoothQueue_.DeQue<xDtype>();
        smoothAddr = (__ubuf__ xDtype*)smoothLocal.GetPhyAddr();
    }

    LocalTensor<yCopyDtype> outLocal = outQueue_.AllocTensor<yCopyDtype>();
    __ubuf__ yCopyDtype* outAddr = (__ubuf__ yCopyDtype*)outLocal.GetPhyAddr();

    // VF
    if constexpr (isSymmetrical) {
        ComputeVFforSymmetric(
            inAddr, smoothAddr, scaleAddr, colMaxLocalAddr, outAddr, blockRowNum, blockColNumForX, blockColNumForY);
    } else {
        ComputeVFforNoSymmetric(
            inAddr, smoothAddr, scaleAddr, colMaxLocalAddr, colMinLocalAddr, outAddr, offsetAddr, blockRowNum,
            blockColNumForX, blockColNumForY);
    }

    outQueue_.EnQue(outLocal);
    if constexpr (hasSmooth) {
        smoothQueue_.FreeTensor(smoothLocal);
    }
    inQueue_.FreeTensor(inLocal);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeVFforSymmetric(
    __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
    __local_mem__ float* colMaxLocalAddr, __local_mem__ yCopyDtype* yAddr, uint32_t blockRowNum,
    uint32_t blockColNumForX, uint32_t blockColNumForY)
{
    constexpr uint16_t elementNumPerLoop = FP32_VF_ALIGNED_NUM; // 从UB一次读入VF的元素个数
    uint16_t nLoopNum = ops::CeilDiv(blockColNumForX, static_cast<uint32_t>(elementNumPerLoop));
    uint16_t mLoopNum = static_cast<uint16_t>(blockRowNum);
    float scalar = maxValue_;
    __VEC_SCOPE__
    {
        // for VF_2
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregInScale;

        MicroAPI::RegTensor<int16_t> vregCastI16;
        MicroAPI::RegTensor<half> vregCastF16;
        MicroAPI::RegTensor<yCopyDtype> vregOut;
        MicroAPI::RegTensor<float> vregOutFp32;
        MicroAPI::MaskReg pregH = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();

        MicroAPI::MaskReg mask;
        uint32_t currColNum = blockColNumForX;
        for (uint16_t nIdx = 0; nIdx < nLoopNum; nIdx++) {
            mask = MicroAPI::UpdateMask<float>(currColNum);
            // 从ub中读入colMax
            // 插入ub同步，直接读取ub中存储的max和scale
            MicroAPI::DataCopy<float>(vregInScale, (__ubuf__ float*)(scaleAddr + nIdx * elementNumPerLoop));
            for (uint16_t mIdx = 0; mIdx < mLoopNum; mIdx++) {
                auto addr = yAddr + mIdx * blockColNumForY + nIdx * elementNumPerLoop;
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregIn, (__ubuf__ xDtype*)(xAddr + mIdx * blockColNumForX + nIdx * elementNumPerLoop));
                MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, mask);
                if constexpr (hasSmooth) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                        vregSmooth, (__ubuf__ xDtype*)(smoothAddr + mIdx));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, mask);
                    MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, mask);
                }
                MicroAPI::Div<float>(vregOutFp32, vregInFp32, vregInScale, mask);
                // cast
                CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, mask);
                if constexpr (IsSameType<yDtype, int4b_t>::value) {
                    addr = yAddr + (mIdx * blockColNumForY + nIdx * elementNumPerLoop) / 2;
                    MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, pregH);
                } else {
                    MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, mask);
                }
            }
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
template <bool isLastBlock>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeMaxVFforSymmetric(
    __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
    __local_mem__ float* colMaxLocalAddr, uint32_t blockRowNum, uint32_t blockColNum)
{
    constexpr uint16_t elementNumPerLoop = FP32_VF_ALIGNED_NUM; // 从UB一次读入VF的元素个数
    uint16_t nLoopNum = ops::CeilDiv(blockColNum, static_cast<uint32_t>(elementNumPerLoop));
    uint16_t mLoopNum = static_cast<uint16_t>(blockRowNum);
    float scalar = maxValue_;
    __VEC_SCOPE__
    {
        // for VF_1
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregAbs;
        MicroAPI::RegTensor<float> vregColMax;

        MicroAPI::MaskReg mask;
        uint32_t currColNum = blockColNum;
        for (uint16_t nIdx = 0; nIdx < nLoopNum; nIdx++) {
            mask = MicroAPI::UpdateMask<float>(currColNum);
            // 从ub中读入colMax
            // 插入ub同步
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            MicroAPI::DataCopy<float>(vregColMax, (__ubuf__ float*)(colMaxLocalAddr + nIdx * elementNumPerLoop));
            for (uint16_t mIdx = 0; mIdx < mLoopNum; mIdx++) {
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregIn, (__ubuf__ xDtype*)(xAddr + mIdx * blockColNum + nIdx * elementNumPerLoop));
                MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, mask);
                if constexpr (hasSmooth) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                        vregSmooth, (__ubuf__ xDtype*)(smoothAddr + mIdx));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, mask);
                    MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, mask);
                }
                MicroAPI::Abs<float>(vregAbs, vregInFp32, mask);
                MicroAPI::Max<float>(vregColMax, vregAbs, vregColMax, mask);
            }
            // 局部colMax写入ub，为后续计算做准备
            MicroAPI::DataCopy<float>((__ubuf__ float*)colMaxLocalAddr + nIdx * elementNumPerLoop, vregColMax, mask);
            if constexpr (isLastBlock) {
                MicroAPI::Muls(vregColMax, vregColMax, scalar, mask);
                MicroAPI::DataCopy<float>((__ubuf__ float*)scaleAddr + nIdx * elementNumPerLoop, vregColMax, mask);
            }
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeVFMinMaxforNoSymmetric(
    __ubuf__ xDtype* inAddr, __ubuf__ xDtype* smoothAddr, __ubuf__ float* scaleAddr, __ubuf__ float* colMaxLocalAddr,
    __ubuf__ float* colMinLocalAddr, __ubuf__ float* offsetAddr, uint32_t curBaseM, uint32_t curBlockSize)
{
    uint16_t elementNumPerLoop = FP32_VF_ALIGNED_NUM;
    uint32_t curBaseNAligned = curBlockSize;
    uint16_t nLoopNum = ops::CeilDiv(curBaseNAligned, static_cast<uint32_t>(elementNumPerLoop));
    uint16_t mLoopNum = static_cast<uint16_t>(curBaseM);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<float> vregColMin;

        MicroAPI::MaskReg preg0;

        uint32_t count = curBaseNAligned;
        for (uint16_t nIdxvf = 0; nIdxvf < nLoopNum; nIdxvf++) {
            preg0 = MicroAPI::UpdateMask<float>(count);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            MicroAPI::DataCopy<float>(vregColMax, (__ubuf__ float*)(colMaxLocalAddr + nIdxvf * elementNumPerLoop));
            MicroAPI::DataCopy<float>(vregColMin, (__ubuf__ float*)(colMinLocalAddr + nIdxvf * elementNumPerLoop));
            for (uint16_t mIdx = 0; mIdx < mLoopNum; mIdx++) {
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregIn, (__ubuf__ xDtype*)(inAddr + mIdx * curBaseNAligned + nIdxvf * elementNumPerLoop));
                MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                if constexpr (hasSmooth) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                        vregSmooth, (__ubuf__ xDtype*)(smoothAddr + mIdx));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                    MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                }
                MicroAPI::Max<float>(vregColMax, vregInFp32, vregColMax, preg0); // max(x)
                MicroAPI::Min<float>(vregColMin, vregInFp32, vregColMin, preg0); // min(x)
            }
            MicroAPI::DataCopy<float>((__ubuf__ float*)colMaxLocalAddr + nIdxvf * elementNumPerLoop, vregColMax, preg0);
            MicroAPI::DataCopy<float>((__ubuf__ float*)colMinLocalAddr + nIdxvf * elementNumPerLoop, vregColMin, preg0);
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeVFforNoSymmetric(
    __local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ float* scaleAddr,
    __local_mem__ float* colMaxLocalAddr, __local_mem__ float* colMinLocalAddr, __local_mem__ yCopyDtype* yAddr,
    __local_mem__ float* offsetAddr, uint32_t blockRowNum, uint32_t blockColNumForX, uint32_t blockColNumForY)
{
    // 一个vf里面最多只能容纳256个字节
    uint16_t elementNumPerLoop = FP32_VF_ALIGNED_NUM;
    // 要区分下是头块还是尾块
    uint32_t curBaseNAligned = blockColNumForX;
    uint32_t curBaseNAlignedB8 = blockColNumForY;
    // 相除的两个类型要相同
    uint16_t nLoopNum = ops::CeilDiv(
        curBaseNAligned,
        static_cast<uint32_t>(elementNumPerLoop)); // 一个vf里面的列需要多少次循环（按个数算）判断是否需要
    uint16_t mLoopNum = static_cast<uint16_t>(blockRowNum);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<float> vregColMin;

        MicroAPI::RegTensor<float> vregSub;
        MicroAPI::RegTensor<float> vregScale;
        MicroAPI::RegTensor<float> vregDiv;
        MicroAPI::RegTensor<float> vregDiv_1;
        MicroAPI::RegTensor<float> vregOffset;

        MicroAPI::RegTensor<int16_t> vregCastI16;
        MicroAPI::RegTensor<half> vregCastF16;
        MicroAPI::RegTensor<yCopyDtype> vregOut;
        MicroAPI::RegTensor<float> vregOutFp32;
        MicroAPI::RegTensor<float> vregMaxFactor;
        MicroAPI::RegTensor<float> vregOffsetDivVal;

        MicroAPI::MaskReg preg0;
        MicroAPI::MaskReg pregH = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<float>(vregMaxFactor, offsetValue_, pregAll);
        MicroAPI::Duplicate<float>(vregOffsetDivVal, offsetDivValue_, pregAll);

        uint32_t count = curBaseNAligned;

        for (uint16_t nIdxvf = 0; nIdxvf < nLoopNum; nIdxvf++) {
            preg0 = MicroAPI::UpdateMask<float>(count);
            // 将min max scale offset拷贝进来
            MicroAPI::DataCopy<float>(vregColMax, (__ubuf__ float*)(colMaxLocalAddr + nIdxvf * elementNumPerLoop));
            MicroAPI::DataCopy<float>(vregColMin, (__ubuf__ float*)(colMinLocalAddr + nIdxvf * elementNumPerLoop));

            // scaleout
            MicroAPI::Sub<float>(vregSub, vregColMax, vregColMin, preg0);      // max(x)-min(x)
            MicroAPI::Mul<float>(vregScale, vregSub, vregOffsetDivVal, preg0); // (max(x)-min(x))/offsetMaxValue

            // offset
            MicroAPI::Div<float, &divHighPrecisionMode >(vregDiv, vregColMax, vregScale, preg0);     // max(x)/scaleout
            MicroAPI::Sub<float>(vregOffset, vregMaxFactor, vregDiv, preg0); // max_value - max(x)/scaleout

            // y
            for (uint16_t mIdx = 0; mIdx < mLoopNum; mIdx++) {
                auto addr = yAddr + mIdx * curBaseNAlignedB8 + nIdxvf * elementNumPerLoop;
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregIn, (__ubuf__ xDtype*)(xAddr + mIdx * curBaseNAligned + nIdxvf * elementNumPerLoop));
                MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                if constexpr (hasSmooth) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                        vregSmooth, (__ubuf__ xDtype*)(smoothAddr + mIdx));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                    MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                }
                MicroAPI::Div<float>(vregDiv_1, vregInFp32, vregScale, preg0);   // input/scaleOut
                MicroAPI::Add<float>(vregOutFp32, vregDiv_1, vregOffset, preg0); // input/scaleOut + offset
                CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                if constexpr (IsSameType<yDtype, int4b_t>::value) {
                    addr = yAddr + (mIdx * curBaseNAlignedB8 + nIdxvf * elementNumPerLoop) / 2;
                    MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, pregH);
                } else {
                    MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, preg0);
                }
            }
            MicroAPI::DataCopy<float>((__ubuf__ float*)scaleAddr + nIdxvf * elementNumPerLoop, vregScale, preg0);
            MicroAPI::DataCopy<float>((__ubuf__ float*)offsetAddr + nIdxvf * elementNumPerLoop, vregOffset, preg0);
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyOutScaleAndOffset(
    uint32_t blockColNum, uint64_t scaleOffset)
{
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(blockColNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleGm_[scaleOffset], scaleLocal, scaleCopyParams);

    LocalTensor<float> offsetLocal;
    if constexpr (isSymmetrical == false) {
        offsetLocal = offsetQueue_.DeQue<float>();
        DataCopyPad(offsetGm_[scaleOffset], offsetLocal, scaleCopyParams);
        offsetQueue_.FreeTensor(offsetLocal);
    }
    scaleQueue_.FreeTensor(scaleLocal);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelRecompute<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyOutY(
    uint32_t blockRowNum, uint32_t blockColNum, uint64_t xOffset)
{
    LocalTensor<yCopyDtype> yLocal = outQueue_.DeQue<yCopyDtype>();
    uint32_t dstStrideNum = nLen_ - blockColNum;
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(blockRowNum), static_cast<uint32_t>(blockColNum * sizeof(yCopyDtype)), 0,
        static_cast<int64_t>(dstStrideNum * sizeof(yCopyDtype)), 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyParams.blockLen = copyParams.blockLen >> 1;
        copyParams.dstStride = copyParams.dstStride >> 1;
        DataCopyPad(outGm_[xOffset / 2], yLocal, copyParams);
    } else {
        DataCopyPad(outGm_[xOffset], yLocal, copyParams);
    }
    outQueue_.FreeTensor(yLocal);
}

} // namespace DynamicQuantPerChannel
#endif
