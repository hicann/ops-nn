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
 * \file max_pool3d_with_argmax_v2_nc_transpose.h
 * \brief NC transpose kernel for arch35: optimized for large NC + small kernel scenarios.
 */

#ifndef MAX_POOL_3D_WITH_ARGMAX_V2_NC_TRANSPOSE_H_
#define MAX_POOL_3D_WITH_ARGMAX_V2_NC_TRANSPOSE_H_

#include "max_pool3d_with_argmax_v2_tiling_struct.h"
#include <type_traits>

namespace MaxPool3DWithArgmaxV2NcTransposeNameSpace {
using namespace AscendC;

constexpr uint32_t NC_TRANS_BUFFER_NUM = 1;
constexpr uint64_t NC_TRANS_ALIGN_16 = 16;
constexpr uint64_t NC_TRANS_BLOCK_SIZE = 32;
constexpr uint64_t NC_TRANS_NUM_TWO = 2;
constexpr uint64_t NC_TRANS_TRANS_LEN_B32 = 8;

// 统一索引类型选择器
template <typename T1>
struct SelectIndexType {
    using type = std::conditional_t<std::is_same_v<T1, half> || std::is_same_v<T1, bfloat16_t>, int16_t, int32_t>;
};

template <typename T1, typename T2, const uint32_t IS_PAD = 0, typename IDX_T = typename SelectIndexType<T1>::type>
class MaxPool3DWithArgmaxV2NcTransposeKernel {
public:
    __aicore__ inline MaxPool3DWithArgmaxV2NcTransposeKernel(
        TPipe& pipeIn, const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessEachLoop(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void TransposeInput();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t loopNum);

    template <typename U>
    __aicore__ inline void TransposeB32(LocalTensor<U> dst, LocalTensor<U> src, uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void TransposeB16(LocalTensor<T1> dst, LocalTensor<T1> src, uint32_t rowNum, uint32_t colNum);

    __aicore__ inline void ComputeMaxArgmax(__local_mem__ T1* transInputAddr, __local_mem__ T1* maxValueAddr,
                                            __local_mem__ T2* argmaxAddr, LocalTensor<T1> transInputLocal,
                                            LocalTensor<T1> maxValueLocal, LocalTensor<T2> argmaxLocal);

    __aicore__ inline void ConvertDeltaToArgmax(LocalTensor<IDX_T> deltaAll, LocalTensor<int32_t> int32Scratch,
                                                LocalTensor<T2> argmaxLocal, int64_t ncAligned, int64_t totalOut,
                                                int64_t ncLoop, int64_t ncTail, int64_t ncStep, int64_t globalDBase,
                                                int64_t globalHBase, int64_t globalWBase, int64_t hInputWInput,
                                                int64_t wInput, int64_t dStride, int64_t hStride, int64_t wStride);

    TPipe& pipe_;
    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData& tilingData_;

    TQue<QuePosition::VECIN, NC_TRANS_BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, NC_TRANS_BUFFER_NUM> maxValueQue_;
    TQue<QuePosition::VECOUT, NC_TRANS_BUFFER_NUM> argmaxQue_;
    TBuf<TPosition::VECCALC> transInputBuf_;
    TBuf<TPosition::VECCALC> deltaIndexBuf_;
    TQue<QuePosition::VECOUT, NC_TRANS_BUFFER_NUM> castQue_; // int32→int64 Cast 输出队列

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;
    int64_t ncActual_ = 0;
    int64_t dOutputReal_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t wOutputReal_ = 0;
    int64_t curCoreProcessNum_ = 0;
    int64_t inputPlaneSize_ = 0;

    int64_t dInputActual_ = 0;
    int64_t hInputActual_ = 0;
    int64_t wInputActual_ = 0;

    int64_t ncIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t ncInputOffset_ = 0;
    int64_t dInputOffset_ = 0;
    int64_t hInputOffset_ = 0;
    int64_t wInputOffset_ = 0;

    constexpr static int32_t blockSize_ = platform::GetUbBlockSize();
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(T1);
    constexpr static uint16_t vlT2_ = platform::GetVRegSize() / sizeof(T2);
    constexpr static uint64_t blockAlignT1_ = NC_TRANS_BLOCK_SIZE / sizeof(T1);
    constexpr static uint16_t vlIDX_ = platform::GetVRegSize() / sizeof(IDX_T);
};

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::Init(GM_ADDR x, GM_ADDR y,
                                                                                           GM_ADDR argmax)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    xGm_.SetGlobalBuffer((__gm__ T1*)x);

    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_.usedCoreNum) ? tilingData_.tailCoreProcessNum :
                                                                      tilingData_.normalCoreProcessNum;
    inputPlaneSize_ = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput;

    pipe_.InitBuffer(inputQue_, NC_TRANS_BUFFER_NUM, tilingData_.inputBufferSize);
    pipe_.InitBuffer(transInputBuf_, tilingData_.transBufferSize);
    pipe_.InitBuffer(maxValueQue_, NC_TRANS_BUFFER_NUM, tilingData_.maxValueBufferSize);
    pipe_.InitBuffer(argmaxQue_, NC_TRANS_BUFFER_NUM, tilingData_.argmaxBufferSize);
    pipe_.InitBuffer(deltaIndexBuf_, tilingData_.deltaIndexBufferSize);
    if constexpr (std::is_same_v<T2, int64_t>) {
        pipe_.InitBuffer(castQue_, NC_TRANS_BUFFER_NUM, tilingData_.castBufferSize);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessEachLoop(loopNum);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::ScalarCompute(int64_t loopNum)
{
    int64_t d_h_w_outer = tilingData_.dOutputOuter * tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t h_w_outer = tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t w_outer = tilingData_.wOutputOuter;

    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;

    ncIndex_ = baseBlockIdx / d_h_w_outer;
    ncActual_ = ncIndex_ == (tilingData_.ncOuter - 1) ? tilingData_.ncTail : tilingData_.ncInner;

    int64_t mod_dhw = baseBlockIdx - d_h_w_outer * ncIndex_;
    dAxisIndex_ = mod_dhw / h_w_outer;
    dOutputReal_ = dAxisIndex_ == (tilingData_.dOutputOuter - 1) ? tilingData_.dOutputTail : tilingData_.dOutputInner;

    int64_t mod_hw = mod_dhw - h_w_outer * dAxisIndex_;
    hAxisIndex_ = mod_hw / w_outer;
    hOutputReal_ = hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner;

    wAxisIndex_ = mod_hw - w_outer * hAxisIndex_;
    wOutputReal_ = wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner;

    dInputActual_ = (dOutputReal_ - 1) * tilingData_.dStride + tilingData_.dKernel;
    hInputActual_ = (hOutputReal_ - 1) * tilingData_.hStride + tilingData_.hKernel;
    wInputActual_ = (wOutputReal_ - 1) * tilingData_.wStride + tilingData_.wKernel;

    ncInputOffset_ = ncIndex_ * tilingData_.ncInner * inputPlaneSize_;
    dInputOffset_ = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride * tilingData_.hInput *
                    tilingData_.wInput;
    hInputOffset_ = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride * tilingData_.wInput;
    wInputOffset_ = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::ProcessEachLoop(int64_t loopNum)
{
    CopyIn();
    TransposeInput();
    Compute();
    CopyOut(loopNum);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::CopyIn()
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    int64_t xGmOffset = ncInputOffset_ + dInputOffset_ + hInputOffset_ + wInputOffset_;
    int64_t dActual = dInputActual_;
    int64_t hActual = hInputActual_;
    int64_t wActual = wInputActual_;
    int64_t wActualAligned = CeilDivision(wActual, blockSize_ / sizeof(T1)) * (blockSize_ / sizeof(T1));

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncActual_;
    loopModeParams.loop2Size = dActual;
    loopModeParams.loop1SrcStride = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput * sizeof(T1);
    loopModeParams.loop2SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
    loopModeParams.loop1DstStride = dActual * hActual * wActualAligned * sizeof(T1);
    loopModeParams.loop2DstStride = hActual * wActualAligned * sizeof(T1);

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    uint8_t rightPad = static_cast<uint8_t>(wActualAligned - wActual);
    DataCopyPadExtParams<T1> padParams = {true, 0, rightPad, static_cast<T1>(0)};
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(hActual);
    copyParams.blockLen = static_cast<uint32_t>(wActual * sizeof(T1));
    copyParams.srcStride = static_cast<uint32_t>((tilingData_.wInput - wActual) * sizeof(T1));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(xLocal, xGm_[xGmOffset], copyParams, padParams);
    inputQue_.EnQue(xLocal);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
template <typename U>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::TransposeB32(LocalTensor<U> dst,
                                                                                                   LocalTensor<U> src,
                                                                                                   uint32_t rowNum,
                                                                                                   uint32_t colNum)
{
    uint64_t dstList[NC_TRANS_ALIGN_16];
    uint64_t srcList[NC_TRANS_ALIGN_16];
    TransDataTo5HDParams transParams;
    transParams.dstHighHalf = false;
    transParams.srcHighHalf = false;
    uint32_t transposeAlign = NC_TRANS_BLOCK_SIZE / sizeof(U);

    if (colNum == transposeAlign) {
        transParams.repeatTimes = rowNum / NC_TRANS_ALIGN_16;
        transParams.dstRepStride = NC_TRANS_ALIGN_16 * sizeof(U) / NC_TRANS_BLOCK_SIZE;
        transParams.srcRepStride = NC_TRANS_ALIGN_16;
        for (int32_t i = 0; i < NC_TRANS_ALIGN_16; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transposeAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < NC_TRANS_TRANS_LEN_B32; i++) {
            dstList[i * NC_TRANS_NUM_TWO] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
            dstList[i * NC_TRANS_NUM_TWO + 1] = static_cast<uint64_t>(dst[i * rowNum + transposeAlign].GetPhyAddr());
        }
        if (transParams.repeatTimes == 1) {
            transParams.srcRepStride = 0;
            transParams.dstRepStride = 0;
        }
        TransDataTo5HD<U>(dstList, srcList, transParams);
    } else {
        transParams.repeatTimes = colNum / transposeAlign;
        transParams.dstRepStride = rowNum;
        transParams.srcRepStride = 1;
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / NC_TRANS_ALIGN_16; rowLoopIdx++) {
            for (int32_t i = 0; i < NC_TRANS_ALIGN_16; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * NC_TRANS_ALIGN_16 * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < NC_TRANS_TRANS_LEN_B32; i++) {
                dstList[i * NC_TRANS_NUM_TWO] = static_cast<uint64_t>(
                    dst[rowLoopIdx * NC_TRANS_ALIGN_16 + i * rowNum].GetPhyAddr());
                dstList[i * NC_TRANS_NUM_TWO + 1] = static_cast<uint64_t>(
                    dst[rowLoopIdx * NC_TRANS_ALIGN_16 + i * rowNum + transposeAlign].GetPhyAddr());
            }
            TransDataTo5HD<U>(dstList, srcList, transParams);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::TransposeB16(LocalTensor<T1> dst,
                                                                                                   LocalTensor<T1> src,
                                                                                                   uint32_t rowNum,
                                                                                                   uint32_t colNum)
{
    uint64_t dstList[NC_TRANS_ALIGN_16];
    uint64_t srcList[NC_TRANS_ALIGN_16];
    TransDataTo5HDParams transParams;
    transParams.dstHighHalf = false;
    transParams.srcHighHalf = false;
    uint32_t transposeAlign = blockAlignT1_;

    if (colNum == transposeAlign) {
        transParams.repeatTimes = rowNum / NC_TRANS_ALIGN_16;
        transParams.dstRepStride = NC_TRANS_ALIGN_16 * sizeof(T1) / NC_TRANS_BLOCK_SIZE;
        transParams.srcRepStride = NC_TRANS_ALIGN_16;
        for (int32_t i = 0; i < NC_TRANS_ALIGN_16; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transposeAlign].GetPhyAddr());
            dstList[i] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
        }
        if (transParams.repeatTimes == 1) {
            transParams.srcRepStride = 0;
            transParams.dstRepStride = 0;
        }
        TransDataTo5HD<T1>(dstList, srcList, transParams);
    } else {
        transParams.repeatTimes = colNum / transposeAlign;
        transParams.dstRepStride = rowNum;
        transParams.srcRepStride = 1;
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / NC_TRANS_ALIGN_16; rowLoopIdx++) {
            for (int32_t i = 0; i < NC_TRANS_ALIGN_16; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * NC_TRANS_ALIGN_16 * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(dst[rowLoopIdx * NC_TRANS_ALIGN_16 + i * rowNum].GetPhyAddr());
            }
            TransDataTo5HD<T1>(dstList, srcList, transParams);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::TransposeInput()
{
    LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();
    LocalTensor<T1> transLocal = transInputBuf_.Get<T1>();

    int64_t ncAligned = CeilDivision(ncActual_, (int64_t)NC_TRANS_ALIGN_16) * NC_TRANS_ALIGN_16;
    int64_t wAligned;
    int64_t dActual, hActual;
    dActual = dInputActual_;
    hActual = hInputActual_;
    wAligned = CeilDivision(wInputActual_, (int64_t)blockAlignT1_) * blockAlignT1_;
    int64_t spatialSize = dActual * hActual * wAligned;

    if constexpr (std::is_same<T1, float>::value) {
        TransposeB32<T1>(transLocal, xLocal, ncAligned, spatialSize);
    } else {
        TransposeB16(transLocal, xLocal, ncAligned, spatialSize);
    }

    inputQue_.FreeTensor(xLocal);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::ComputeMaxArgmax(
    __local_mem__ T1* transInputAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr,
    LocalTensor<T1> transInputLocal, LocalTensor<T1> maxValueLocal, LocalTensor<T2> argmaxLocal)
{
    int64_t ncAligned = CeilDivision(ncActual_, (int64_t)NC_TRANS_ALIGN_16) * NC_TRANS_ALIGN_16;
    int64_t wAligned = CeilDivision(wInputActual_, (int64_t)blockAlignT1_) * blockAlignT1_;
    int64_t hActual = hInputActual_;
    int64_t dActual = dInputActual_;

    int64_t dStride = tilingData_.dStride;
    int64_t hStride = tilingData_.hStride;
    int64_t wStride = tilingData_.wStride;
    int64_t dKernel = tilingData_.dKernel;
    int64_t hKernel = tilingData_.hKernel;
    int64_t wKernel = tilingData_.wKernel;

    constexpr uint16_t ncStep = vlT1_;
    int64_t ncLoop = CeilDivision(ncActual_, (int64_t)ncStep);
    int64_t ncTail = (ncActual_ % ncStep) == 0 ? ncStep : (ncActual_ % ncStep);

    int64_t globalDBase = dAxisIndex_ * tilingData_.dOutputInner * dStride;
    int64_t globalHBase = hAxisIndex_ * tilingData_.hOutputInner * hStride;
    int64_t globalWBase = wAxisIndex_ * tilingData_.wOutputInner * wStride;

    int64_t wInput = tilingData_.wInput;
    int64_t hInputWInput = tilingData_.hInput * tilingData_.wInput;

    LocalTensor<IDX_T> deltaUb = deltaIndexBuf_.template Get<IDX_T>();
    int32_t maxDeltaCount = 0;
    for (int64_t kd = 0; kd < dKernel; kd++) {
        for (int64_t kh = 0; kh < hKernel; kh++) {
            for (int64_t kw = 0; kw < wKernel; kw++) {
                IDX_T delta = static_cast<IDX_T>(kd * hInputWInput + kh * wInput + kw);
                Duplicate(deltaUb[maxDeltaCount * vlIDX_], delta, vlIDX_);
                maxDeltaCount++;
            }
        }
    }
    PipeBarrier<PIPE_V>();
    __local_mem__ IDX_T* deltaUbAddr = (__local_mem__ IDX_T*)deltaUb.GetPhyAddr();

    int64_t dSpatialStride = hActual * wAligned;
    int64_t hSpatialStride = wAligned;
    int64_t wSpatialStride = 1;
    uint16_t correctDK = static_cast<uint16_t>(dKernel);
    uint16_t correctHK = static_cast<uint16_t>(hKernel);
    uint16_t correctWK = static_cast<uint16_t>(wKernel);

    for (int64_t ncChunk = 0; ncChunk < ncLoop; ncChunk++) {
        uint32_t ncOffset = ncChunk * ncStep;
        uint32_t computeVL = (ncChunk == ncLoop - 1) ? ncTail : ncStep;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T1> vreg0;
            MicroAPI::RegTensor<T1> vreg1;
            MicroAPI::RegTensor<IDX_T> argDeltaVreg;
            MicroAPI::RegTensor<IDX_T> argUpdateVreg;
            MicroAPI::MaskReg gtMask, neMask;
            MicroAPI::MaskReg computeMaskT1 = MicroAPI::UpdateMask<T1>(computeVL);

            for (uint16_t dOut = 0; dOut < static_cast<uint16_t>(dOutputReal_); dOut++) {
                for (uint16_t hOut = 0; hOut < static_cast<uint16_t>(hOutputReal_); hOut++) {
                    for (uint16_t wOut = 0; wOut < static_cast<uint16_t>(wOutputReal_); wOut++) {
                        int64_t localDStart = dOut * dStride;
                        int64_t localHStart = hOut * hStride;
                        int64_t localWStart = wOut * wStride;
                        int64_t baseSpatialIdx = localDStart * dSpatialStride + localHStart * hSpatialStride +
                                                 localWStart * wSpatialStride;

                        MicroAPI::DataCopy(vreg0, transInputAddr + baseSpatialIdx * ncAligned + ncOffset);
                        MicroAPI::Duplicate(argDeltaVreg, static_cast<IDX_T>(0));

                        int32_t dIdx = 0;
                        for (uint16_t kd = 0; kd < correctDK; kd++) {
                            for (uint16_t kh = 0; kh < correctHK; kh++) {
                                for (uint16_t kw = 0; kw < correctWK; kw++) {
                                    int64_t spatialIdx = baseSpatialIdx + kd * dSpatialStride + kh * hSpatialStride +
                                                         kw * wSpatialStride;
                                    MicroAPI::DataCopy(vreg1, transInputAddr + spatialIdx * ncAligned + ncOffset);

                                    MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, vreg1, vreg0, computeMaskT1);
                                    MicroAPI::Compare<T1, CMPMODE::NE>(neMask, vreg1, vreg1, computeMaskT1);
                                    MicroAPI::MaskOr(gtMask, gtMask, neMask, computeMaskT1);

                                    MicroAPI::DataCopy(argUpdateVreg, deltaUbAddr + dIdx * vlIDX_);
                                    MicroAPI::Select(argDeltaVreg, argUpdateVreg, argDeltaVreg, gtMask);
                                    MicroAPI::Max(vreg0, vreg0, vreg1, computeMaskT1);
                                    dIdx++;
                                }
                            }
                        }

                        int64_t outputOffset = (dOut * hOutputReal_ * wOutputReal_ + hOut * wOutputReal_ + wOut) *
                                                   ncAligned +
                                               ncOffset;
                        MicroAPI::DataCopy(maxValueAddr + outputOffset, vreg0, computeMaskT1);
                        MicroAPI::DataCopy((__local_mem__ IDX_T*)(argmaxAddr) + outputOffset, argDeltaVreg,
                                           computeMaskT1);
                    }
                }
            }
        }
    }

    PipeBarrier<PIPE_V>();
    {
        LocalTensor<IDX_T> deltaAll = argmaxLocal.template ReinterpretCast<IDX_T>();
        LocalTensor<int32_t> int32Scratch = transInputLocal.template ReinterpretCast<int32_t>();
        int64_t totalOut = dOutputReal_ * hOutputReal_ * wOutputReal_;
        ConvertDeltaToArgmax(deltaAll, int32Scratch, argmaxLocal, ncAligned, totalOut, ncLoop, ncTail, ncStep,
                             globalDBase, globalHBase, globalWBase, hInputWInput, wInput, dStride, hStride, wStride);
    }
}

// ==================== 索引转换 ====================
template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::ConvertDeltaToArgmax(
    LocalTensor<IDX_T> deltaAll, LocalTensor<int32_t> int32Scratch, LocalTensor<T2> argmaxLocal, int64_t ncAligned,
    int64_t totalOut, int64_t ncLoop, int64_t ncTail, int64_t ncStep, int64_t globalDBase, int64_t globalHBase,
    int64_t globalWBase, int64_t hInputWInput, int64_t wInput, int64_t dStride, int64_t hStride, int64_t wStride)
{
    for (int64_t ncChunkR = ncLoop - 1; ncChunkR >= 0; ncChunkR--) {
        uint32_t curVL = (ncChunkR == ncLoop - 1) ? ncTail : ncStep;
        uint32_t curNcOffset = ncChunkR * ncStep;

        if (static_cast<int64_t>(curVL) == ncAligned) {
            int64_t totalElements = totalOut * static_cast<int64_t>(curVL);
            int64_t outIdx = 0;

            // 当IDX_T已经是int32_t时（float输入），直接使用原地操作，跳过Cast
            if constexpr (std::is_same_v<IDX_T, int32_t>) {
                LocalTensor<int32_t> deltaAllI32 = deltaAll.template ReinterpretCast<int32_t>();
                for (uint16_t dOut = 0; dOut < static_cast<uint16_t>(dOutputReal_); dOut++) {
                    for (uint16_t hOut = 0; hOut < static_cast<uint16_t>(hOutputReal_); hOut++) {
                        for (uint16_t wOut = 0; wOut < static_cast<uint16_t>(wOutputReal_); wOut++) {
                            int64_t gh = globalHBase + hOut * hStride;
                            int64_t gw = globalWBase + wOut * wStride;
                            int64_t gd = globalDBase + dOut * dStride;
                            int32_t startIdx = static_cast<int32_t>(gd * hInputWInput + gh * wInput + gw);

                            LocalTensor<int32_t> argSlice = argmaxLocal.template ReinterpretCast<
                                int32_t>()[outIdx * ncAligned + curNcOffset];
                            Adds(argSlice, deltaAllI32[outIdx * curVL + curNcOffset], startIdx, curVL);
                            outIdx++;
                        }
                    }
                }
            } else {
                // 对于int16_t类型cast到int32再累加
                Cast(int32Scratch, deltaAll[curNcOffset], RoundMode::CAST_NONE, totalElements);

                for (uint16_t dOut = 0; dOut < static_cast<uint16_t>(dOutputReal_); dOut++) {
                    for (uint16_t hOut = 0; hOut < static_cast<uint16_t>(hOutputReal_); hOut++) {
                        for (uint16_t wOut = 0; wOut < static_cast<uint16_t>(wOutputReal_); wOut++) {
                            int64_t gd = globalDBase + dOut * dStride;
                            int64_t gh = globalHBase + hOut * hStride;
                            int64_t gw = globalWBase + wOut * wStride;
                            int32_t startIdx = static_cast<int32_t>(gd * hInputWInput + gh * wInput + gw);

                            LocalTensor<int32_t> argSlice = argmaxLocal.template ReinterpretCast<
                                int32_t>()[outIdx * ncAligned + curNcOffset];
                            Adds(argSlice, int32Scratch[outIdx * curVL], startIdx, curVL);
                            outIdx++;
                        }
                    }
                }
            }
        } else {
            if constexpr (std::is_same_v<IDX_T, int32_t>) {
                // float 输入: IDX_T=int32, in-place 安全，反向遍历
                int64_t revIdx = totalOut - 1;
                for (int64_t dOut = dOutputReal_ - 1; dOut >= 0; dOut--) {
                    for (int64_t hOut = hOutputReal_ - 1; hOut >= 0; hOut--) {
                        for (int64_t wOut = wOutputReal_ - 1; wOut >= 0; wOut--) {
                            int64_t gd = globalDBase + dOut * dStride;
                            int64_t gh = globalHBase + hOut * hStride;
                            int64_t gw = globalWBase + wOut * wStride;
                            int32_t startIdx = static_cast<int32_t>(gd * hInputWInput + gh * wInput + gw);
                            int64_t offset = revIdx * ncAligned + curNcOffset;

                            LocalTensor<int32_t> argSlice = argmaxLocal.template ReinterpretCast<int32_t>()[offset];
                            LocalTensor<int32_t> deltaSliceI32 = deltaAll[offset].template ReinterpretCast<int32_t>();
                            Adds(argSlice, deltaSliceI32, startIdx, curVL);
                            revIdx--;
                        }
                    }
                }
            } else {
                // fp16/bf16 输入, 先批量 Cast 所有 totalOut 个 slice 的 delta 到 int32Scratch，再正向 Adds 写回。
                int64_t totalElements = totalOut * ncAligned;
                Cast(int32Scratch, deltaAll[curNcOffset], RoundMode::CAST_NONE, totalElements);
                PipeBarrier<PIPE_ALL>();

                int64_t outIdx = 0;
                for (int64_t dOut = 0; dOut < dOutputReal_; dOut++) {
                    for (int64_t hOut = 0; hOut < hOutputReal_; hOut++) {
                        for (int64_t wOut = 0; wOut < wOutputReal_; wOut++) {
                            int64_t gd = globalDBase + dOut * dStride;
                            int64_t gh = globalHBase + hOut * hStride;
                            int64_t gw = globalWBase + wOut * wStride;
                            int32_t startIdx = static_cast<int32_t>(gd * hInputWInput + gh * wInput + gw);
                            int64_t offset = outIdx * ncAligned + curNcOffset;

                            LocalTensor<int32_t> argSlice = argmaxLocal.template ReinterpretCast<int32_t>()[offset];
                            Adds(argSlice, int32Scratch[outIdx * ncAligned], startIdx, curVL);
                            outIdx++;
                        }
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::Compute()
{
    LocalTensor<T1> transLocal = transInputBuf_.Get<T1>();
    LocalTensor<T1> maxValueLocal = maxValueQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();

    __local_mem__ T1* transInputAddr = (__local_mem__ T1*)transLocal.GetPhyAddr();
    __local_mem__ T1* maxValueAddr = (__local_mem__ T1*)maxValueLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    ComputeMaxArgmax(transInputAddr, maxValueAddr, argmaxAddr, transLocal, maxValueLocal, argmaxLocal);

    maxValueQue_.EnQue(maxValueLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename IDX_T>
__aicore__ inline void MaxPool3DWithArgmaxV2NcTransposeKernel<T1, T2, IS_PAD, IDX_T>::CopyOut(int64_t loopNum)
{
    LocalTensor<T1> maxValueLocal = maxValueQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    LocalTensor<T1> transLocal = transInputBuf_.Get<T1>();

    int64_t ncAligned = CeilDivision(ncActual_, (int64_t)NC_TRANS_ALIGN_16) * NC_TRANS_ALIGN_16;
    int64_t outSpatialSize = dOutputReal_ * hOutputReal_ * wOutputReal_;
    int64_t outSpatialAligned = CeilDivision(outSpatialSize, (int64_t)NC_TRANS_ALIGN_16) * NC_TRANS_ALIGN_16;
    int64_t outPlaneSize = tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput;

    int64_t ncOutOffset = ncIndex_ * tilingData_.ncInner * outPlaneSize;
    int64_t dOutputOffset = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.hOutput * tilingData_.wOutput;
    int64_t hOutputOffset = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.wOutput;
    int64_t wOutputOffset = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outGmOffset = ncOutOffset + dOutputOffset + hOutputOffset + wOutputOffset;

    if constexpr (std::is_same<T1, float>::value)
        TransposeB32<T1>(transLocal, maxValueLocal, outSpatialAligned, ncAligned);
    else
        TransposeB16(transLocal, maxValueLocal, outSpatialAligned, ncAligned);

    // argmax转置使用B32
    int64_t argTransOffsetT1 = ncAligned * outSpatialAligned;
    LocalTensor<int32_t> transLocalI32ForArg = transLocal[argTransOffsetT1].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> argmaxLocalI32 = argmaxLocal.template ReinterpretCast<int32_t>();
    TransposeB32<int32_t>(transLocalI32ForArg, argmaxLocalI32, outSpatialAligned, ncAligned);

    // 同步：确保 VEC 转置写入 transLocal 完成后，MTE3 才能读取搬出
    event_t eventIdVtoMTE3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);

    DataCopyExtParams maxCopyParams;
    maxCopyParams.blockCount = 1;
    maxCopyParams.blockLen = static_cast<uint32_t>(outSpatialSize * sizeof(T1));
    maxCopyParams.srcStride = 0;
    maxCopyParams.dstStride = 0;
    maxCopyParams.rsv = 0;

    LoopModeParams maxLoopParams;
    maxLoopParams.loop1Size = ncActual_;
    maxLoopParams.loop2Size = 1;
    maxLoopParams.loop1SrcStride = outSpatialAligned * sizeof(T1);
    maxLoopParams.loop1DstStride = outPlaneSize * sizeof(T1);
    SetLoopModePara(maxLoopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGm_[outGmOffset], transLocal, maxCopyParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    if constexpr (std::is_same_v<T2, int64_t>) {
        DataCopyExtParams argCopyParams;
        argCopyParams.blockCount = 1;
        argCopyParams.blockLen = static_cast<uint32_t>(outSpatialSize * sizeof(int64_t));
        argCopyParams.srcStride = 0;
        argCopyParams.dstStride = 0;
        argCopyParams.rsv = 0;

        for (int64_t nc = 0; nc < ncActual_; nc++) {
            LocalTensor<int32_t> srcSlice = transLocalI32ForArg[nc * outSpatialAligned];
            LocalTensor<int64_t> castDstI64 = castQue_.AllocTensor<int64_t>();
            Cast(castDstI64, srcSlice, RoundMode::CAST_NONE, outSpatialAligned);
            castQue_.EnQue(castDstI64);
            castQue_.DeQue<int64_t>();

            int64_t gmOffset = outGmOffset + nc * outPlaneSize;
            DataCopyPad(argmaxGm_[gmOffset], castDstI64, argCopyParams);
            castQue_.FreeTensor(castDstI64);
        }
    } else {
        // int32 路径
        LocalTensor<T2> transLocalT2ForArg = transLocalI32ForArg.template ReinterpretCast<T2>();
        DataCopyExtParams argCopyParams;
        argCopyParams.blockCount = 1;
        argCopyParams.blockLen = static_cast<uint32_t>(outSpatialSize * sizeof(T2));
        argCopyParams.srcStride = 0;
        argCopyParams.dstStride = 0;
        argCopyParams.rsv = 0;

        LoopModeParams argLoopParams;
        argLoopParams.loop1Size = ncActual_;
        argLoopParams.loop2Size = 1;
        argLoopParams.loop1SrcStride = outSpatialAligned * sizeof(T2);
        argLoopParams.loop1DstStride = outPlaneSize * sizeof(T2);
        SetLoopModePara(argLoopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(argmaxGm_[outGmOffset], transLocalT2ForArg, argCopyParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

        event_t eventMte3V = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V);
    }

    maxValueQue_.FreeTensor(maxValueLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

} // namespace MaxPool3DWithArgmaxV2NcTransposeNameSpace
#endif