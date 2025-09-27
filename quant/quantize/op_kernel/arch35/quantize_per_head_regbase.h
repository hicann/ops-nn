/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_perhead_regbase.h
 * \brief quantize perhead kernel regbase
 */

#ifndef QUANTIZE_PERHEAD_REGBASE_H
#define QUANTIZE_PERHEAD_REGBASE_H

#include "quantize.h"

namespace QuantizeOp {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
class QuantizePerHeadRegbase : public QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode> {
public:
    __aicore__ inline QuantizePerHeadRegbase(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, const QuantizeTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ yCopyDtype*>(y));

        this->ParseTilingData(tilingData, tilingData_);
        ParsePerHeadCoreBlocks();

        auto dataLen = tilingData_.baseN * tilingData_.baseLen;
        pipe_.InitBuffer(inQueueX_, bufferNum_, dataLen * sizeof(T));
        pipe_.InitBuffer(outQueueY_, bufferNum_, dataLen * sizeof(U));
        pipe_.InitBuffer(inQueueScale_, bufferNum_, dataLen * sizeof(T1));
        pipe_.InitBuffer(inQueueOffset_, bufferNum_, dataLen * sizeof(T2));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tilingData_.numCore || blockN_ == 0) {
            return;
        }
        CalcOffset();
        ProcessLoop();
    }

private:
    using yCopyDtype = std::conditional_t<IsSameType<U, int4b_t>::value, uint8_t, U>;
    constexpr static int32_t bufferNum_ = 1;
    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<T2> offsetGm_;
    GlobalTensor<yCopyDtype> yGm_;
    QuantizeTilingData tilingData_;
    uint32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t gmSOffset_ = 0;
    int64_t blockS_ = 1;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;

    __aicore__ inline void ParseTilingData(const QuantizeTilingData* tilingData, QuantizeTilingData& runTilingData)
    {
        runTilingData = *tilingData;
    }

    __aicore__ inline void GetXInCopyParams(
        const QuantizeTilingData& runTilingData, int64_t xN, int64_t xLen, int64_t lastDimLen,
        DataCopyExtParams& copyParams)
    {
        copyParams.blockCount = xN;
        copyParams.blockLen = xLen * sizeof(T);
        if (runTilingData.baseLen > xLen) {
            copyParams.dstStride = (runTilingData.baseLen - xLen) * sizeof(T) / this->BLOCK_SIZE;
        } else {
            copyParams.dstStride = 0;
        }
        if (lastDimLen > xLen) {
            copyParams.srcStride = (lastDimLen - xLen) * sizeof(T);
        } else {
            copyParams.srcStride = 0;
        }
    }

    __aicore__ inline void GetOutCopyParams(
        const QuantizeTilingData& runTilingData, int64_t yN, int64_t yLen, int64_t lastDimLen,
        DataCopyExtParams& copyParams)
    {
        int64_t yLenReal = yLen;
        if constexpr (IsSameType<U, int4b_t>::value) {
            yLenReal = yLenReal / this->INT4_NUMS_IN_INT8_SPACE;
        }

        copyParams.blockCount = yN;
        copyParams.blockLen = yLenReal * sizeof(yCopyDtype);
        if (lastDimLen > yLen) {
            if constexpr (IsSameType<U, int4b_t>::value) {
                copyParams.dstStride = (lastDimLen - yLen) * sizeof(yCopyDtype) / this->INT4_NUMS_IN_INT8_SPACE;
            } else {
                copyParams.dstStride = (lastDimLen - yLen) * sizeof(yCopyDtype);
            }
        } else {
            copyParams.dstStride = 0;
        }
        if (runTilingData.baseLen > yLenReal) {
            copyParams.srcStride = (runTilingData.baseLen - yLenReal) * sizeof(yCopyDtype) / this->BLOCK_SIZE;
        } else {
            copyParams.srcStride = 0;
        }
    }

    __aicore__ inline void ParsePerHeadCoreBlocks()
    {
        if (tilingData_.blockAxis == 0) {
            // blockFactor is in [1, S]
            if (blockIdx_ == tilingData_.numCore - 1) {
                blockS_ = tilingData_.blockTailFactor;
            } else {
                blockS_ = tilingData_.blockFactor;
            }
            blockN_ = tilingData_.dim1;
            blockLen_ = tilingData_.dim2;
        } else if (tilingData_.blockAxis == 1) {
            // blockFactor is in [1, N], blockUnion is in [1, N/blockFactor]
            if (blockIdx_ % tilingData_.blockUnion == tilingData_.blockUnion - 1) {
                blockN_ = tilingData_.blockTailFactor;
            } else {
                blockN_ = tilingData_.blockFactor;
            }
            blockLen_ = tilingData_.dim2;
        } else {
            // blockFactor is in [1, D]
            blockN_ = 1;
            if (blockIdx_ % tilingData_.blockUnion == tilingData_.blockUnion - 1) {
                blockLen_ = tilingData_.blockTailFactor;
            } else {
                blockLen_ = tilingData_.blockFactor;
            }
        }
    }

    __aicore__ inline void CalcOffset()
    {
        if (tilingData_.blockAxis == 0) {
            // only split axis 0
            gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.dim1 * tilingData_.dim2;
            gmSOffset_ = 0;
        } else if (tilingData_.blockAxis == 1) {
            // only split axis 1, blockUnion means factor per block on split axis
            gmXOffset_ = blockIdx_ / tilingData_.blockUnion * tilingData_.dim1 * tilingData_.dim2 +
                         blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor * tilingData_.dim2;
            gmSOffset_ = blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor;
        } else {
            gmXOffset_ =
                (blockIdx_ / tilingData_.blockUnion * tilingData_.dim2 +
                 blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor);
            gmSOffset_ = blockIdx_ / tilingData_.blockUnion;
        }
    }

    __aicore__ inline void ProcessLoop()
    {
        for (int64_t i = 0; i < blockS_; ++i) {
            ProcessParamLoop();
            gmXOffset_ += tilingData_.dim1 * tilingData_.dim2;
        }
    }

    __aicore__ inline void ProcessParamLoop()
    {
        // main loop with column, for scale and offset only need copy once
        auto nLoopLen = tilingData_.baseN;
        auto nLoopNum = blockN_ / nLoopLen;
        auto nLoopTail = blockN_ % nLoopLen;

        // scale allows start from begin on each core
        int64_t baseSOffset = gmSOffset_;
        int64_t baseXOffset = gmXOffset_;

        for (int64_t i = 0; i < nLoopNum; ++i) {
            ProcessParamOneLoop(nLoopLen, baseSOffset, baseXOffset);
            baseXOffset += nLoopLen * tilingData_.dim2;
            baseSOffset += tilingData_.baseN;
        }
        if (nLoopTail != 0) {
            ProcessParamOneLoop(nLoopTail, baseSOffset, baseXOffset);
        }
    }

    // x，scale和offset用同样的处理逻辑，elementwise
    __aicore__ inline void ProcessParamOneLoop(int64_t nLoopLen, int64_t baseSOffset, int64_t baseXOffset)
    {
        // copy in scale
        CopyInParam<T1>(inQueueScale_, scaleGm_, nLoopLen, baseSOffset % tilingData_.dim1);
        auto scaleLocal = inQueueScale_.DeQue<T1>();
        CopyInParam<T2>(inQueueOffset_, offsetGm_, nLoopLen, baseSOffset % tilingData_.dim1);
        auto offsetLocal = inQueueOffset_.DeQue<T2>();
        ProcessInputLoop(nLoopLen, baseXOffset, scaleLocal, offsetLocal);
        inQueueOffset_.FreeTensor(offsetLocal);

        inQueueScale_.FreeTensor(scaleLocal);
    }

    __aicore__ inline void ProcessInputLoop(
        int64_t nLoopLen, int64_t baseXOffset, LocalTensor<T1>& scaleLocal, LocalTensor<T2>& offsetLocal)
    {
        auto loopLen = tilingData_.baseLen;
        auto lenLoopNum = blockLen_ / loopLen;
        auto lenLoopTail = blockLen_ % loopLen;
        for (auto i = 0; i < lenLoopNum; ++i) {
            CopyInX(nLoopLen, loopLen, baseXOffset);
            Compute(nLoopLen, loopLen, scaleLocal, offsetLocal);
            CopyOutY(nLoopLen, loopLen, baseXOffset);
            baseXOffset += tilingData_.baseLen;
        }
        if (lenLoopTail != 0) {
            CopyInX(nLoopLen, lenLoopTail, baseXOffset);
            Compute(nLoopLen, lenLoopTail, scaleLocal, offsetLocal);
            CopyOutY(nLoopLen, lenLoopTail, baseXOffset);
        }
    }

    template <typename dtypeCopyIn>
    __aicore__ inline void CopyInParam(
        TQue<QuePosition::VECIN, bufferNum_>& inQueue, GlobalTensor<dtypeCopyIn>& inGm, int64_t paramLen,
        int64_t paramOffset)
    {
        auto paramLocal = inQueue.AllocTensor<dtypeCopyIn>();
        static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
        MultiCopyLoopInfo<QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::MULTI_COPY_DIM> multiCopyParams;
        // src stride info per loop.
        multiCopyParams.loopSrcStride[0] = 0;
        multiCopyParams.loopSrcStride[1] = 1;
        // dst stride info per loop.
        multiCopyParams.loopDstStride[0] = 1;
        multiCopyParams.loopDstStride[1] = tilingData_.baseLen;
        // Loop size per loop.
        multiCopyParams.loopSize[0] = tilingData_.baseLen;
        multiCopyParams.loopSize[1] = paramLen;
        dtypeCopyIn constValue = 0;
        AscendC::MultiCopyParams<dtypeCopyIn, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::MULTI_COPY_DIM>
            paramsMain = {multiCopyParams, constValue};
        AscendC::DataCopy<
            dtypeCopyIn, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::MULTI_COPY_DIM, copyConfig>(
            paramLocal, inGm[paramOffset], paramsMain);
        inQueue.EnQue(paramLocal);
    }

    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset)
    {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim2, copyParams);
        DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset)
    {
        LocalTensor<yCopyDtype> outLocal = outQueueY_.DeQue<yCopyDtype>();
        if constexpr (IsSameType<U, int4b_t>::value) {
            yOutOffset = yOutOffset >> 1;
        }
        DataCopyExtParams copyParams;
        this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim2, copyParams);
        DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
        outQueueY_.FreeTensor(outLocal);
    }

    __aicore__ inline void Compute(
        int64_t nRow, int64_t dataCount, LocalTensor<T1>& scaleLocal, LocalTensor<T2>& offsetLocal)
    {
        auto inLocal = inQueueX_.DeQue<T>();
        auto outLocal = outQueueY_.AllocTensor<yCopyDtype>();

        __local_mem__ T* xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
        __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)scaleLocal.GetPhyAddr();
        __local_mem__ T2* offsetLocalAddr = (__local_mem__ T2*)offsetLocal.GetPhyAddr();
        __local_mem__ yCopyDtype* outLocalAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        __VEC_SCOPE__
        {
            // x: fp32, fp16, bf16
            AscendC::MicroAPI::RegTensor<T> vregX;
            AscendC::MicroAPI::RegTensor<float> vregFloatX;
            // scales: fp32, fp16, bf16
            AscendC::MicroAPI::RegTensor<T1> vregS;
            AscendC::MicroAPI::RegTensor<float> vregFloatS;
            // zero_points: fp32, fp16, bf16
            AscendC::MicroAPI::RegTensor<T2> vregO;
            AscendC::MicroAPI::RegTensor<half> vregHalfO;
            AscendC::MicroAPI::RegTensor<float> vregFloatO;
            // y: int8, hifp8, fp8_e5m2, fp8_e4m3, int4
            AscendC::MicroAPI::RegTensor<float> vregFloatY;
            AscendC::MicroAPI::RegTensor<half> vregHalfY;
            AscendC::MicroAPI::RegTensor<yCopyDtype> vregY;

            AscendC::MicroAPI::RegTensor<float> vregTmp1;
            AscendC::MicroAPI::RegTensor<float> vregTmp2;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg mask4Int4 =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

            mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(nRow); ++j) {
                uint32_t count = dataCount;
                uint16_t vfLoopNum = (dataCount + VL - 1) / VL;
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    // ld and cast for x
                    if constexpr (IsSameType<T, float>::value) {
                        // fp32
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregFloatX, xLocalAddr + i * VL + j * tilingData_.baseLen);
                    } else if constexpr (IsSameType<T, half>::value) {
                        // fp16
                        AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, xLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                            vregFloatX, vregX, mask);
                    } else {
                        // bf16
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, xLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, T,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatX, vregX, mask);
                    }

                    // ld and cast for scale
                    if constexpr (IsSameType<T1, float>::value) {
                        // fp32
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregFloatS, scaleLocalAddr + i * VL + j * tilingData_.baseLen);
                    } else if constexpr (IsSameType<T1, half>::value) {
                        // fp16
                        AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                            vregFloatS, vregS, mask);
                    } else {
                        // bf16
                        AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, T1,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatS, vregS, mask);
                    }

                    // ld and cast for offset
                    if constexpr (IsSameType<T2, int32_t>::value) {
                        // int32
                        AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, T2,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT32_TO_FP32>(
                            vregFloatO, vregO, mask);
                    } else if constexpr (IsSameType<T2, int8_t>::value) {
                        // int8
                        AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            vregO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            half, T2,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT8_TO_HALF>(
                            vregHalfO, vregO, mask);
                        AscendC::MicroAPI::Cast<
                            float, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                            vregFloatO, vregHalfO, mask);
                    } else if constexpr (IsSameType<T2, uint8_t>::value) {
                        // uint8
                        AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            vregO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            half, T2,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_UINT8_TO_HALF>(
                            vregHalfO, vregO, mask);
                        AscendC::MicroAPI::Cast<
                            float, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                            vregFloatO, vregHalfO, mask);
                    } else if constexpr (IsSameType<T2, float>::value) {
                        // fp32
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregFloatO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                    } else if constexpr (IsSameType<T2, half>::value) {
                        // fp16
                        AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                            vregFloatO, vregO, mask);
                    } else {
                        // bf16
                        AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregO, offsetLocalAddr + i * VL + j * tilingData_.baseLen);
                        AscendC::MicroAPI::Cast<
                            float, T2,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatO, vregO, mask);
                    }

                    if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                        static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {
                            AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
                        AscendC::MicroAPI::Div<float, &divMode>(vregTmp1, vregFloatX, vregFloatS, mask);
                    } else {
                        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                            vregTmp1, vregFloatX, vregFloatS, mask);
                        if constexpr (SqrtMode == TPL_SQRT_MODE) {
                            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                                vregTmp1, vregTmp1, vregFloatS, mask);
                        }
                    }

                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        vregFloatY, vregTmp1, vregFloatO, mask);

                    // cast and sd for y
                    if constexpr (IsSameType<U, hifloat8_t>::value) {
                        // hifp8
                        AscendC::MicroAPI::Cast<
                            U, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                        // fp8_e5m2
                        AscendC::MicroAPI::Cast<
                            U, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
                        // fp8_e4m3
                        AscendC::MicroAPI::Cast<
                            U, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, int8_t>::value) {
                        // int8 支持
                        AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
                        AscendC::MicroAPI::Cast<
                            int16_t, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                            vregInt16Y, vregFloatY, mask);
                        AscendC::MicroAPI::Cast<
                            half, int16_t,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                            vregHalfY, vregInt16Y, mask);
                        AscendC::MicroAPI::Cast<
                            int8_t, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_INT8>(
                            vregY, vregHalfY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, uint8_t>::value) {
                        // uint8
                        AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
                        AscendC::MicroAPI::Cast<
                            int16_t, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                            vregInt16Y, vregFloatY, mask);
                        AscendC::MicroAPI::Cast<
                            half, int16_t,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                            vregHalfY, vregInt16Y, mask);
                        AscendC::MicroAPI::Cast<
                            uint8_t, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_UINT8>(
                            vregY, vregHalfY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, int32_t>::value) {
                        // int32
                        AscendC::MicroAPI::Cast<
                            U, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT32>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                            outLocalAddr + i * VL + j * tilingData_.baseLen, vregY, mask);
                    } else if constexpr (IsSameType<U, int4b_t>::value) {
                        AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
                        AscendC::MicroAPI::RegTensor<uint16_t> vregTmp1Y;
                        AscendC::MicroAPI::RegTensor<yCopyDtype> vregTmp2Y;
                        AscendC::MicroAPI::Cast<
                            int16_t, float,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                            vregInt16Y, vregFloatY, mask);
                        AscendC::MicroAPI::Cast<
                            half, int16_t,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                            vregHalfY, vregInt16Y, mask);

                        AscendC::MicroAPI::Pack(vregTmp1Y, (AscendC::MicroAPI::RegTensor<uint32_t>&)vregHalfY);
                        AscendC::MicroAPI::Cast<
                            int4x2_t, half,
                            QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_F16_TO_I8>(
                            (AscendC::MicroAPI::RegTensor<int4x2_t>&)vregTmp2Y,
                            (AscendC::MicroAPI::RegTensor<half>&)vregTmp1Y, mask);
                        AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + (i * VL / 2 + j * tilingData_.baseLen), vregTmp2Y, mask4Int4);
                    }
                }
            }
        }
        inQueueX_.FreeTensor(inLocal);
        outQueueY_.EnQue(outLocal);
    }
};
} // namespace QuantizeOp
#endif // QUANTIZE_PERHEAD_REGBASE_H