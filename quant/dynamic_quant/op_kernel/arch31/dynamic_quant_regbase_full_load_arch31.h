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
 * \file dynamic_quant_regbase_full_load_arch31.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_FULL_LOAD_H
#define DYNAMIC_QUANT_REGBASE_FULL_LOAD_H

#include "dynamic_quant_regbase_base_arch31.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;

// isSymmetric == false 使用
constexpr float FP8_E5M2_OFFSET_VALUE = 114688.0f;
constexpr float FP8_E4M3FN_OFFSET_VALUE = 896.0f;
constexpr float HIFLOAT8_OFFSET_VALUE = 65536.0f;
constexpr float INT8_OFFSET_VALUE = 255.0f;
constexpr float INT4_OFFSET_VALUE = 15.0f;
constexpr float NEGATIVE_ONE = -1.0f;
constexpr uint32_t REG_LEN = 64;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

template <typename xDtype, typename yDtype, bool hasSmooth, uint32_t useBufferNum, bool isSymmetrical = true>
class DynamicQuantRegbaseFullLoad : public DynamicQuantBase {
public:
    __aicore__ inline DynamicQuantRegbaseFullLoad(TPipe* pipe) { pPipe = pipe; }

    // 没有group_index输入
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingDataArch31* __restrict tilingData)
    {
        DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
        ParseTilingData(tilingData);
        InitParams(offset);
        InitAndSetBuffer(x, smooth_scales, y, scale, offset);
        SetMaxValue();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= tilingData_.coreNum) {
            return;
        }

        // loopCnt是UB搬运次数,multiRowNum是一次搬运（一次用满UB）的row数量
        for (int32_t i = 0; i < loopCnt; i++) {
            LoopProcess(multiRowNum, i);
        }

        // 处理剩余row，最后一个loop
        if (remainRow > 0) {
            LoopProcess(remainRow, loopCnt);
        }
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        if constexpr (hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
            pPipe->InitBuffer(smoothQueue, useBufferNum, sizeHalfLen * sizeof(xDtype));
        }

        // headCoreGM申请
        if (blockIdx < tilingData_.headCoreNum) {
            baseRow = blockIdx * rowPerHeadCore;
            // headCore处理的元素总数lenHead、outLenHead。如果输出是INT4，这边outLenHead数量是lenHead一半
            inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
            // scale每次偏移每个核处理的row数量的地址
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
            if constexpr (isSymmetrical == false) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerHeadCore);
            }
            // tailCoreGM申请
        } else {
            baseRow = tilingData_.headCoreNum * rowPerHeadCore + (blockIdx - tilingData_.headCoreNum) * rowPerTailCore;
            inGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead +
                                      (blockIdx - tilingData_.headCoreNum) * outLenTail,
                                  outLenTail);
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + baseRow, rowPerTailCore);
            if constexpr (isSymmetrical == false) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerTailCore);
            }
        }

        // 申请Buffer大小offsetQueue，inQueue，outQueue，scaleQueue，smoothQueue，groupIndexQueue
        if constexpr (isSymmetrical == false) {
            pPipe->InitBuffer(offsetQueue, useBufferNum, sizeFloatLen * sizeof(float));
        }
        pPipe->InitBuffer(inQueue, useBufferNum, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outQueue, useBufferNum, outLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleQueue, useBufferNum, sizeFloatLen * sizeof(float));
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            pPipe->InitBuffer(halfOutBuf_, lenMultiRow * sizeof(half));
        }
        tailNum = tilingData_.rowLen % REG_LEN;
        if (tailNum == 0) {
            tailNum = REG_LEN;
        }
    }

    __aicore__ inline void LoopProcess(int32_t multiRow, int32_t loopNum)
    {
        CopyIn(multiRow, loopNum);
        Compute(multiRow);
        CopyOut(multiRow, loopNum);
    }

    __aicore__ inline void CopyIn(int32_t multiRow, int32_t loopNum)
    {
        LocalTensor<xDtype> inLocal = inQueue.template AllocTensor<xDtype>();
        DataCopyExtParams copyParams = {static_cast<uint16_t>(multiRow),
                                        static_cast<uint32_t>(tilingData_.rowLen * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> padParams{true, 0, rightPadding, 0};
        DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
        inQueue.template EnQue(inLocal);

        if constexpr (hasSmooth) {
            LocalTensor<xDtype> smoothLocal = smoothQueue.template AllocTensor<xDtype>();
            DataCopyExtParams smoothCopyParams = {1, static_cast<uint32_t>(tilingData_.rowLen * sizeof(xDtype)), 0, 0,
                                                  0};
            DataCopyPad(smoothLocal, smoothGm, smoothCopyParams, padParams);
            smoothQueue.template EnQue(smoothLocal);
        }
    }

    __aicore__ inline void Compute(int32_t multiRow)
    {
        uint32_t index = 0;
        LocalTensor<float> scaleLocal = scaleQueue.template AllocTensor<float>();
        LocalTensor<yDtype> yLocal = outQueue.template AllocTensor<yDtype>();
        LocalTensor<xDtype> xLocal = inQueue.template DeQue<xDtype>();
        LocalTensor<xDtype> smoothLocal;
        LocalTensor<float> offsetLocal;

        __local_mem__ xDtype* xAddr = (__local_mem__ xDtype*)xLocal.GetPhyAddr();
        __local_mem__ xDtype* smoothAddr;

        __local_mem__ yDtype* yAddr = (__local_mem__ yDtype*)yLocal.GetPhyAddr();
        __local_mem__ float* scaleAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
        __local_mem__ float* offsetAddr;

        if constexpr (isSymmetrical == false) {
            offsetLocal = offsetQueue.template AllocTensor<float>();
            offsetAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
        }

        if constexpr (hasSmooth) {
            smoothLocal = smoothQueue.template DeQue<xDtype>();
            smoothAddr = (__local_mem__ xDtype*)smoothLocal.GetPhyAddr();
        }

        ComputeVF(xAddr, smoothAddr, yAddr, scaleAddr, offsetAddr, multiRow);

        if constexpr (IsSameType<yDtype, int8_t>::value) {
            auto halfLocal = halfOutBuf_.Get<half>();
            PipeBarrier<PIPE_V>();
            Cast(yLocal, halfLocal, RoundMode::CAST_TRUNC, multiRow * sizeHalfLen);
            PipeBarrier<PIPE_V>();
        }

        outQueue.template EnQue<yDtype>(yLocal);
        scaleQueue.template EnQue<float>(scaleLocal);
        if constexpr (isSymmetrical == false) {
            offsetQueue.template EnQue<float>(offsetLocal);
        }
        inQueue.FreeTensor(xLocal);
        if constexpr (hasSmooth) {
            smoothQueue.FreeTensor(smoothLocal);
        }
    }

    __aicore__ inline void CopyOut(int32_t multiRow, int32_t loopCount)
    {
        LocalTensor<yDtype> yLocal = outQueue.template DeQue<yDtype>();
        LocalTensor<float> scaleLocal = scaleQueue.template DeQue<float>();

        if constexpr (isSymmetrical == false) {
            LocalTensor<float> offsetLocal = offsetQueue.template DeQue<float>();
            DataCopyExtParams offsetCopyParams{1, static_cast<uint32_t>(multiRow * sizeof(float)), 0, 0, 0};
            DataCopyPad(offsetGm[loopCount * multiRowNum], offsetLocal, offsetCopyParams);
            offsetQueue.FreeTensor(offsetLocal);
        }

        DataCopyExtParams copyParams{static_cast<uint16_t>(multiRow),
                                     static_cast<uint32_t>(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};

        DataCopyPad(outGm[loopCount * lenGMMultiRow], yLocal, copyParams);
        DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(multiRow * sizeof(float)), 0, 0, 0};
        DataCopyPad(scaleGm[loopCount * multiRowNum], scaleLocal, scaleCopyParams);

        outQueue.FreeTensor(yLocal);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void DataCopyInputVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                           AscendC::Reg::RegTensor<float>& vregRes, AscendC::Reg::MaskReg pregMask)
    {
        AscendC::Reg::RegTensor<xDtype> vregX;
        AscendC::Reg::RegTensor<xDtype> vregSmooth;
        AscendC::Reg::RegTensor<float> vregSmoothFp32;

        AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xAddr);
        AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vregRes, vregX, pregMask);
        if constexpr (hasSmooth) {
            AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregSmooth, smoothAddr);
            AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, pregMask);
            AscendC::Reg::Mul(vregRes, vregRes, vregSmoothFp32, pregMask);
        }
    }

    __aicore__ inline void ComputeYVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                      __local_mem__ yDtype* yAddr, AscendC::Reg::RegTensor<float>& vregDupScale,
                                      AscendC::Reg::RegTensor<float>& vregDupOffset, int32_t indexRow)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<float> vregXDivScale;
        AscendC::Reg::RegTensor<float> vregYFp32;
        AscendC::Reg::RegTensor<int16_t> vregYInt16; // cast成最终y之前的int16类型
        AscendC::Reg::RegTensor<half> vregYFp16;     // cast成最终y之前的half类型
        AscendC::Reg::RegTensor<yDtype> vregY;       // 最终y

        AscendC::Reg::MaskReg preg2;
        AscendC::Reg::MaskReg pregHalf = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::H>();

        __local_mem__ half* halfAddr = nullptr;
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            auto halfLocal = halfOutBuf_.Get<half>();
            halfAddr = (__local_mem__ half*)halfLocal.GetPhyAddr();
        }

        uint32_t sreg1 = sizeHalfLen;
        uint16_t vfLoop = (sizeHalfLen + VL - 1) / VL;
        for (uint16_t j = 0; j < vfLoop; j++) {
            auto addr = yAddr + indexRow * outAlignLen + j * VL;
            preg2 = AscendC::Reg::UpdateMask<float>(sreg1);
            DataCopyInputVF(xAddr + indexRow * sizeHalfLen + j * VL, smoothAddr + j * VL, vregInput, preg2);

            if constexpr (isSymmetrical == true) {
                AscendC::Reg::Div(vregYFp32, vregInput, vregDupScale, preg2);
            } else if constexpr (isSymmetrical == false) {
                AscendC::Reg::Div(vregXDivScale, vregInput, vregDupScale, preg2);
                AscendC::Reg::Add(vregYFp32, vregXDivScale, vregDupOffset, preg2);
            }
            if constexpr (IsSameType<yDtype, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregYFp32, preg2);
                AscendC::Reg::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
                AscendC::Reg::DataCopy<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    halfAddr + indexRow * sizeHalfLen + j * VL, vregYFp16, preg2);
            } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<yDtype, float, castTraitF32tofp8>(vregY, vregYFp32, preg2);
                AscendC::Reg::DataCopy<yDtype, AscendC::Reg::StoreDist::DIST_PACK_B16>(addr, vregY, preg2);
            }
        }
    }

    __aicore__ inline void ComputeScaleVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                          __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr,
                                          AscendC::Reg::RegTensor<float>& vregDupScale,
                                          AscendC::Reg::RegTensor<float>& vregDupOffset, int32_t indexRow)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<float> vregAbs;
        AscendC::Reg::RegTensor<float> vregScale;
        AscendC::Reg::RegTensor<float> vregMinX;
        AscendC::Reg::RegTensor<float> vregMaxX;
        AscendC::Reg::RegTensor<float> vregReduceMaxX;
        AscendC::Reg::RegTensor<float> vregReduceMinX;
        AscendC::Reg::RegTensor<float> vregMaxSubMin;
        AscendC::Reg::RegTensor<float> vregMaxDivScale;
        AscendC::Reg::RegTensor<float> vregNegMaxDivScale;
        AscendC::Reg::RegTensor<float> vregOffset;
        AscendC::Reg::RegTensor<float> vregReduceMaxXTail;
        AscendC::Reg::RegTensor<float> vregReduceMinXTail;
        AscendC::Reg::RegTensor<float> vregFinalMax;
        AscendC::Reg::RegTensor<float> vregFinalMin;

        AscendC::Reg::MaskReg preg0;
        AscendC::Reg::MaskReg preg1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg preg4;
        AscendC::Reg::MaskReg preg5 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();

        AscendC::Reg::UnalignReg ureg0;
        AscendC::Reg::UnalignReg ureg1;

        uint32_t rowCount = sizeHalfLen;
        uint16_t vfLoop = (rowCount + VL - 1) / VL;
        uint32_t sreg0 = rowCount;
        uint32_t sregTail = tailNum;
        AscendC::Reg::Duplicate(vregMaxX, NEG_INFINITY, preg1);
        if constexpr (isSymmetrical == true) {
            // 1. compute max value for every vf loop  2.do reducemax in the end
            for (uint16_t j = 0; j < vfLoop; j++) {
                preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
                DataCopyInputVF(xAddr + indexRow * rowCount + j * VL, smoothAddr + j * VL, vregInput, preg0);
                AscendC::Reg::Abs(vregAbs, vregInput, preg0);
                AscendC::Reg::Max(vregMaxX, vregAbs, vregMaxX, preg1);
            }
            AscendC::Reg::ReduceMax(vregReduceMaxX, vregMaxX, preg1);
            AscendC::Reg::Muls(vregScale, vregReduceMaxX, maxValue, preg1);
            AscendC::Reg::Duplicate(vregDupScale, vregScale, preg1);
            AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, vregScale,
                                                                                              ureg0, 1);
        } else if constexpr (isSymmetrical == false) {
            AscendC::Reg::Duplicate(vregMinX, POS_INFINITY, preg1);
            // mask can't conver all loop, because 0(no mask value) is not min or max, compute until last second loop
            for (uint16_t j = 0; j < vfLoop - 1; j++) {
                preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
                DataCopyInputVF(xAddr + indexRow * rowCount + j * VL, smoothAddr + j * VL, vregInput, preg0);
                AscendC::Reg::Max(vregMaxX, vregInput, vregMaxX, preg1);
                AscendC::Reg::Min(vregMinX, vregInput, vregMinX, preg1);
            }
            AscendC::Reg::ReduceMax(vregReduceMaxX, vregMaxX, preg1);
            AscendC::Reg::ReduceMin(vregReduceMinX, vregMinX, preg1);

            // finnal compute max and min
            preg4 = AscendC::Reg::UpdateMask<float>(sregTail);
            DataCopyInputVF(xAddr + indexRow * rowCount + (vfLoop - 1) * VL, smoothAddr + (vfLoop - 1) * VL, vregInput,
                            preg4);
            AscendC::Reg::ReduceMax(vregReduceMaxXTail, vregInput, preg4);
            AscendC::Reg::ReduceMin(vregReduceMinXTail, vregInput, preg4);
            AscendC::Reg::Max(vregFinalMax, vregReduceMaxX, vregReduceMaxXTail, preg5);
            AscendC::Reg::Min(vregFinalMin, vregReduceMinX, vregReduceMinXTail, preg5);

            // compute scale and offset
            AscendC::Reg::Sub(vregMaxSubMin, vregFinalMax, vregFinalMin, preg5);
            AscendC::Reg::Muls(vregScale, vregMaxSubMin, offsetDivValue, preg5);
            AscendC::Reg::Duplicate(vregDupScale, vregScale, preg1);
            AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, vregScale,
                                                                                              ureg0, 1);
            AscendC::Reg::Div<float, &mode>(vregMaxDivScale, vregFinalMax, vregScale, preg5);
            AscendC::Reg::Muls(vregNegMaxDivScale, vregMaxDivScale, NEGATIVE_ONE, preg5);
            AscendC::Reg::Adds(vregOffset, vregNegMaxDivScale, offsetValue, preg5); //
            AscendC::Reg::Duplicate(vregDupOffset, vregOffset, preg1);
            AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(offsetAddr, vregOffset,
                                                                                              ureg1, 1);
        }
        AscendC::Reg::DataCopyUnAlignPost(scaleAddr, ureg0, 0);
        if constexpr (isSymmetrical == false) {
            AscendC::Reg::DataCopyUnAlignPost(offsetAddr, ureg1, 0);
        }
    }

    __aicore__ inline void ComputeVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                     __local_mem__ yDtype* yAddr, __local_mem__ float* scaleAddr,
                                     __local_mem__ float* offsetAddr, int32_t multiRow)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> vregDupScale;
            AscendC::Reg::RegTensor<float> vregDupOffset;

            for (uint16_t indexRow = 0; indexRow < (uint16_t)multiRow; indexRow++) {
                ComputeScaleVF(xAddr, smoothAddr, scaleAddr + indexRow, offsetAddr + indexRow, vregDupScale,
                               vregDupOffset, indexRow);
                ComputeYVF(xAddr, smoothAddr, yAddr, vregDupScale, vregDupOffset, indexRow);
            }
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            maxValue = static_cast<float>(1.0) / INT8_MAX_VALUE;
            offsetValue = INT8_MAX_VALUE;
            offsetDivValue = static_cast<float>(1.0) / INT8_OFFSET_VALUE;
        } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
            offsetValue = FP8_E5M2_MAX_VALUE;
            offsetDivValue = static_cast<float>(1.0) / FP8_E5M2_OFFSET_VALUE;
        } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
            offsetValue = FP8_E4M3FN_MAX_VALUE;
            offsetDivValue = static_cast<float>(1.0) / FP8_E4M3FN_OFFSET_VALUE;
        }
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, useBufferNum> inQueue;
    TQue<QuePosition::VECIN, useBufferNum> smoothQueue;
    TQue<QuePosition::VECOUT, useBufferNum> outQueue;
    TQue<QuePosition::VECOUT, useBufferNum> scaleQueue;
    TQue<QuePosition::VECOUT, useBufferNum> offsetQueue;
    TBuf<QuePosition::VECCALC> halfOutBuf_;

    /* global memory address */
    GlobalTensor<xDtype> inGm, smoothGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<yDtype> outGm;

    int32_t baseRow = 0;
    float maxValue = 0.0;
    float offsetValue = 0.0;
    float offsetDivValue = 0.0;
    uint32_t tailNum = 0;
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    static constexpr AscendC::Reg::DivSpecificMode mode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
    constexpr static AscendC::Reg::CastTrait castTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTraitF32ToI16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castTraitI16ToF16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::Reg::CastTrait castTraitF16ToI8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};
    constexpr static AscendC::Reg::CastTrait castTraitF32tofp8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castTraitF32toh8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
};
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_REGBASE_FULL_LOAD_H
