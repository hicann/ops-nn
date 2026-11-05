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
 * \file dynamic_quant_regbase_full_load_pertensor_arch31.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_FULL_LOAD_PERTENSOR_H
#define DYNAMIC_QUANT_REGBASE_FULL_LOAD_PERTENSOR_H
#include "dynamic_quant_regbase_base_arch31.h"
#include <cmath>

namespace DynamicQuantNDPerTensorOpt {
using namespace AscendC;

constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;
constexpr float FP8_E5M2_MAX_VALUE_NO_SYM = 114688.0f;
constexpr float FP8_E4M3FN_MAX_VALUE_NO_SYM = 896.0f;
constexpr float HIFLOAT8_MAX_VALUE_NO_SYM = 65536.0f;
constexpr float INT8_MAX_VALUE_NO_SYM = 255.0f;
constexpr float INT4_MAX_VALUE_NO_SYM = 15.0f;
constexpr uint32_t TWO_FIVE_SIX = 256;
constexpr float MIN_FLOAT_VALUE = -INFINITY;
constexpr float MAX_FLOAT_VALUE = INFINITY;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t ONE = 1;

template <typename xDtype, typename yDtype, bool hasSmooth, uint32_t useBufferNum, bool isSymmertrical>
class DynamicQuantRegbaseFullLoadPertensor : public DynamicQuantNDOpt::DynamicQuantBase {
public:
    __aicore__ inline DynamicQuantRegbaseFullLoadPertensor(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingDataArch31* __restrict tilingData)
    {
        DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
        ParseTilingData(tilingData);
        InitParams(offset);
        InitAndSetBuffer(x, smooth_scales, y, scale, offset, workSpace);
        SetMaxValue();
    }

    __aicore__ inline void ProcessScaleRowLoop(int32_t multiRow, int32_t loopNum)
    {
        CopyIn(multiRow, loopNum);
        ComputeMaxRowScale(multiRow);
    }

    __aicore__ inline void ProcessScaleRow()
    {
        LocalTensor<float> MaxToWorkSpaceLocal;
        LocalTensor<float> MinToWorkSpaceLocal;
        LocalTensor<float> scaleToWorkSpaceLocal;

        if constexpr (isSymmertrical == false) {
            MaxToWorkSpaceLocal = MaxToWorkSpaceQueue.template AllocTensor<float>();
            AscendC::Duplicate(MaxToWorkSpaceLocal, MIN_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MaxToWorkSpaceQueue.template EnQue(MaxToWorkSpaceLocal);

            MinToWorkSpaceLocal = MinToWorkSpaceQueue.template AllocTensor<float>();
            AscendC::Duplicate(MinToWorkSpaceLocal, MAX_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MinToWorkSpaceQueue.template EnQue(MinToWorkSpaceLocal);
        } else {
            scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.template AllocTensor<float>();
            AscendC::Duplicate(scaleToWorkSpaceLocal, (float)0.0, SIXTY_FOUR, ONE, ONE, EIGHT);
            scaleToWorkSpaceQueue.template EnQue(scaleToWorkSpaceLocal);
        }
        for (int32_t i = 0; i < loopCnt; i++) {
            ProcessScaleRowLoop(multiRowNum, i);
        }

        if (remainRow > 0) {
            ProcessScaleRowLoop(remainRow, loopCnt);
        }
        CopyUB2Workspace(1);
    }
    __aicore__ inline void ProcessScaleCol()
    {
        LocalTensor<float> MaxOutLocal;
        LocalTensor<float> MinOutLocal;
        LocalTensor<float> scaleOutLocal;
        LocalTensor<float> offsetLocal;
        __local_mem__ float* MaxOutLocalAddr;
        __local_mem__ float* MinOutLocalAddr;
        __local_mem__ float* scaleOutLocalAddr;
        __local_mem__ float* offsetLocalAddr;

        if constexpr (isSymmertrical == false) {
            MaxOutLocal = MaxOutQueue.template AllocTensor<float>();
            AscendC::Duplicate(MaxOutLocal, MIN_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MaxOutLocalAddr = (__local_mem__ float*)MaxOutLocal.GetPhyAddr();

            MinOutLocal = MinOutQueue.template AllocTensor<float>();
            AscendC::Duplicate(MinOutLocal, MAX_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MinOutLocalAddr = (__local_mem__ float*)MinOutLocal.GetPhyAddr();

            scaleOutLocal = scaleOutQueue.template AllocTensor<float>();
            AscendC::Duplicate(scaleOutLocal, (float)0.0, SIXTY_FOUR, ONE, ONE, EIGHT);
            scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();

            offsetLocal = offsetQueue.template AllocTensor<float>();
            AscendC::Duplicate(offsetLocal, (float)0.0, SIXTY_FOUR, ONE, ONE, EIGHT);
            offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
        } else {
            scaleOutLocal = scaleOutQueue.template AllocTensor<float>();
            AscendC::Duplicate(scaleOutLocal, (float)0.0, SIXTY_FOUR, ONE, ONE, EIGHT);
            scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
        }
        scaleOffset = 0;
        CopyInScaleByEle(scaleOffset, tilingData_.coreNum);
        ComputeMaxColScale(tilingData_.coreNum, MaxOutLocalAddr, MinOutLocalAddr, scaleOutLocalAddr, offsetLocalAddr);
        if constexpr (isSymmertrical == false) {
            MaxOutQueue.FreeTensor(MaxOutLocal);
            MinOutQueue.FreeTensor(MinOutLocal);
            scaleOutQueue.template EnQue(scaleOutLocal);
            offsetQueue.template EnQue(offsetLocal);
        } else {
            scaleOutQueue.template EnQue(scaleOutLocal);
        }
    }

    __aicore__ inline void ProcessScale()
    {
        ProcessScaleRow();
        SyncAll();
        ProcessScaleCol();
    }

    __aicore__ inline void ProcessYRow(int32_t multiRow, int32_t loopNum, __local_mem__ float* scaleAddr,
                                       __local_mem__ float* offsetAddr)
    {
        CopyIn(multiRow, loopNum);
        ComputeY(multiRow, scaleAddr, offsetAddr);
        CopyOut(multiRow, loopNum);
    }

    __aicore__ inline void ProcessY()
    {
        LocalTensor<float> scaleOutLocal = scaleOutQueue.template DeQue<float>();
        __local_mem__ float* scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
        LocalTensor<float> offsetLocal;
        __local_mem__ float* offsetLocalAddr;

        if constexpr (isSymmertrical == false) {
            offsetLocal = offsetQueue.template DeQue<float>();
            offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
        }

        for (int32_t i = 0; i < loopCnt; i++) {
            ProcessYRow(multiRowNum, i, scaleOutLocalAddr, offsetLocalAddr);
        }
        if (remainRow > 0) {
            ProcessYRow(remainRow, loopCnt, scaleOutLocalAddr, offsetLocalAddr);
        }
        if (blockIdx == 0) {
            DataCopyParams copyParams{1, (uint16_t)(sizeof(float)), 0, 0};
            DataCopyPad(scaleGm[0], scaleOutLocal, copyParams);
            if constexpr (isSymmertrical == false) {
                DataCopyPad(offsetGm[0], offsetLocal, copyParams);
            }
        }
        scaleOutQueue.FreeTensor(scaleOutLocal);
        if constexpr (isSymmertrical == false) {
            offsetQueue.FreeTensor(offsetLocal);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= tilingData_.coreNum) {
            return;
        }
        ProcessScale();
        ProcessY();
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                            GM_ADDR workSpace)
    {
        if constexpr (hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
            pPipe->InitBuffer(smoothQueue, useBufferNum, sizeHalfLen * sizeof(xDtype));
        }

        if (blockIdx < tilingData_.headCoreNum) {
            inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
        } else {
            inGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead +
                                      (blockIdx - tilingData_.headCoreNum) * outLenTail,
                                  outLenTail);
        }

        if (blockIdx == 0) {
            scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);
        }

        if constexpr (isSymmertrical == false) {
            workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tilingData_.coreNum);
            workspaceTmp2.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workSpace + tilingData_.coreNum * sizeof(float)), tilingData_.coreNum);
        } else {
            workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tilingData_.coreNum);
        }
        if constexpr (isSymmertrical == false) {
            offsetGm.SetGlobalBuffer((__gm__ float*)offset);
            pPipe->InitBuffer(offsetQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MaxOutQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MinOutQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MaxToWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MaxFromWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MinToWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(MinFromWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
        } else {
            pPipe->InitBuffer(scaleToWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
            pPipe->InitBuffer(scaleFromWorkSpaceQueue, useBufferNum, TWO_FIVE_SIX);
        }
        pPipe->InitBuffer(inQueue, useBufferNum, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outQueue, useBufferNum, outLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleOutQueue, useBufferNum, TWO_FIVE_SIX);
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            pPipe->InitBuffer(halfOutBuf_, outLen * sizeof(half));
        }
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

    __aicore__ inline void CopyInScaleByEle(int64_t offset, uint32_t elementNum)
    {
        DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(float)), 0, 0};
        if constexpr (isSymmertrical == false) {
            LocalTensor<float> MaxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(MaxFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
            MaxFromWorkSpaceQueue.template EnQue(MaxFromWorkSpaceLocal);

            LocalTensor<float> MinFromWorkSpaceLocal = MinFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(MinFromWorkSpaceLocal, workspaceTmp2[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
            MinFromWorkSpaceQueue.template EnQue(MinFromWorkSpaceLocal);
        } else {
            LocalTensor<float> scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(scaleFromWorkSpaceLocal, workspaceTmp1[offset],
                        {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            scaleFromWorkSpaceQueue.template EnQue(scaleFromWorkSpaceLocal);
        }
    }
    __aicore__ inline void CopyUB2Workspace(int64_t size)
    {
        if constexpr (isSymmertrical == false) {
            auto tmp1 = MaxToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp1[blockIdx], tmp1, {1, (uint16_t)(size * sizeof(float)), 0, 0});
            MaxToWorkSpaceQueue.FreeTensor(tmp1);

            auto tmp2 = MinToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp2[blockIdx], tmp2, {1, (uint16_t)(size * sizeof(float)), 0, 0});
            MinToWorkSpaceQueue.FreeTensor(tmp2);
        } else {
            auto tmp = scaleToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp1[blockIdx], tmp, {1, (uint16_t)(size * sizeof(float)), 0, 0});
            scaleToWorkSpaceQueue.FreeTensor(tmp);
        }
    }

    __aicore__ inline void ComputeMaxRowScaleVF(__local_mem__ xDtype* inLocalAddr,
                                                __local_mem__ xDtype* smoothLocalAddr,
                                                __local_mem__ float* scaleLocalAddr, __local_mem__ float* maxLocalAddr,
                                                __local_mem__ float* minLocalAddr, uint32_t multiRow)
    {
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint32_t rowCount = sizeHalfLen;
        uint16_t vfLoop = (tilingData_.rowLen + VL - 1) / VL;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<xDtype> vreg1;
            AscendC::Reg::RegTensor<xDtype> vreg2;
            AscendC::Reg::RegTensor<float> vreg3;
            AscendC::Reg::RegTensor<float> vreg4;
            AscendC::Reg::RegTensor<float> vreg5;
            AscendC::Reg::RegTensor<float> vreg6_1;
            AscendC::Reg::RegTensor<float> vreg6_2;
            AscendC::Reg::RegTensor<float> vreg7_1;
            AscendC::Reg::RegTensor<float> vreg7_2;

            AscendC::Reg::RegTensor<float> vreg8_1;
            AscendC::Reg::RegTensor<float> vreg8_2;

            AscendC::Reg::RegTensor<float> vreg9_1;
            AscendC::Reg::RegTensor<float> vreg9_2;

            AscendC::Reg::MaskReg preg0;
            AscendC::Reg::MaskReg preg1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

            if constexpr (isSymmertrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_1, maxLocalAddr);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_2, minLocalAddr);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_1, scaleLocalAddr);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(multiRow); i++) {
                uint32_t sreg0 = tilingData_.rowLen;
                if constexpr (isSymmertrical == false) {
                    AscendC::Reg::Duplicate(vreg7_1, MIN_FLOAT_VALUE, preg1);
                    AscendC::Reg::Duplicate(vreg7_2, MAX_FLOAT_VALUE, preg1);
                } else {
                    AscendC::Reg::Duplicate(vreg7_1, MIN_FLOAT_VALUE, preg1);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(vfLoop - 1); j++) {
                    preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
                    AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        vreg1, inLocalAddr + i * rowCount + j * VL);
                    AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vreg2, smoothLocalAddr + j * VL);
                        AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::Reg::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    if constexpr (isSymmertrical == false) {
                        AscendC::Reg::Max(vreg7_1, vreg3, vreg7_1, preg0);
                        AscendC::Reg::Min(vreg7_2, vreg3, vreg7_2, preg0);
                    } else {
                        AscendC::Reg::Abs(vreg6_1, vreg3, preg0);
                        AscendC::Reg::Muls(vreg6_1, vreg6_1, float(1.0) / maxValue, preg0);
                        AscendC::Reg::Max(vreg7_1, vreg6_1, vreg7_1, preg0);
                    }
                }
                if constexpr (isSymmertrical == false) {
                    AscendC::Reg::ReduceMax(vreg8_1, vreg7_1, preg1);
                    AscendC::Reg::ReduceMin<Reg::ReduceType::MIN>(vreg8_2, vreg7_2, preg1);
                } else {
                    AscendC::Reg::ReduceMax<Reg::ReduceType::MAX>(vreg8_1, vreg7_1, preg1);
                }
                for (uint16_t j = static_cast<uint16_t>(vfLoop - 1); j < vfLoop; j++) {
                    preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
                    AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        vreg1, inLocalAddr + i * rowCount + j * VL);
                    AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vreg2, smoothLocalAddr + j * VL);
                        AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::Reg::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    if constexpr (isSymmertrical == false) {
                        AscendC::Reg::ReduceMax(vreg7_1, vreg3, preg0);
                        AscendC::Reg::ReduceMin(vreg7_2, vreg3, preg0);
                        AscendC::Reg::Max(vreg8_1, vreg8_1, vreg7_1, preg0);
                        AscendC::Reg::Min(vreg8_2, vreg8_2, vreg7_2, preg0);
                    } else {
                        AscendC::Reg::Abs(vreg6_1, vreg3, preg0);
                        AscendC::Reg::Muls(vreg6_1, vreg6_1, float(1.0) / maxValue, preg0);
                        AscendC::Reg::ReduceMax(vreg7_1, vreg6_1, preg0);
                        AscendC::Reg::Max(vreg8_1, vreg8_1, vreg7_1, preg0);
                    }
                }
                if constexpr (isSymmertrical == false) {
                    AscendC::Reg::Max(vreg9_1, vreg9_1, vreg8_1, preg1);
                    AscendC::Reg::Min(vreg9_2, vreg9_2, vreg8_2, preg1);
                } else {
                    AscendC::Reg::Max(vreg9_1, vreg9_1, vreg8_1, preg1);
                }
            }
            if constexpr (isSymmertrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vreg9_1,
                                                                                               preg0);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vreg9_2,
                                                                                               preg0);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg9_1,
                                                                                               preg0);
            }
        }
    }
    __aicore__ inline void ComputeMaxColScaleVF(__local_mem__ float* scaleLocalAddr,
                                                __local_mem__ float* scaleOutLocalAddr,
                                                __local_mem__ float* maxLocalAddr, __local_mem__ float* maxOutLocalAddr,
                                                __local_mem__ float* minLocalAddr, __local_mem__ float* minOutLocalAddr,
                                                uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg0_1;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg0_2;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg1_1;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg1_2;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg2_1;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg2_2;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg3_1;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg3_2;

            if constexpr (isSymmertrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_1, maxOutLocalAddr);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_2, minOutLocalAddr);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_1, scaleOutLocalAddr);
            }

            AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++) {
                maskNum = elementNum - i * VL;
                mask = AscendC::Reg::UpdateMask<float>(maskNum);

                if constexpr (isSymmertrical == false) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                    AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                    AscendC::Reg::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                    AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                }
            }
            if constexpr (isSymmertrical == false) {
                AscendC::Reg::ReduceMax<float>(vreg2_1, vreg1_1, mask);
                AscendC::Reg::ReduceMin<float>(vreg2_2, vreg1_2, mask);
            } else {
                AscendC::Reg::ReduceMax<float>(vreg2_1, vreg1_1, mask);
            }

            for (uint16_t i = vfLoopNum - 1; i < vfLoopNum; i++) {
                maskNum = elementNum - i * VL;
                mask = AscendC::Reg::UpdateMask<float>(maskNum);
                if constexpr (isSymmertrical == false) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                    AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::Reg::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                    AscendC::Reg::Max(vreg3_1, vreg2_1, vreg3_1, mask);

                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                    AscendC::Reg::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                    AscendC::Reg::ReduceMin<float>(vreg3_2, vreg1_2, mask);
                    AscendC::Reg::Min(vreg3_2, vreg2_2, vreg3_2, mask);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                    AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::Reg::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                    AscendC::Reg::Max(vreg3_1, vreg2_1, vreg3_1, mask);
                }
            }
            if constexpr (isSymmertrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxOutLocalAddr, vreg3_1,
                                                                                               mask);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(minOutLocalAddr, vreg3_2,
                                                                                               mask);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr,
                                                                                               vreg3_1, mask);
            }
        }
    }

    __aicore__ inline void ComputeScaleSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr,
                                             __local_mem__ float* scaleLocalAddr, uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg0;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg1;

            AscendC::Reg::MaskReg mask;

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                maskNum = elementNum - i * VL;
                mask = AscendC::Reg::UpdateMask<float>(maskNum);

                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, minLocalAddr + i * VL);
                AscendC::Reg::Sub(vreg1, vreg0, vreg1, mask);
                AscendC::Reg::Muls(vreg1, vreg1, float(1.0) / maxValueNoSym, mask);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg1,
                                                                                               mask);
            }
        }
    }

    __aicore__ inline void ComputeOffsetSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr,
                                              __local_mem__ float* offsetLocalAddr, uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__
        {
            static constexpr AscendC::Reg::DivSpecificMode mode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg0;
            AscendC::Reg::RegTensor<float, Reg::RegTraitNumOne> vreg1;

            AscendC::Reg::MaskReg mask;

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                maskNum = elementNum - i * VL;
                mask = AscendC::Reg::UpdateMask<float>(maskNum);

                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, scaleLocalAddr + i * VL);
                AscendC::Reg::Div<float, &mode>(vreg1, vreg0, vreg1, mask);
                AscendC::Reg::Muls(vreg1, vreg1, float(-1.0), mask);
                AscendC::Reg::Adds(vreg1, vreg1, maxValue, mask);

                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(offsetLocalAddr, vreg1,
                                                                                               mask);
            }
        }
    }

    __aicore__ inline void ComputeMaxRowScale(int32_t multiRow)
    {
        LocalTensor<xDtype> inLocal = inQueue.template DeQue<xDtype>();
        __local_mem__ xDtype* inLocalAddr = (__local_mem__ xDtype*)inLocal.GetPhyAddr();
        LocalTensor<xDtype> smoothLocal;
        __local_mem__ xDtype* smoothLocalAddr;

        LocalTensor<float> maxToWorkSpaceLocal;
        __local_mem__ float* maxToWorkSpaceLocalAddr;
        LocalTensor<float> minToWorkSpaceLocal;
        __local_mem__ float* minToWorkSpaceLocalAddr;
        LocalTensor<float> scaleToWorkSpaceLocal;
        __local_mem__ float* scaleToWorkSpaceLocalAddr;

        if constexpr (hasSmooth == 1) {
            smoothLocal = smoothQueue.template DeQue<xDtype>();
            smoothLocalAddr = (__local_mem__ xDtype*)smoothLocal.GetPhyAddr();
        }
        if constexpr (isSymmertrical == false) {
            maxToWorkSpaceLocal = MaxToWorkSpaceQueue.template DeQue<float>();
            maxToWorkSpaceLocalAddr = (__local_mem__ float*)maxToWorkSpaceLocal.GetPhyAddr();

            minToWorkSpaceLocal = MinToWorkSpaceQueue.template DeQue<float>();
            minToWorkSpaceLocalAddr = (__local_mem__ float*)minToWorkSpaceLocal.GetPhyAddr();
        } else {
            scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.template DeQue<float>();
            scaleToWorkSpaceLocalAddr = (__local_mem__ float*)scaleToWorkSpaceLocal.GetPhyAddr();
        }

        ComputeMaxRowScaleVF(inLocalAddr, smoothLocalAddr, scaleToWorkSpaceLocalAddr, maxToWorkSpaceLocalAddr,
                             minToWorkSpaceLocalAddr, multiRow);
        if constexpr (hasSmooth == 1) {
            smoothQueue.FreeTensor(smoothLocal);
        }
        if constexpr (isSymmertrical == false) {
            MaxToWorkSpaceQueue.template EnQue(maxToWorkSpaceLocal);
            MinToWorkSpaceQueue.template EnQue(minToWorkSpaceLocal);
        } else {
            scaleToWorkSpaceQueue.template EnQue(scaleToWorkSpaceLocal);
        }
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeMaxColScale(uint32_t elementNum, __local_mem__ float* maxAddr,
                                              __local_mem__ float* minAddr, __local_mem__ float* scaleAddr,
                                              __local_mem__ float* offsetAddr)
    {
        LocalTensor<float> scaleFromWorkSpaceLocal;
        __local_mem__ float* scaleFromWorkSpaceLocalAddr;

        LocalTensor<float> maxFromWorkSpaceLocal;
        __local_mem__ float* maxFromWorkSpaceLocalAddr;

        LocalTensor<float> minFromWorkSpaceLocal;
        __local_mem__ float* minFromWorkSpaceLocalAddr;

        if constexpr (isSymmertrical == false) {
            maxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.template DeQue<float>();
            maxFromWorkSpaceLocalAddr = (__local_mem__ float*)maxFromWorkSpaceLocal.GetPhyAddr();

            minFromWorkSpaceLocal = MinFromWorkSpaceQueue.template DeQue<float>();
            minFromWorkSpaceLocalAddr = (__local_mem__ float*)minFromWorkSpaceLocal.GetPhyAddr();

            ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr, scaleAddr, maxFromWorkSpaceLocalAddr, maxAddr,
                                 minFromWorkSpaceLocalAddr, minAddr, elementNum);
            ComputeScaleSymVF(maxAddr, minAddr, scaleAddr, 1);
            ComputeOffsetSymVF(maxAddr, scaleAddr, offsetAddr, 1);

            MaxFromWorkSpaceQueue.FreeTensor(maxFromWorkSpaceLocal);
            MinFromWorkSpaceQueue.FreeTensor(minFromWorkSpaceLocal);
        } else {
            scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.template DeQue<float>();
            scaleFromWorkSpaceLocalAddr = (__local_mem__ float*)scaleFromWorkSpaceLocal.GetPhyAddr();

            ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr, scaleAddr, maxFromWorkSpaceLocalAddr, maxAddr,
                                 minFromWorkSpaceLocalAddr, minAddr, elementNum);
            scaleFromWorkSpaceQueue.FreeTensor(scaleFromWorkSpaceLocal);
        }
    }

    __aicore__ inline void ComputeY(int32_t multiRow, __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr)
    {
        uint32_t index = 0;
        LocalTensor<yDtype> yLocal = outQueue.template AllocTensor<yDtype>();
        LocalTensor<xDtype> xLocal = inQueue.template DeQue<xDtype>();
        LocalTensor<xDtype> smoothLocal;

        __local_mem__ xDtype* xAddr = (__local_mem__ xDtype*)xLocal.GetPhyAddr();
        __local_mem__ yDtype* yAddr = (__local_mem__ yDtype*)yLocal.GetPhyAddr();
        __local_mem__ xDtype* smoothAddr;

        if constexpr (hasSmooth) {
            smoothLocal = smoothQueue.template DeQue<xDtype>();
            smoothAddr = (__local_mem__ xDtype*)smoothLocal.GetPhyAddr();
        }

        ComputeYVF(xAddr, smoothAddr, yAddr, scaleAddr, offsetAddr, multiRow);

        if constexpr (IsSameType<yDtype, int8_t>::value) {
            auto halfLocal = halfOutBuf_.Get<half>();
            PipeBarrier<PIPE_V>();
            Cast(yLocal, halfLocal, RoundMode::CAST_TRUNC, outLen);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (hasSmooth == 1) {
            smoothQueue.FreeTensor(smoothLocal);
        }
        inQueue.FreeTensor(xLocal);
        outQueue.template EnQue<yDtype>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t multiRow, int32_t loopCount)
    {
        LocalTensor<yDtype> yLocal = outQueue.template DeQue<yDtype>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(multiRow),
                                     static_cast<uint32_t>(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
        DataCopyPad(outGm[loopCount * lenGMMultiRow], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeYVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                      __local_mem__ yDtype* yAddr, __local_mem__ float* scaleAddr,
                                      __local_mem__ float* offsetAddr, int32_t multiRow)
    {
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint32_t rowCount = sizeHalfLen;
        uint16_t vfLoop = (rowCount + VL - 1) / VL;

        __local_mem__ half* halfAddr = nullptr;
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            auto halfLocal = halfOutBuf_.Get<half>();
            halfAddr = (__local_mem__ half*)halfLocal.GetPhyAddr();
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> vreg_scale;
            AscendC::Reg::RegTensor<float> vreg_offset;

            AscendC::Reg::RegTensor<xDtype> vreg1;
            AscendC::Reg::RegTensor<xDtype> vreg2;
            AscendC::Reg::RegTensor<float> vreg3;
            AscendC::Reg::RegTensor<float> vreg4;
            AscendC::Reg::RegTensor<float> vreg5;
            AscendC::Reg::RegTensor<int16_t> vreg6;
            AscendC::Reg::RegTensor<half> vreg7;
            AscendC::Reg::RegTensor<yDtype> vreg8;

            AscendC::Reg::MaskReg preg0;
            AscendC::Reg::MaskReg preg1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::H>();

            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_scale, scaleAddr);
            if constexpr (isSymmertrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_offset, offsetAddr);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(multiRow); i++) {
                uint32_t sreg0 = rowCount;
                for (uint16_t j = 0; j < vfLoop; j++) {
                    auto addr = yAddr + i * outAlignLen + j * VL;
                    preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
                    AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        vreg1, xAddr + i * rowCount + j * VL);
                    AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::Reg::DataCopy<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg2,
                                                                                                smoothAddr + j * VL);
                        AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::Reg::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    AscendC::Reg::Div(vreg5, vreg3, vreg_scale, preg0);
                    if constexpr (isSymmertrical == false) {
                        AscendC::Reg::Add(vreg5, vreg5, vreg_offset, preg0);
                    }
                    if constexpr (IsSameType<yDtype, int8_t>::value) {
                        AscendC::Reg::Cast<int16_t, float, castTraitF32ToI16>(vreg6, vreg5, preg0);
                        AscendC::Reg::Cast<half, int16_t, castTraitI16ToF16>(vreg7, vreg6, preg0);
                        AscendC::Reg::DataCopy<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            halfAddr + i * outAlignLen + j * VL, vreg7, preg0);
                    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value ||
                                         IsSameType<yDtype, fp8_e5m2_t>::value) {
                        AscendC::Reg::Cast<yDtype, float, castTraitF32tofp8>(vreg8, vreg5, preg0);
                        AscendC::Reg::DataCopy<yDtype, AscendC::Reg::StoreDist::DIST_PACK_B16>(addr, vreg8, preg0);
                    }
                }
            }
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            maxValue = INT8_MAX_VALUE;
            maxValueNoSym = INT8_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
            maxValue = FP8_E5M2_MAX_VALUE;
            maxValueNoSym = FP8_E5M2_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
            maxValue = FP8_E4M3FN_MAX_VALUE;
            maxValueNoSym = FP8_E4M3FN_MAX_VALUE_NO_SYM;
        }
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, useBufferNum> inQueue;
    TQue<QuePosition::VECIN, useBufferNum> smoothQueue;
    TQue<QuePosition::VECOUT, useBufferNum> scaleToWorkSpaceQueue;
    TQue<QuePosition::VECIN, useBufferNum> scaleFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, useBufferNum> MaxToWorkSpaceQueue;
    TQue<QuePosition::VECIN, useBufferNum> MaxFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, useBufferNum> MinToWorkSpaceQueue;
    TQue<QuePosition::VECIN, useBufferNum> MinFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, useBufferNum> MaxOutQueue;
    TQue<QuePosition::VECOUT, useBufferNum> MinOutQueue;
    TQue<QuePosition::VECOUT, useBufferNum> scaleOutQueue;
    TQue<QuePosition::VECOUT, useBufferNum> offsetQueue;
    TQue<QuePosition::VECOUT, useBufferNum> outQueue;
    TBuf<QuePosition::VECCALC> halfOutBuf_;

    /* global memory address */
    GlobalTensor<xDtype> inGm;
    GlobalTensor<xDtype> smoothGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<float> workspaceTmp1;
    GlobalTensor<float> workspaceTmp2;

    int64_t offsetBase = 0;
    int64_t srcOffset = 0;
    int64_t scaleOffset = 0;
    float maxValue = 0.0;
    float maxValueNoSym = 0.0;

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
} // namespace DynamicQuantNDPerTensorOpt
#endif // DYNAMIC_QUANT_REGBASE_FULL_LOAD_H
