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
 * \file dynamic_quant_regbase_full_load_pertensor.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_FULL_LOAD_PERTENSOR_H
#define DYNAMIC_QUANT_REGBASE_FULL_LOAD_PERTENSOR_H
#include "dynamic_quant_regbase_base.h"
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
private:
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;

public:
    __aicore__ inline DynamicQuantRegbaseFullLoadPertensor(TPipe* pipe) {
        pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData) {
        DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
        ParseTilingData(tilingData);
        InitParams(offset);
        InitAndSetBuffer(x, smooth_scales, y, scale, offset,workSpace);
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

        if constexpr (isSymmertrical == false)
        {
            MaxToWorkSpaceLocal = MaxToWorkSpaceQueue.template AllocTensor<float>();
            AscendC::Duplicate(MaxToWorkSpaceLocal, MIN_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MaxToWorkSpaceQueue.template EnQue(MaxToWorkSpaceLocal);

            MinToWorkSpaceLocal = MinToWorkSpaceQueue.template AllocTensor<float>();
            AscendC::Duplicate(MinToWorkSpaceLocal, MAX_FLOAT_VALUE, SIXTY_FOUR, ONE, ONE, EIGHT);
            MinToWorkSpaceQueue.template EnQue(MinToWorkSpaceLocal);
        }
        else
        {
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

        if constexpr (isSymmertrical == false)
        {
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
        }
        else
        {
            scaleOutLocal = scaleOutQueue.template AllocTensor<float>();
            AscendC::Duplicate(scaleOutLocal, (float)0.0, SIXTY_FOUR, ONE, ONE, EIGHT);
            scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
        }
        scaleOffset = 0;
        CopyInScaleByEle(scaleOffset, tilingData_.coreNum);
        ComputeMaxColScale(tilingData_.coreNum,MaxOutLocalAddr,MinOutLocalAddr,scaleOutLocalAddr,offsetLocalAddr);
        if constexpr (isSymmertrical == false)
        {
            MaxOutQueue.FreeTensor(MaxOutLocal);
            MinOutQueue.FreeTensor(MinOutLocal);
            scaleOutQueue.template EnQue(scaleOutLocal);
            offsetQueue.template EnQue(offsetLocal);
        }
        else
        {
            scaleOutQueue.template EnQue(scaleOutLocal);
        }
    }

    __aicore__ inline void ProcessScale()
    {
        ProcessScaleRow();
        SyncAll();
        ProcessScaleCol();
    }

    __aicore__ inline void ProcessYRow(int32_t multiRow, int32_t loopNum,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr)
    {
        CopyIn(multiRow, loopNum);
        ComputeY(multiRow,scaleAddr,offsetAddr);
        CopyOut(multiRow, loopNum);
    }

    __aicore__ inline void ProcessY()
    {
        LocalTensor<float> scaleOutLocal = scaleOutQueue.template DeQue<float>();
        __local_mem__ float* scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
        LocalTensor<float> offsetLocal;
        __local_mem__ float* offsetLocalAddr;

        if constexpr (isSymmertrical == false)
        {
            offsetLocal = offsetQueue.template DeQue<float>();
            offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
        }

        for (int32_t i = 0; i < loopCnt; i++) {
            ProcessYRow(multiRowNum, i,scaleOutLocalAddr,offsetLocalAddr);
        }
        if (remainRow > 0) {
            ProcessYRow(remainRow, loopCnt,scaleOutLocalAddr,offsetLocalAddr);
        }
        if(blockIdx == 0)
        {
            DataCopyParams copyParams{1, (uint16_t)(sizeof(float)), 0, 0};
            DataCopyPad(scaleGm[0], scaleOutLocal, copyParams);
            if constexpr (isSymmertrical == false)
            {
                DataCopyPad(offsetGm[0], offsetLocal, copyParams);
            }
        }
        scaleOutQueue.FreeTensor(scaleOutLocal);
        if constexpr (isSymmertrical == false)
        {
            offsetQueue.FreeTensor(offsetLocal);
        }
    }

    __aicore__ inline void Process() {
        if (blockIdx >= tilingData_.coreNum) {
            return;
        }
        ProcessScale();
        ProcessY();
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,GM_ADDR workSpace) {
        if constexpr (hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
            pPipe->InitBuffer(smoothQueue, useBufferNum, sizeHalfLen * sizeof(xDtype));
        }

        if (blockIdx < tilingData_.headCoreNum) {
            inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
            outGm.SetGlobalBuffer((__gm__ yCopyDtype*)y + blockIdx * outLenHead, outLenHead);
        } else {
            inGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
            outGm.SetGlobalBuffer(
                (__gm__ yCopyDtype*)y + tilingData_.headCoreNum * outLenHead + (blockIdx - tilingData_.headCoreNum) * outLenTail,
                outLenTail);
        }

        if( blockIdx == 0 )
        {
            scaleGm.SetGlobalBuffer((__gm__ float*)scale,1);
        }

        if constexpr (isSymmertrical == false)
        {
            workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace),tilingData_.coreNum);
            workspaceTmp2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace + tilingData_.coreNum*sizeof(float)),tilingData_.coreNum);
        }
        else
        {
            workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace),tilingData_.coreNum);
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
        pPipe->InitBuffer(outQueue, useBufferNum, outLen * sizeof(yCopyDtype));
        pPipe->InitBuffer(scaleOutQueue, useBufferNum, TWO_FIVE_SIX);
    }

    __aicore__ inline void CopyIn(int32_t multiRow, int32_t loopNum) {
        LocalTensor<xDtype> inLocal = inQueue.template AllocTensor<xDtype>();
        DataCopyExtParams copyParams = {static_cast<uint16_t>(multiRow),
                                        static_cast<uint32_t>(tilingData_.rowLen * sizeof(xDtype)),
                                        0, 0, 0};
        DataCopyPadExtParams<xDtype> padParams{true, 0, rightPadding, 0};
        DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
        inQueue.template EnQue(inLocal);

        if constexpr (hasSmooth) {
            LocalTensor<xDtype> smoothLocal = smoothQueue.template AllocTensor<xDtype>();
            DataCopyExtParams smoothCopyParams = {1,
                                                  static_cast<uint32_t>(tilingData_.rowLen * sizeof(xDtype)),
                                                  0, 0, 0};
            DataCopyPad(smoothLocal, smoothGm, smoothCopyParams, padParams);
            smoothQueue.template EnQue(smoothLocal);
        }
    }

    __aicore__ inline void CopyInScaleByEle(int64_t offset,uint32_t elementNum) {
        DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(float)), 0, 0};
        if constexpr (isSymmertrical == false)
        {
            LocalTensor<float> MaxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(MaxFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            MaxFromWorkSpaceQueue.template EnQue(MaxFromWorkSpaceLocal);

            LocalTensor<float> MinFromWorkSpaceLocal = MinFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(MinFromWorkSpaceLocal, workspaceTmp2[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            MinFromWorkSpaceQueue.template EnQue(MinFromWorkSpaceLocal);
        }
        else
        {
            LocalTensor<float> scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.template AllocTensor<float>();
            DataCopyPad(scaleFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            scaleFromWorkSpaceQueue.template EnQue(scaleFromWorkSpaceLocal);
        }
    }
    __aicore__ inline void CopyUB2Workspace(int64_t size) {
        if constexpr (isSymmertrical == false)
        {
            auto tmp1 = MaxToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp1[blockIdx],
                      tmp1,
                      {1, (uint16_t)(size * sizeof(float)), 0, 0});
            MaxToWorkSpaceQueue.FreeTensor(tmp1);

            auto tmp2 = MinToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp2[blockIdx],
              tmp2,
              {1, (uint16_t)(size * sizeof(float)), 0, 0});
            MinToWorkSpaceQueue.FreeTensor(tmp2);
        }
        else
        {
            auto tmp = scaleToWorkSpaceQueue.template DeQue<float>();
            DataCopyPad(workspaceTmp1[blockIdx],
                      tmp,
                      {1, (uint16_t)(size * sizeof(float)), 0, 0});
            scaleToWorkSpaceQueue.FreeTensor(tmp);
        }
    }

    __aicore__ inline void ComputeMaxRowScaleVF(__local_mem__ xDtype* inLocalAddr, __local_mem__ xDtype* smoothLocalAddr,
                                                                                              __local_mem__ float* scaleLocalAddr, __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, uint32_t multiRow)
    {
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint32_t rowCount = sizeHalfLen;
        uint16_t vfLoop = (tilingData_.rowLen + VL - 1) / VL;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<xDtype> vreg1;
            AscendC::MicroAPI::RegTensor<xDtype> vreg2;
            AscendC::MicroAPI::RegTensor<float> vreg3;
            AscendC::MicroAPI::RegTensor<float> vreg4;
            AscendC::MicroAPI::RegTensor<float> vreg5;
            AscendC::MicroAPI::RegTensor<float> vreg6_1;
            AscendC::MicroAPI::RegTensor<float> vreg6_2;
            AscendC::MicroAPI::RegTensor<float> vreg7_1;
            AscendC::MicroAPI::RegTensor<float> vreg7_2;

            AscendC::MicroAPI::RegTensor<float> vreg8_1;
            AscendC::MicroAPI::RegTensor<float> vreg8_2;

            AscendC::MicroAPI::RegTensor<float> vreg9_1;
            AscendC::MicroAPI::RegTensor<float> vreg9_2;

            AscendC::MicroAPI::MaskReg preg0;
            AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg9_1, maxLocalAddr);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg9_2, minLocalAddr);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg9_1, scaleLocalAddr);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(multiRow); i++) {
                uint32_t sreg0 = tilingData_.rowLen;
                if constexpr (isSymmertrical == false)
                {
                    AscendC::MicroAPI::Duplicate(vreg7_1, MIN_FLOAT_VALUE, preg1);
                    AscendC::MicroAPI::Duplicate(vreg7_2, MAX_FLOAT_VALUE, preg1);
                }
                else
                {
                    AscendC::MicroAPI::Duplicate(vreg7_1, MIN_FLOAT_VALUE, preg1);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(vfLoop-1); j++) {
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, inLocalAddr + i * rowCount + j * VL);
                    AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + j * VL);
                        AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    if constexpr (isSymmertrical == false)
                    {
                        AscendC::MicroAPI::Max(vreg7_1, vreg3, vreg7_1, preg0);
                        AscendC::MicroAPI::Min(vreg7_2, vreg3, vreg7_2, preg0);
                    }
                    else
                    {
                        AscendC::MicroAPI::Abs(vreg6_1, vreg3, preg0);
                        AscendC::MicroAPI::Muls(vreg6_1, vreg6_1, float(1.0)/maxValue, preg0);
                        AscendC::MicroAPI::Max(vreg7_1, vreg6_1, vreg7_1, preg0);
                    }
                }
                if constexpr (isSymmertrical == false)
                {
                    AscendC::MicroAPI::ReduceMax(vreg8_1, vreg7_1, preg1);
                    AscendC::MicroAPI::ReduceMin(vreg8_2, vreg7_2, preg1);
                }
                else
                {
                    AscendC::MicroAPI::ReduceMax(vreg8_1, vreg7_1, preg1);
                }
                for (uint16_t j = static_cast<uint16_t>(vfLoop-1); j < vfLoop; j++) {
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, inLocalAddr + i * rowCount + j * VL);
                    AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + j * VL);
                        AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    if constexpr (isSymmertrical == false)
                    {
                        AscendC::MicroAPI::ReduceMax(vreg7_1, vreg3, preg0);
                        AscendC::MicroAPI::ReduceMin(vreg7_2, vreg3, preg0);
                        AscendC::MicroAPI::Max(vreg8_1, vreg8_1, vreg7_1, preg0);
                        AscendC::MicroAPI::Min(vreg8_2, vreg8_2, vreg7_2, preg0);
                    }
                    else
                    {
                        AscendC::MicroAPI::Abs(vreg6_1, vreg3, preg0);
                        AscendC::MicroAPI::Muls(vreg6_1, vreg6_1, float(1.0)/maxValue, preg0);
                        AscendC::MicroAPI::ReduceMax(vreg7_1, vreg6_1, preg0);
                        AscendC::MicroAPI::Max(vreg8_1, vreg8_1, vreg7_1, preg0);
                    }
                }
                if constexpr (isSymmertrical == false)
                {
                    AscendC::MicroAPI::Max(vreg9_1, vreg9_1, vreg8_1, preg1);
                    AscendC::MicroAPI::Min(vreg9_2, vreg9_2, vreg8_2, preg1);
                }
                else
                {
                    AscendC::MicroAPI::Max(vreg9_1, vreg9_1, vreg8_1, preg1);
                }
            }
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vreg9_1, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vreg9_2, preg0);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg9_1, preg0);
            }
        }
    }
    __aicore__ inline void ComputeMaxColScaleVF(__local_mem__ float* scaleLocalAddr,__local_mem__ float* scaleOutLocalAddr,__local_mem__ float* maxLocalAddr,__local_mem__ float* maxOutLocalAddr,__local_mem__ float* minLocalAddr,__local_mem__ float* minOutLocalAddr,uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__ {
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0_1;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0_2;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1_1;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1_2;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg2_1;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg2_2;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3_1;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3_2;

            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_1, maxOutLocalAddr);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_2, minOutLocalAddr);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_1, scaleOutLocalAddr);
            }

            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++)
            {
                maskNum = elementNum - i * VL;
                mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

                if constexpr (isSymmertrical == false)
                {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                    AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                    AscendC::MicroAPI::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                }
                else
                {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                    AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                }
            }
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::ReduceMax<float>(vreg2_1, vreg1_1, mask);
                AscendC::MicroAPI::ReduceMin<float>(vreg2_2, vreg1_2, mask);
            }
            else
            {
                AscendC::MicroAPI::ReduceMax<float>(vreg2_1, vreg1_1, mask);
            }

            for (uint16_t i = vfLoopNum - 1; i < vfLoopNum; i++)
            {
                maskNum = elementNum - i * VL;
                mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);
                if constexpr (isSymmertrical == false)
                {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                    AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::MicroAPI::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                    AscendC::MicroAPI::Max(vreg3_1, vreg2_1, vreg3_1, mask);

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                    AscendC::MicroAPI::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                    AscendC::MicroAPI::ReduceMin<float>(vreg3_2, vreg1_2, mask);
                    AscendC::MicroAPI::Min(vreg3_2, vreg2_2, vreg3_2, mask);
                }
                else
                {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                    AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                    AscendC::MicroAPI::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                    AscendC::MicroAPI::Max(vreg3_1, vreg2_1, vreg3_1, mask);
                }
            }
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxOutLocalAddr, vreg3_1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minOutLocalAddr, vreg3_2, mask);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr, vreg3_1, mask);
            }
        }
    }

    __aicore__ inline void ComputeScaleSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, __local_mem__ float* scaleLocalAddr,uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__ {
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;

            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t i = 0; i < vfLoopNum; i++)
            {
                maskNum = elementNum - i * VL;
                mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, minLocalAddr + i * VL);
                AscendC::MicroAPI::Sub(vreg1, vreg0, vreg1, mask);
                AscendC::MicroAPI::Muls(vreg1, vreg1, float(1.0)/maxValueNoSym, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg1, mask);
            }
        }
    }

    __aicore__ inline void ComputeOffsetSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr,uint32_t elementNum)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
        uint32_t maskNum;

        __VEC_SCOPE__ {
            static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0;
            AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;

            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t i = 0; i < vfLoopNum; i++)
            {
                maskNum = elementNum - i * VL;
                mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, scaleLocalAddr + i * VL);
                AscendC::MicroAPI::Div<float,&mode>(vreg1, vreg0, vreg1, mask);
                AscendC::MicroAPI::Muls(vreg1, vreg1, float(-1.0), mask);
                AscendC::MicroAPI::Adds(vreg1, vreg1, maxValue,  mask);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(offsetLocalAddr, vreg1, mask);
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
        if constexpr (isSymmertrical == false)
        {
            maxToWorkSpaceLocal = MaxToWorkSpaceQueue.template DeQue<float>();
            maxToWorkSpaceLocalAddr = (__local_mem__ float*)maxToWorkSpaceLocal.GetPhyAddr();

            minToWorkSpaceLocal = MinToWorkSpaceQueue.template DeQue<float>();
            minToWorkSpaceLocalAddr = (__local_mem__ float*)minToWorkSpaceLocal.GetPhyAddr();
        }
        else
        {
            scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.template DeQue<float>();
            scaleToWorkSpaceLocalAddr = (__local_mem__ float*)scaleToWorkSpaceLocal.GetPhyAddr();
        }

        ComputeMaxRowScaleVF(inLocalAddr, smoothLocalAddr, scaleToWorkSpaceLocalAddr,maxToWorkSpaceLocalAddr,minToWorkSpaceLocalAddr,multiRow);
        if constexpr (hasSmooth == 1) {
            smoothQueue.FreeTensor(smoothLocal);
        }
        if constexpr (isSymmertrical == false)
        {
            MaxToWorkSpaceQueue.template EnQue(maxToWorkSpaceLocal);
            MinToWorkSpaceQueue.template EnQue(minToWorkSpaceLocal);
        }
        else
        {
            scaleToWorkSpaceQueue.template EnQue(scaleToWorkSpaceLocal);
        }
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeMaxColScale(uint32_t elementNum,__local_mem__ float* maxAddr,__local_mem__ float* minAddr,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr)
    {
        LocalTensor<float> scaleFromWorkSpaceLocal;
        __local_mem__ float* scaleFromWorkSpaceLocalAddr;

        LocalTensor<float> maxFromWorkSpaceLocal;
        __local_mem__ float* maxFromWorkSpaceLocalAddr;

        LocalTensor<float> minFromWorkSpaceLocal;
        __local_mem__ float* minFromWorkSpaceLocalAddr;

        if constexpr (isSymmertrical == false)
        {
            maxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.template DeQue<float>();
            maxFromWorkSpaceLocalAddr = (__local_mem__ float*)maxFromWorkSpaceLocal.GetPhyAddr();

            minFromWorkSpaceLocal = MinFromWorkSpaceQueue.template DeQue<float>();
            minFromWorkSpaceLocalAddr = (__local_mem__ float*)minFromWorkSpaceLocal.GetPhyAddr();

            ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr,scaleAddr,maxFromWorkSpaceLocalAddr,maxAddr,minFromWorkSpaceLocalAddr,minAddr, elementNum);
            ComputeScaleSymVF(maxAddr,minAddr,scaleAddr, 1);
            ComputeOffsetSymVF(maxAddr,scaleAddr,offsetAddr,1);

            MaxFromWorkSpaceQueue.FreeTensor(maxFromWorkSpaceLocal);
            MinFromWorkSpaceQueue.FreeTensor(minFromWorkSpaceLocal);
        }
        else
        {
            scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.template DeQue<float>();
            scaleFromWorkSpaceLocalAddr = (__local_mem__ float*)scaleFromWorkSpaceLocal.GetPhyAddr();

            ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr,scaleAddr,maxFromWorkSpaceLocalAddr,maxAddr,minFromWorkSpaceLocalAddr,minAddr, elementNum);
            scaleFromWorkSpaceQueue.FreeTensor(scaleFromWorkSpaceLocal);
        }
    }

    __aicore__ inline void ComputeY(int32_t multiRow,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr) {
        uint32_t index = 0;
        LocalTensor<yCopyDtype> yLocal = outQueue.template AllocTensor<yCopyDtype>();
        LocalTensor<xDtype> xLocal = inQueue.template DeQue<xDtype>();
        LocalTensor<xDtype> smoothLocal;

        __local_mem__ xDtype* xAddr = (__local_mem__ xDtype*)xLocal.GetPhyAddr();
        __local_mem__ yCopyDtype* yAddr = (__local_mem__ yCopyDtype*)yLocal.GetPhyAddr();
        __local_mem__ xDtype* smoothAddr;

        if constexpr (hasSmooth) {
            smoothLocal = smoothQueue.template DeQue<xDtype>();
            smoothAddr = (__local_mem__ xDtype*)smoothLocal.GetPhyAddr();
        }

        ComputeYVF(xAddr, smoothAddr, yAddr, scaleAddr,offsetAddr, multiRow);

        if constexpr (hasSmooth == 1) {
            smoothQueue.FreeTensor(smoothLocal);
        }
        inQueue.FreeTensor(xLocal);
        outQueue.template EnQue<yCopyDtype>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t multiRow, int32_t loopCount) {
        LocalTensor<yCopyDtype> yLocal = outQueue.template DeQue<yCopyDtype>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(multiRow),
                                     static_cast<uint32_t>(tilingData_.rowLen * sizeof(yCopyDtype)),
                                     0, 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyParams.blockLen = copyParams.blockLen >> 1;
            uint32_t index = (loopCount * lenGMMultiRow) / 2;
            DataCopyPad(outGm[index], yLocal, copyParams);
        } else {
            DataCopyPad(outGm[loopCount * lenGMMultiRow], yLocal, copyParams);
        }
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeYVF(__local_mem__ xDtype* xAddr, __local_mem__ xDtype* smoothAddr,
                                     __local_mem__ yCopyDtype* yAddr, __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr,
                                     int32_t multiRow) {
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint32_t rowCount = sizeHalfLen;
        uint16_t vfLoop = (rowCount + VL - 1) / VL;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vreg_scale;
            AscendC::MicroAPI::RegTensor<float> vreg_offset;

            AscendC::MicroAPI::RegTensor<xDtype> vreg1;
            AscendC::MicroAPI::RegTensor<xDtype> vreg2;
            AscendC::MicroAPI::RegTensor<float> vreg3;
            AscendC::MicroAPI::RegTensor<float> vreg4;
            AscendC::MicroAPI::RegTensor<float> vreg5;
            AscendC::MicroAPI::RegTensor<int16_t> vreg6;
            AscendC::MicroAPI::RegTensor<half> vreg7;
            AscendC::MicroAPI::RegTensor<yCopyDtype> vreg8;

            AscendC::MicroAPI::MaskReg preg0;
            AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg_scale, scaleAddr);
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg_offset, offsetAddr);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(multiRow); i++) {
                uint32_t sreg0 = rowCount;
                for (uint16_t j = 0; j < vfLoop; j++) {
                    auto addr = yAddr + i * outAlignLen + j * VL;
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xAddr + i * rowCount + j * VL);
                    AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg3, vreg1, preg0);
                    if constexpr (hasSmooth) {
                        AscendC::MicroAPI::DataCopy<xDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothAddr + j * VL);
                        AscendC::MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vreg4, vreg2, preg0);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, preg0);
                    }
                    AscendC::MicroAPI::Div(vreg5, vreg3, vreg_scale, preg0);
                    if constexpr (isSymmertrical == false)
                    {
                        AscendC::MicroAPI::Add(vreg5, vreg5, vreg_offset, preg0);
                    }
                    if constexpr (IsSameType<yDtype, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vreg6, vreg5, preg0);
                        AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vreg7, vreg6, preg0);
                        AscendC::MicroAPI::Cast<yDtype, half, castTraitF16ToI8>(vreg8, vreg7, preg0);
                    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<yDtype, float, castTraitF32toh8>(vreg8, vreg5, preg0);
                    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value) {
                        AscendC::MicroAPI::Cast<yDtype, float, castTraitF32tofp8>(vreg8, vreg5, preg0);
                    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        AscendC::MicroAPI::RegTensor<uint16_t> vreg9;
                        AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vreg6, vreg5, preg0);
                        AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vreg7, vreg6, preg0);
                        AscendC::MicroAPI::Pack(vreg9, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg7);
                        AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16ToI8>(
                            (AscendC::MicroAPI::RegTensor<int4x2_t>&)vreg8,
                            (AscendC::MicroAPI::RegTensor<half>&)vreg9, preg0);
                        addr = yAddr + (i * outAlignLen + j * VL) / 2;
                    }
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, preg1);
                    } else {
                        AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, preg0);
                    }
                }
            }
        }
    }

    __aicore__ inline void SetMaxValue() {
        if constexpr (IsSameType<yDtype, int8_t>::value) {
            maxValue = INT8_MAX_VALUE;
            maxValueNoSym= INT8_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
            maxValue = INT4_MAX_VALUE;
            maxValueNoSym = INT4_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
            maxValue = FP8_E5M2_MAX_VALUE;
            maxValueNoSym = FP8_E5M2_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
            maxValue = FP8_E4M3FN_MAX_VALUE;
            maxValueNoSym = FP8_E4M3FN_MAX_VALUE_NO_SYM;
        } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
            maxValue = HIFLOAT8_MAX_VALUE;
            maxValueNoSym = HIFLOAT8_MAX_VALUE_NO_SYM;
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

    /* global memory address */
    GlobalTensor<xDtype> inGm;
    GlobalTensor<xDtype> smoothGm;
    GlobalTensor<yCopyDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<float> workspaceTmp1;
    GlobalTensor<float> workspaceTmp2;

    int64_t offsetBase = 0;
    int64_t srcOffset = 0;
    int64_t scaleOffset = 0;
    float maxValue = 0.0;
    float maxValueNoSym = 0.0;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN
    };
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToI16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT
    };
    constexpr static AscendC::MicroAPI::CastTrait castTraitI16ToF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND
    };
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC
    };
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};
}  // namespace DynamicQuantNDOpt
#endif  // DYNAMIC_QUANT_REGBASE_FULL_LOAD_H
