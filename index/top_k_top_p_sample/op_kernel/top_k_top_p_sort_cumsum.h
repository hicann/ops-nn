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
 * \file top_k_top_p_sort_cumsum.h
 * \brief
 */

#ifndef TOP_K_TOP_P_SORT_CUMSUM_H
#define TOP_K_TOP_P_SORT_CUMSUM_H

#include "top_k_top_p_comm.h"

namespace TopKTopPSample {
// Sub queue length as depletion merge-sort unit in single cut_s segmentation in MrgSort4Que     
constexpr uint32_t MRG_SORT_INNER_LEN = 256;    //  MRG_SORT_INNER_LEN * 16 ≤ 8192
constexpr uint32_t MRG_PER_MAX_NUM = 4;         // queue amount involved in single merge group 
constexpr uint32_t REPEAT_MAX = 512;            

constexpr uint32_t S_PART_MIN_LEN = 1024;  
constexpr uint32_t MRG_FIRST_MAX_NUM = 1024; 
constexpr uint32_t GM_COPY_PER_DATA_LEN = 32768;
constexpr uint32_t GM_COPY_PER_FLOAT_MAX = GM_COPY_PER_DATA_LEN / 4;
constexpr uint32_t GM_COPY_PER_FLOAT_SORT_MAX = GM_COPY_PER_DATA_LEN / 8;

template <typename T>
class TopKTopPSampleSortKernel
{
public:
    using MrgSrcGMList = GlobalTensor<float> (&)[MRG_PER_MAX_NUM];
    __aicore__ inline TopKTopPSampleSortKernel(){};

    __aicore__ inline void SumErreyOne(
        LocalTensor<float>& src, uint32_t& count, float* sumVal, float topp, uint32_t& topPNum, bool& ifRet)
    {
        for (uint32_t index = 0; index < count; index++) {
            *sumVal += src.GetValue(index);
            if (*sumVal > topp) {
                topPNum = index + 1;
                ifRet = false;
                break;
            }
        }
    }

    __aicore__ inline void SortOneTime(
        LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex, uint32_t dataLen, LocalTensor<float>& sortedLocal,
        LocalTensor<float>& sortTemp)
    {
        LocalTensor<float> concatLocal;
        Concat(concatLocal, srcData, sortTemp, dataLen / SIXTEEN);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(
            sortedLocal, concatLocal, srcIndex.ReinterpretCast<uint32_t>(), sortTemp, dataLen / THIRTY_TWO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortSPartAll(
        TOPKPParams& params, LocalTensor<float>& bufLocal, const GlobalTensor<float>& srcGlobal,
        const GlobalTensor<float>& destGlobal)
    {
        uint32_t innerCopyLen = S_PART_MIN_LEN;
        uint32_t countLen = S_PART_MIN_LEN;
        uint32_t indexOffset = 0;
        uint32_t inStartIndex = params.rowId * params.rowLen;
        uint32_t outStartIndex = inStartIndex * MRG_PER_ELE;
        LocalTensor<float> localValueCast = bufLocal;
        LocalTensor<uint32_t> localIndex = bufLocal[S_PART_MIN_LEN].template ReinterpretCast<uint32_t>();
        LocalTensor<float> sortReslut = bufLocal[S_PART_MIN_LEN * NUM_TWO];
        LocalTensor<float> sortTemp = bufLocal[S_PART_MIN_LEN * NUM_FOUR];
        TEventID eventIDMTE2ToV = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        TEventID eventIDMTE2ToS = GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>();
        TEventID eventIDVToMTE3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        TEventID eventIDSToV = GetTPipePtr()->FetchEventID<HardEvent::S_V>();
        TEventID eventIDMTE3ToMTE2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        for (int32_t innerLoopCount = 0; innerLoopCount < params.eightKPartNum; innerLoopCount++) {
            uint32_t gmOffset = inStartIndex + innerLoopCount * innerCopyLen;
            uint32_t gmOutOffset = outStartIndex + innerLoopCount * innerCopyLen * MRG_PER_ELE;
            if (params.eightKPartTail > 0 && innerLoopCount == params.eightKPartNum - 1) {
                innerCopyLen = params.eightKPartTailPad;
                countLen = params.eightKPartTail;
            }
            if (innerLoopCount > 0) {
                indexOffset += S_PART_MIN_LEN;
            }
            auto srcPos = srcGlobal[gmOffset];
            DataCopy(localValueCast, srcPos, innerCopyLen);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            CreateVecIndex(localIndex[NUM_ZERO].ReinterpretCast<int32_t>(), (int32_t)(indexOffset), innerCopyLen);
            PipeBarrier<PIPE_V>();

            if (innerCopyLen > countLen) {
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                for (uint32_t i = countLen; i < innerCopyLen; i++) {
                    localValueCast.SetValue(i, FLOAT_MIN);
                }
                SetFlag<HardEvent::S_V>(eventIDSToV);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
            }
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            SortOneTime(
                localValueCast, localIndex, innerCopyLen, sortReslut,
                sortTemp); // 分段排好序，localIndex，sortReslut为结果，存在buflocal（buf0）里面了
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            auto destPos = destGlobal[gmOutOffset];
            DataCopyPad(
                destPos, sortReslut, {1, (uint32_t)(countLen * sizeof(float) * MRG_PER_ELE), 0, 0, 0}); // 分段排好的结果存在了destGlobal里面
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }

    __aicore__ inline void MrgSortOneTime(
        LocalTensor<float>& bufLocal, MrgSrcGMList& srcGMList, uint32_t queNum, uint32_t* queLenOri, uint32_t* queIndex,
        uint32_t* currOffset, uint16_t* mrgLen)
    {
        LocalTensor<float> srcSortLocal[MRG_PER_MAX_NUM] = {
            bufLocal, bufLocal[MRG_SORT_INNER_LEN * NUM_TWO], bufLocal[MRG_SORT_INNER_LEN * NUM_FOUR],
            bufLocal[MRG_SORT_INNER_LEN * NUM_SIX]};
        LocalTensor<float> mrgRsLocal = bufLocal[MRG_SORT_INNER_LEN * EIGHT];

        uint16_t valueBits[MRG_PER_MAX_NUM + 1] = {0b11, 0b11, 0b11, 0b111, 0b1111};
        TEventID eventIDMTE2ToV = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();

        uint16_t mrgInnerLen = MRG_SORT_INNER_LEN;
        for (uint32_t i = 0; i < queNum; i++) {
            mrgInnerLen = mrgInnerLen < queLenOri[i] ? mrgInnerLen : queLenOri[i];
        }
        for (uint32_t i = 0; i < queNum; i++) {
            DataCopyPad(
                srcSortLocal[i], srcGMList[queIndex[i]][currOffset[i]],
                {1, static_cast<uint16_t>(sizeof(float) * mrgInnerLen * MRG_PER_ELE), 0, 0}, {false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        uint16_t elementLengths[MRG_PER_MAX_NUM] = {mrgInnerLen, mrgInnerLen, mrgInnerLen, mrgInnerLen};
        MrgSort4Info localParams = {elementLengths, true, valueBits[queNum], 1};
        MrgSort<float>(mrgRsLocal, {srcSortLocal[NUM_ZERO], srcSortLocal[NUM_ONE], srcSortLocal[NUM_TWO], srcSortLocal[NUM_THREE]}, localParams);
        PipeBarrier<PIPE_V>();
        GetMrgSortResult(mrgLen[NUM_ZERO], mrgLen[NUM_ONE], mrgLen[NUM_TWO], mrgLen[NUM_THREE]);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CumsumAndOut(
        LocalTensor<float>& bufLocal, uint32_t& totalMrgCnt, uint32_t& partTotalMrg, float* sumVal, float topp,
        bool& ifRet, const GlobalTensor<float>& sortValeGM, const GlobalTensor<uint32_t>& sortIndexGM)
    {
        LocalTensor<float> mrgRsValLocal = bufLocal; // 0k
        LocalTensor<uint32_t> mrgRsIndexLocal =
            bufLocal[MRG_SORT_INNER_LEN * NUM_FOUR].template ReinterpretCast<uint32_t>(); // 12k
        LocalTensor<float> mrgRsLocal = bufLocal[MRG_SORT_INNER_LEN * EIGHT];          // 24k
        TEventID eventIDVToS = GetTPipePtr()->FetchEventID<HardEvent::V_S>();
        TEventID eventIDSToMTE3 = GetTPipePtr()->FetchEventID<HardEvent::S_MTE3>();
        uint64_t rsvdCnt = 0;
        uint16_t oriNum = SafeCeil(partTotalMrg * NUM_TWO, SIXTY_FOUR);
        GatherMaskParams gatherMaskParams{1, oriNum, 8, 8};
        GatherMask(
            mrgRsIndexLocal,
            mrgRsLocal.ReinterpretCast<
                uint32_t>(), // 由于mrgRsLocal中的值和索引是交错存储的，所以GatherMask的作用是将Index和Value拿出来
            static_cast<uint8_t>(NUM_TWO), false, 0, gatherMaskParams, rsvdCnt);
        GatherMask(mrgRsValLocal, mrgRsLocal, static_cast<uint8_t>(1), false, 0, gatherMaskParams, rsvdCnt);

        PipeBarrier<PIPE_V>();
        ReduceSum(mrgRsLocal, mrgRsValLocal, mrgRsLocal, partTotalMrg); // 基础API
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        uint32_t topPNum = 0;
        float tempSumVal = *sumVal + mrgRsLocal.GetValue(0);
        if (tempSumVal < topp) {
            *sumVal += mrgRsLocal.GetValue(0);
        } else if (tempSumVal == topp) {
            *sumVal += mrgRsLocal.GetValue(0);
            ifRet = false;
            topPNum = partTotalMrg;
        } else {
            SumErreyOne(mrgRsValLocal, partTotalMrg, sumVal, topp, topPNum, ifRet);
        }

        if (!ifRet)
            partTotalMrg = topPNum;
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        DataCopyPad(sortValeGM[totalMrgCnt], mrgRsValLocal, {1, (uint32_t)(partTotalMrg * sizeof(float)), 0, 0, 0});
        DataCopyPad(
            sortIndexGM[totalMrgCnt], mrgRsIndexLocal, {1, (uint32_t)(partTotalMrg * sizeof(uint32_t)), 0, 0, 0});
        totalMrgCnt += partTotalMrg;
    }

    __aicore__ inline void LastCumsumAndOut(
        TOPKPParams& params, LocalTensor<float> bufLocal, uint32_t queLen, GlobalTensor<float>&& srcGm,
        uint32_t& totalMrgCnt, float* sumVal, const GlobalTensor<float>& sortValeGM,
        const GlobalTensor<uint32_t>& sortIndexGM)
    {
        TEventID eventIDMTE3ToMTE2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        TEventID eventIDMTE2ToV = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();

        LocalTensor<float> mrgRsLocal = bufLocal[MRG_SORT_INNER_LEN * EIGHT]; // bufLocal用来分配内存的
        uint32_t oriNum = 0;
        uint32_t offset = 0;
        uint32_t copyGmNum = REPEAT_MAX; // 5120
        bool ifRet = true;
        while (queLen > 0) {
            copyGmNum = copyGmNum < queLen ? copyGmNum : queLen;
            DataCopyPad(
                mrgRsLocal, srcGm[offset * NUM_TWO], //
                {1, static_cast<uint16_t>(copyGmNum * sizeof(float) * MRG_PER_ELE), 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            CumsumAndOut(bufLocal, totalMrgCnt, copyGmNum, sumVal, params.topp, ifRet, sortValeGM, sortIndexGM);
            if (!ifRet) {
                params.toppNum = totalMrgCnt;
                break;
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            offset += copyGmNum;
            queLen -= copyGmNum;
        }

        if (ifRet) {
            params.toppNum = params.rowLen;
        }
    }

    __aicore__ inline void CopyGMToGM(
        LocalTensor<float>& bufLocal, const GlobalTensor<float>&& srcGlobal, const GlobalTensor<float>&& distGlobal,
        uint32_t dataLen, uint32_t perDataLen)
    {
        TEventID eventIDMTE2ToMTE3 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
        TEventID eventIDMTE3ToMTE2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        uint32_t startOffset = 0;
        uint32_t dataLenOri = dataLen;
        while (dataLenOri > 0) {
            uint32_t copyLen = dataLenOri > GM_COPY_PER_FLOAT_SORT_MAX ? GM_COPY_PER_FLOAT_SORT_MAX : dataLenOri;
            DataCopyPad(
                bufLocal, srcGlobal[startOffset],
                {1, static_cast<uint16_t>(copyLen * sizeof(float) * perDataLen), 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            DataCopyPad(
                distGlobal[startOffset], bufLocal, {1, (uint32_t)(copyLen * sizeof(float) * perDataLen), 0, 0, 0});
            startOffset += copyLen * perDataLen;
            dataLenOri -= copyLen;
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }

    __aicore__ inline void MrgSort4Que(
        TOPKPParams& params, LocalTensor<float>& bufLocal, const GlobalTensor<float>& srcGlobal,
        const GlobalTensor<float>& destGlobal, uint32_t queToMrgNum, uint32_t* queLenToMrg, bool ifCumsum,
        const GlobalTensor<float>& sortValeGM, const GlobalTensor<uint32_t>& sortIndexGM)
    {
        uint32_t queNumOri = queToMrgNum;
        GlobalTensor<float> srcMrgGM[MRG_PER_MAX_NUM]{};
        uint32_t startOffset = 0;
        for (int32_t i = 0; i < queNumOri; i++) {
            srcMrgGM[i] = srcGlobal[startOffset];
            startOffset += queLenToMrg[i] * MRG_PER_ELE;
        }
        uint32_t queIndex[MRG_PER_MAX_NUM] = {0, 1, 2, 3};
        uint32_t currOffset[MRG_PER_MAX_NUM]{0};
        uint16_t mrgLen[MRG_PER_MAX_NUM]{0};
        MrgSrcGMList srcGMList{srcMrgGM};
        uint32_t totalMrgCnt = 0;
        LocalTensor<float> mrgRsLocal = bufLocal[MRG_SORT_INNER_LEN * EIGHT];

        TEventID eventIDVToMTE3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        TEventID eventIDMTE3ToMTE2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        float sumVal = 0.0f;
        bool ifRet = true;
        while (queNumOri > 1) {
            uint32_t partTotalMrg = 0;
            MrgSortOneTime(bufLocal, srcGMList, queNumOri, queLenToMrg, queIndex, currOffset, mrgLen);
            for (uint32_t i = 0; i < queNumOri; i++) {
                partTotalMrg += mrgLen[i];
                queLenToMrg[i] -= mrgLen[i];
                currOffset[i] += mrgLen[i] * MRG_PER_ELE;
            }
            for (uint32_t i = 0; i < queNumOri; i++) {
                if (queLenToMrg[i] == 0) {
                    for (int32_t j = i; j < queNumOri - 1; j++) {
                        queLenToMrg[j] = queLenToMrg[j + 1];
                        currOffset[j] = currOffset[j + 1];
                        queIndex[j] = queIndex[j + 1];
                    }
                    queNumOri--;
                    break;
                }
            }
            if (ifCumsum) {
                CumsumAndOut(bufLocal, totalMrgCnt, partTotalMrg, &sumVal, params.topp, ifRet, sortValeGM, sortIndexGM);
                if (!ifRet) {
                    params.toppNum = totalMrgCnt;
                    break;
                }
            } else {
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopyPad(
                    destGlobal[totalMrgCnt * MRG_PER_ELE], mrgRsLocal,
                    {1, (uint32_t)(partTotalMrg * sizeof(float) * MRG_PER_ELE), 0, 0, 0});
                totalMrgCnt += partTotalMrg;
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
        if (queLenToMrg[NUM_ZERO] > 0) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            if (ifCumsum) {
                if (ifRet) {
                    LastCumsumAndOut(
                        params, bufLocal, queLenToMrg[NUM_ZERO], srcMrgGM[queIndex[NUM_ZERO]][currOffset[NUM_ZERO]], totalMrgCnt, &sumVal,
                        sortValeGM, sortIndexGM);
                }
            } else {
                CopyGMToGM(
                    mrgRsLocal, srcMrgGM[queIndex[NUM_ZERO]][currOffset[NUM_ZERO]], destGlobal[totalMrgCnt * MRG_PER_ELE],
                    queLenToMrg[NUM_ZERO], MRG_PER_ELE);
            }
        }
    }

    __aicore__ inline void MrgSortMQue(
        TOPKPParams& params, LocalTensor<float>& bufLocal, uint32_t queToMrgNum, uint32_t perQueLen,
        uint32_t queLenTail, const GlobalTensor<float>& mrgGM, const GlobalTensor<float>& mrgDestGM, uint32_t& mrgCnt,
        uint32_t* resultMrgLen)
    {
        GlobalTensor<float> sortValeGM{};
        GlobalTensor<uint32_t> sortIndexGM{};
        auto queNumOri = queToMrgNum;
        uint32_t startOffset = 0;
        uint32_t queLenToMrg[MRG_PER_MAX_NUM]{0};
        TEventID eventIDMTE3ToMTE2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        while (queNumOri > 1) {
            uint32_t qNum = queNumOri > MRG_PER_MAX_NUM ? MRG_PER_MAX_NUM : queNumOri;
            auto totalMrgLen = 0;
            for (int i = 0; i < qNum; i++) {
                queLenToMrg[i] = perQueLen;
                if ((mrgCnt * MRG_PER_MAX_NUM + i) == (queToMrgNum - 1)) {
                    if (queLenTail > 0)
                        queLenToMrg[i] = queLenTail;
                }
                totalMrgLen += queLenToMrg[i];
            }
            MrgSort4Que(
                params, bufLocal, mrgGM[startOffset], mrgDestGM[startOffset], qNum, queLenToMrg, false, sortValeGM,
                sortIndexGM);
            startOffset += totalMrgLen * MRG_PER_ELE;
            queNumOri -= qNum;
            resultMrgLen[mrgCnt] = totalMrgLen;
            mrgCnt++;
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
        if (queNumOri > 0) {
            uint32_t dataLen = perQueLen;
            if (queLenTail > 0)
                dataLen = queLenTail;
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            CopyGMToGM(bufLocal, mrgGM[startOffset], mrgDestGM[startOffset], dataLen, MRG_PER_ELE);
            resultMrgLen[mrgCnt] = dataLen;
            mrgCnt++;
        }
    }

    __aicore__ inline void MrgSPartAll(
        TOPKPParams& params, LocalTensor<float>& bufLocal,
        const GlobalTensor<float>& srcGlobal, // 存了分段排序结果
        const GlobalTensor<float>& destGlobal,
        const GlobalTensor<float>& sortValueResult, 
        const GlobalTensor<uint32_t>& sortIndexResult)
    {
        uint32_t queToMrgNum = params.eightKPartNum;
        uint32_t queLenToMrg[MRG_PER_MAX_NUM]{0};

        auto mrgGM = srcGlobal[params.rowId * params.rowLen * MRG_PER_ELE];
        auto mrgDestGM = destGlobal[params.rowId * params.rowLen * MRG_PER_ELE];
        auto sortValeGM = sortValueResult[params.rowId * params.rowLen];
        auto sortIndexGM = sortIndexResult[params.rowId * params.rowLen];

        if (queToMrgNum <= 1) {
            queLenToMrg[NUM_ZERO] = params.eightKPartTail > 0 ? params.eightKPartTail : S_PART_MIN_LEN;
            uint32_t totalMrgCnt = 0;
            float sumVal = 0.0f;
            LastCumsumAndOut(params, bufLocal, queLenToMrg[NUM_ZERO], mrgGM[NUM_ZERO], totalMrgCnt, &sumVal, sortValeGM, sortIndexGM);
        } else if (queToMrgNum <= MRG_PER_MAX_NUM) {
            for (int32_t i = 0; i < queToMrgNum; i++) {
                queLenToMrg[i] = S_PART_MIN_LEN;
            }
            if (params.eightKPartTail > 0) {
                queLenToMrg[queToMrgNum - 1] = params.eightKPartTail;
            }

            MrgSort4Que(params, bufLocal, mrgGM, mrgDestGM, queToMrgNum, queLenToMrg, true, sortValeGM, sortIndexGM);
        } else {
            uint32_t resultMrgLen[MRG_FIRST_MAX_NUM]{0};
            auto queNumOri = queToMrgNum;
            auto mrgGMOri = mrgGM;
            auto mrgDestGMOri = mrgDestGM;
            uint32_t perQueLen = S_PART_MIN_LEN;
            uint32_t queLenTail = params.eightKPartTail;

            while (queNumOri > MRG_PER_MAX_NUM) {
                uint32_t mrgCnt = 0;

                MrgSortMQue(
                    params, bufLocal, queNumOri, perQueLen, queLenTail, mrgGMOri, mrgDestGMOri, mrgCnt, resultMrgLen);
                if (mrgCnt <= MRG_PER_MAX_NUM) {
                    MrgSort4Que(
                        params, bufLocal, mrgDestGMOri, mrgDestGMOri, mrgCnt, resultMrgLen, true, sortValeGM,
                        sortIndexGM);
                    break;
                } else {
                    queNumOri = mrgCnt;
                    auto tempMiddle = mrgGMOri;
                    mrgGMOri = mrgDestGMOri;
                    mrgDestGMOri = tempMiddle;
                    perQueLen = resultMrgLen[NUM_ZERO];
                    queLenTail = resultMrgLen[mrgCnt - 1];
                }
            }
        }
    }

    __aicore__ inline void SortAll(
        TOPKPParams& params, const GlobalTensor<float>& srcGlobal, const GlobalTensor<float>& sortSrc0Global,
        const GlobalTensor<float>& sortSrc1Global, GlobalTensor<uint32_t>& srcIndexGlobal)
    {
        auto tensor0 = params.tensor0;
        LocalTensor<float> bufLocal = tensor0.template ReinterpretCast<float>();
        // 归并排序
        // 每一段先排好
        SortSPartAll(
            params, bufLocal, srcGlobal,
            sortSrc0Global); // 未排序结果在GlobalUser里，即srcGlobal中。分段排好的结果存在了sortSrc0Global里面，也存在了bufLocal里面
        // 合起来
        MrgSPartAll(params, bufLocal, sortSrc0Global, sortSrc1Global, srcGlobal, srcIndexGlobal);
    }
};
} // namespace TopKTopPSample

#endif // TOP_K_TOP_P_SORT_CUMSUM_H