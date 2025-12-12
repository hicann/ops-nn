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
 * \file top_k_top_p_sample.h
 * \brief
 */

#ifndef TOP_K_TOP_P_SAMPLE_H
#define TOP_K_TOP_P_SAMPLE_H

#include "top_k_top_p_comm.h"
#include "top_k_top_p_sort_cumsum.h"

namespace TopKTopPSample {

constexpr uint32_t INNER_LOOP_ELE = 8 * 1024;
constexpr uint32_t TOPP_K_FIX = 32;
constexpr uint32_t ROW_LEN_MAX = 128;   // max batch size processed for single core under CUT_B template
constexpr uint32_t TOPK_MAX = 1 * 1024;
constexpr uint32_t SORT32_PER_LEN = 2 * 1024;
constexpr uint32_t SORT32_MAX_LOOP = INNER_LOOP_ELE / SORT32_PER_LEN;
constexpr uint32_t MRGSORT_PER_NUM = 4;
constexpr uint32_t MRGSORT_PER_NUM_LAST = 2;
constexpr uint32_t SORT32_PERGROUP_LEN = 2 * TOPK_MAX;
constexpr float TOPP_MAX = 1.0f;
constexpr uint32_t FP32_NEG_INF_BITS = 0b11111111100000000000000000000000;

template <typename T>
class TopKTopPSampleKernel
{
public:
    TOPKPParams params{};
    TopKTopPSampleSortKernel<T> sortOp;

    __aicore__ inline TopKTopPSampleKernel(){};
    __aicore__ inline void Init(
        GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKPSelect,
        GM_ADDR workspace, TopKTopPSampleTilingData tilingData, TPipe* tPipe)
    {
        if ASCEND_IS_AIV {
            InitTilingParams(tilingData);
            InitAndSetBuffer(logits, topKs, topPs, q, logitsSelectIdx, logitsTopKPSelect, workspace, tPipe);
        }
    }

    __aicore__ inline void Process()
    {
        if (g_coreType == AscendC::AIC){
            return;
        }

        for (int32_t processRowCount = 0; processRowCount < this->rtCoreRowNum; processRowCount++) {
            uint32_t rowId = processRowCount;
            uint32_t kCount = 0;
            this->p = topPGlobal.GetValue(rowId); // 取p值
            float fp32P = 0;
            if constexpr (std::is_same<T, bfloat16_t>::value){
                fp32P = ToFloat(this->p);
            }else {
                fp32P = static_cast<float>(this->p);
            }
            // float fp32P = std::is_same<T, bfloat16_t>::value ? ToFloat(this->p) : static_cast<float>(this->p);
            params.topp = fp32P;
            this->ifFind = 0;
            this->topPNum = 0;
            uint32_t maxIndex{0};
            params.softmaxLoopTime = this->softmaxLoopTime;
            params.softmaxLoopEleTail = this->softmaxLoopEleTail;
            params.softmaxLoopEleTailPad = this->softmaxLoopEleTailPad;

            if (this->rowIdTopkList[rowId] == 1) {
                // classic topK and topP
                this->k = topKGlobal.GetValue(rowId);
                kCount = this->k;
                this->k_pad = Align(this->k, EIGHT);  // logits转成float32后，需满足32字节对齐，k值必须为8的整倍数
                TopKLoopProcess(rowId, kCount);       // 排序且取前k个，存在了localValueRs中
                SoftMaxCompute(kCount, localValueRs); // SoftMaxCompute的计算结果固定存在了sampleLogitsLocal中
                if (fp32P > 0 && fp32P < TOPP_MAX) {
                    TopPCompute(rowId, kCount, fp32P); // 使能TopP，计算topPNum
                    SoftMaxCompute(this->topPNum, localValueRs);
                } else {
                    this->topPNum = kCount; // 不使能TopP
                }
                if(this->ifQSampleCompute){
                    QSampleCompute(rowId, this->k_pad, this->topPNum); // 从sampleLogitsLocal中取值计算Q，计算结果存在了sampleLogitsLocal中
                }
                ArgMaxAndGather(this->topPNum, maxIndex); // ReduceMax softMax的结果（sampleLogitsLocal）找到maxIndex
                CopyOut(rowId, this->k_pad, maxIndex, true, this->isNeedLogits); // 存maxIndex, 如果isNeedLogits==true,将源Logits搬出到dstGM，源Logits存在localValueRs里，前面计算不会污染源Logits
            } else if (fp32P <= 0 || fp32P >= TOPP_MAX) {
                // 无 topK 和 topP : Performance preservation using vector softmax and argmax          
                kCount = TOPP_K_FIX;
                this->k_pad = Align(kCount, EIGHT);
                bool ifGetIndex = false;
                // 不做排序，全长度Softmax取maxIndex
                SoftMaxAll(rowId, kCount, logitsGlobal, this->ifQSampleCompute, maxIndex, this->isNeedLogits); //不做Qsample时，拿到的是sortmax全排序后的一行
                if(!this->ifQSampleCompute){
                    ifGetIndex = true;
                }
                CopyOut(rowId, this->rowLen, maxIndex, ifGetIndex, false); 
            } else {
                // 无topK 有topP 猜topK
                kCount = this->topKGuess > this->rowLen ? this->rowLen : this->topKGuess;
                this->k_pad = Align(kCount, EIGHT);
                SoftMaxAll(rowId, kCount, logitsGlobal, false, maxIndex, false); //做了Softmax但是没做sort的存在了logitsGlobalUser里，做了softmax继续做了sort的存在了buf3里
                TopPCompute(rowId, kCount, fp32P); // 因为已经做完softmax了 
                if (this->ifFind == 1) {           // 猜到了
                    SoftMaxCompute(this->topPNum, localValueRs);
                    if (this->ifQSampleCompute) {
                        QSampleCompute(rowId, this->k_pad, this->topPNum);
                    }
                    // Copy out this->topPNum filtered logits toward result GM tensor
                    ArgMaxAndGather(this->topPNum, maxIndex);
                    CopyOut(rowId, this->topPNum, localIndexRs, maxIndex);
                } else { // GuessKFailed: 没猜到
                    params.tensor0 = buf0.Get<float>();
                    params.rowNum = this->rowNum;
                    params.rowLen = this->rowLen;
                    params.rowId = rowId;
                    // 全排序，拿到排序结果，也会做topP，拿到toppNum
                    sortOp.SortAll(
                        params, logitsGlobalUser, sortPartGlobalUser, sortAllGlobalUser, // 从logitsGlobalUser取做了softmax但是没排序的值做sort，然后求topPNum，sort的value结果存在了logitsGlobalUser中，index结果放在了srcIndexGlobalUser中
                        srcIndexGlobalUser);  
                    auto loopTime = SafeCeil(params.toppNum, SOFTMAX_PER_LEN); //(params.toppNum + SOFTMAX_PER_LEN - 1) / SOFTMAX_PER_LEN;
                    auto loopTail = params.toppNum % SOFTMAX_PER_LEN;
                    auto loopTailPad = (loopTail + THIRTY_ONE) / THIRTY_TWO * THIRTY_TWO;
                    params.softmaxLoopTime = loopTime;
                    params.softmaxLoopEleTail = loopTail;
                    params.softmaxLoopEleTailPad = loopTailPad;
                    GlobalTensor<uint32_t> sampleIndexGlobal = srcIndexGlobalUser[rowId * params.rowLen];
                    // 做softmax
                    this->k_pad = Align(params.toppNum, EIGHT);
                    if (this->ifQSampleCompute) {
                        SoftMaxAll(rowId, 0, logitsGlobalUser, this->ifQSampleCompute, maxIndex, false); 
                    }
                    CopyOut(rowId, params.toppNum, sampleIndexGlobal, maxIndex);
                }
            }
        }
    }

private:
    __aicore__ inline void InitTilingParams(TopKTopPSampleTilingData tilingData)
    {
        this->rowNum = tilingData.rowNum;
        this->rowLen = tilingData.rowLen;
        this->headCoreNum = tilingData.headCoreNum;
        this->innerLoopEle = tilingData.innerLoopEle;
        this->eps = tilingData.eps;
        this->isNeedLogits = tilingData.isNeedLogits;
        this->topKGuess = tilingData.topKGuess;
        this->innerLoopTime = tilingData.innerLoopTime;
        this->innerLoopEleTail = tilingData.innerLoopEleTail;
        this->softmaxLoopTime = tilingData.softmaxLoopTime;
        this->softmaxLoopEleTail = tilingData.softmaxLoopEleTail;
        this->innerLoopEleTailPad = tilingData.innerLoopEleTailPad;
        this->softmaxLoopEleTailPad = tilingData.softmaxLoopEleTailPad;
        this->mrgMode = tilingData.mrgMode;
        params.eightKPartNum = tilingData.eightKPartNum;
        params.eightKPartTail = tilingData.eightKPartTail;
        params.eightKPartTailPad = tilingData.eightKPartTailPad;

        uint32_t blockId = GetBlockIdx();
        uint32_t perHeadCoreRowNum = tilingData.perHeadCoreRowNum;
        int32_t coreHeadBias = (blockId - headCoreNum);

        if (coreHeadBias < 0) { // blockId < headCoreNum
            // head core
            this->rtCoreRowNum = perHeadCoreRowNum;
            this->startTaskId = blockId * perHeadCoreRowNum;
        } else {
            // tail core
            this->rtCoreRowNum = tilingData.tailCoreRowNum;
            this->startTaskId = headCoreNum * perHeadCoreRowNum + coreHeadBias * tilingData.tailCoreRowNum;
        }
        if (this->topKGuess > TOPK_MAX) {
            this->topKGuess = TOPK_MAX;
        }
    }

    __aicore__ inline void InitAndSetBuffer(
        GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKPSelect,
        GM_ADDR workspace, TPipe* tPipe)
    {
        // InitGlobalBuffer as '(type *)<ADDR_OFFSET>' where <ELEMENT AMOUNT> is not recommended.
        uint32_t gmOffset = this->startTaskId * this->rowLen;   // element offset by row-offset * col_size (B_offset * S)

        // Read-only tensors
        logitsGlobal.SetGlobalBuffer((__gm__ T*)logits + gmOffset);                 // logits in [B, S]
        topKGlobal.SetGlobalBuffer((__gm__ uint32_t*)topKs + this->startTaskId);    // top_k in [B, ]
        topPGlobal.SetGlobalBuffer((__gm__ T*)topPs + this->startTaskId);           // top_p in [B, ]
        
        uint32_t coreEle_ = this->rtCoreRowNum == 0 ? this->rowLen : this->rtCoreRowNum * this->rowLen; 
        // Write tensors shall aligned to their batch offset
        dstLogitsGlobal.SetGlobalBuffer((__gm__ float*)logitsTopKPSelect + gmOffset, coreEle_);           // logits_top_kp_select in [B, S]
        dstIndexGlobal.SetGlobalBuffer((__gm__ uint64_t*)logitsSelectIdx + this->startTaskId);  // logits_select_idx in [B, ]

        if (q != nullptr) {
            qGlobal.SetGlobalBuffer((__gm__ float*)q + gmOffset);  // q in [B, S]
            this->ifQSampleCompute = true;
        }
        else {
            this->ifQSampleCompute = false;
        }

        // Init shared workspace
        // the workspace seems to be a shared cache for intermediate results from the GM data
        uint32_t count = this->rowNum * this->rowLen;
        uint32_t offset = 0;    // offsets for different cache blocks shared by ALL kernels
        uint32_t wsKernelOfs = gmOffset;   // Kernel addr offset in current workspace block

        // If there is NO inter-kernel communications, local workspace shall start from the GM offset of current core.
        logitsGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs);  // global Logits workspace in [B, S]
        offset += count;
        wsKernelOfs *= MRG_PER_ELE; // Kernel Addr offset in workspace block is sized by column factor for ONLY ONCE
        sortPartGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs + offset);
        offset += count * MRG_PER_ELE;
        sortAllGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs + offset);
        offset += count * MRG_PER_ELE;
        srcIndexGlobalUser.SetGlobalBuffer((__gm__ uint32_t*)workspace + gmOffset + offset);

        // Init top_kp_selected_logits GM to zeros when if isNeedLogits
        if(this->isNeedLogits){
            AscendC::InitGlobalMemory(dstLogitsGlobal, this->rtCoreRowNum * this->rowLen, static_cast<float>(SEL_LOGITS_DEF_VAL));
            // Sync single core I/O steps
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            // Prevent dirty write under multi-core scenarios
            SyncAll();
        } 

        this->pipe = tPipe;
        pipe->InitBuffer(buf0, INNER_LOOP_ELE * sizeof(float));
        pipe->InitBuffer(buf1, INNER_LOOP_ELE * sizeof(float));
        pipe->InitBuffer(buf2, INNER_LOOP_ELE * sizeof(float));
        pipe->InitBuffer(buf3, INNER_LOOP_ELE * sizeof(float));
        pipe->InitBuffer(buf4, INNER_LOOP_ELE * sizeof(float));
        pipe->InitBuffer(buf5, INNER_LOOP_ELE * sizeof(float));
        localValueRs = buf0.Get<float>();
        localIndexRs = buf1.Get<uint32_t>();
        sampleLogitsLocal = buf3.Get<float>();

        for (uint32_t row = 0; row < this->rtCoreRowNum; row++) {
            uint32_t temp = topKGlobal.GetValue(row); // 自GM取当前行logits的k值
            uint32_t topKMax = this->rowLen > TOPK_MAX ? TOPK_MAX : this->rowLen;
            if (temp > topKMax) {                    // k值大于1024 or s[b], topK不使能
                this->rowIdToppList[row] = 1;
            } else {
                this->rowIdTopkList[row] = 1;
            }
        }
    }

    __aicore__ inline void GetRowMaxInner(
        LocalTensor<float>& src, LocalTensor<float>& reduceMaxMiddle, uint32_t countLen, float* rowMax,
        const GlobalTensor<float>& dist)
    {
        ReduceMax(reduceMaxMiddle, src, reduceMaxMiddle, countLen, false);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto temp = reduceMaxMiddle.GetValue(0);
        *rowMax = *rowMax > temp ? *rowMax : temp;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        DataCopyPad(
            dist, src, {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0}); // 将cast后未做排序的logits存入globalUser
    }

    template <typename DTYPE>
    __aicore__ inline void GetRowMax(uint32_t rowId, float* rowMax, const GlobalTensor<DTYPE>& src, bool ifCopyOut)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN;
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint32_t startIndex = rowId * this->rowLen;
        LocalTensor<T> valueLocal = buf0.Get<T>();
        LocalTensor<float> valueLocalCast = buf2.Get<float>();
        LocalTensor<float> reduceMaxMiddle = buf4.Get<float>();
        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; innerLoopCount++) {
            uint32_t gmOffset = startIndex + innerLoopCount * innerCopyLen;
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            DataCopy(valueLocal, logitsGlobal[gmOffset], innerCopyLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(valueLocalCast, valueLocal, RoundMode::CAST_NONE, countLen);
            PipeBarrier<PIPE_V>();
            GetRowMaxInner(valueLocalCast, reduceMaxMiddle, countLen, rowMax, logitsGlobalUser[gmOffset]);
            if (ifCopyOut) { // 写死，false，不走该分支
                DataCopyPad(
                    dstLogitsGlobal[gmOffset], valueLocalCast, {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0});
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void SoftMaxFstCompute(uint32_t rowId, float rowMax, float* reduceSumMax)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN; // 8 * 1024 * 2 个元素，每个元素2字节，满足32k
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint32_t startIndex = rowId * this->rowLen;
        LocalTensor<float> valueLocalCast = buf0.Get<float>();
        LocalTensor<float> valueLocalMiddle = buf2.Get<float>();
        LocalTensor<float> reduceSumVal = buf4.Get<float>();

        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; innerLoopCount++) {
            uint32_t gmOffset = startIndex + innerLoopCount * innerCopyLen;
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            DataCopy(
                valueLocalCast, logitsGlobalUser[gmOffset],
                innerCopyLen); // 从globalUser里将GetRowMax函数中存储的未排序的做了cast的logits拿出来
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Duplicate(valueLocalMiddle, rowMax, countLen); // 复制rowMax并填充
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Sub(valueLocalMiddle, valueLocalCast, valueLocalMiddle, countLen);
            PipeBarrier<PIPE_V>();
            Exp(valueLocalMiddle, valueLocalMiddle, countLen);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(
                logitsGlobalUser[gmOffset], valueLocalMiddle,
                {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0});           // 将EXP结果存储到globalUser里
            ReduceSum(reduceSumVal, valueLocalMiddle, reduceSumVal, countLen); // 基础API
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            *reduceSumMax = *reduceSumMax + reduceSumVal.GetValue(0); // S
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void SoftMaxSecCompute( // 做softmax中的exp/S的步骤
        LocalTensor<float>& valueLocalCast, float reduceSumMax, uint32_t gmOffset, uint32_t innerCopyLen,
        uint32_t countLen)
    {
        LocalTensor<float> valueRs = buf0.Get<float>();
        DataCopy(
            valueRs, logitsGlobalUser[gmOffset],
            innerCopyLen); // 将softmaxFst中算的exp临时存储在了GM中，现在搬到local中
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(valueLocalCast, reduceSumMax, countLen);
        PipeBarrier<PIPE_V>();
        Div(valueLocalCast, valueRs, valueLocalCast, countLen); // softmax计算结果存储在valueLocalCast中
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftMaxAndTopkCompute(uint32_t rowId, float reduceSumMax, uint32_t kCount)
    {
        uint32_t innerCopyLen = INNER_LOOP_ELE;
        uint32_t countLen = INNER_LOOP_ELE;
        uint32_t startIndex = rowId * this->rowLen;

        LocalTensor<uint32_t> localIndex = buf1.Get<uint32_t>();
        LocalTensor<float> localSortResult = buf2.Get<float>();
        LocalTensor<float> localValueCast = buf3.Get<float>();

        uint32_t gmOffset = startIndex;
        uint32_t indexOffset = 0;
        for (int32_t innerLoopCount = 0; innerLoopCount < this->innerLoopTime; ++innerLoopCount) {
            if (this->innerLoopEleTail > 0 && innerLoopCount == this->innerLoopTime - 1) {
                // tail block
                innerCopyLen = this->innerLoopEleTailPad;
                countLen = this->innerLoopEleTail;
            }
            SoftMaxSecCompute(localValueCast, reduceSumMax, gmOffset, innerCopyLen, countLen);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3); // It should be ok to use SetWaitFlag for flows share the same EVENT_ID.

            DataCopyPad(
                logitsGlobalUser[gmOffset], localValueCast,
                {1, static_cast<uint32_t>(countLen * SIZEOF_FP32), 0, 0,
                 0}); // 将softmax结果存储到GMUser里，在SoftMaxAl之后如果没猜到k值，进入全排序分支后会用到
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            CreateVecIndex(localIndex[NUM_ZERO].ReinterpretCast<int32_t>(), static_cast<int32_t>(indexOffset), countLen);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (kCount > 0) {
                if (innerLoopCount == 0) { // softmax结束后做排序，这是一个滚动排序
                    // Cast full sort on the first section of current batch
                    Sort32Block(localValueCast, localIndex, countLen, false);
                } else {
                    // Update topK list by sorting previous topK with current section.
                    Sort32BlockNext(localValueCast, localIndex, countLen, localSortResult, kCount);
                }
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            }
            gmOffset += innerCopyLen;
            indexOffset += countLen;
        }
        if (kCount > 0) {
            Extract(localValueRs, localIndex, localSortResult, SafeCeil(kCount, THIRTY_TWO)); // 取出排序的值和索引
            PipeBarrier<PIPE_V>();
            DataCopy(localValueCast, localValueRs, this->k_pad);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        }        
    }

    __aicore__ inline void SoftMaxAndQSampleCompute(uint32_t rowId, float reduceSumMax, uint32_t& maxIndex)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN;
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint32_t indexOffset = 0;
        uint32_t startIndex = rowId * this->rowLen;
        uint32_t gmOffset = startIndex;
        LocalTensor<float> localValueCast = buf2.Get<float>();
        LocalTensor<float> sampleDist = buf4.Get<float>();
        float maxValue{0};
        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; ++innerLoopCount) {
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            SoftMaxSecCompute(localValueCast, reduceSumMax, gmOffset, innerCopyLen, countLen);
            DataCopy(sampleDist, qGlobal[gmOffset], innerCopyLen);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Abs(sampleDist, sampleDist, countLen);
            PipeBarrier<PIPE_V>();
            Adds(sampleDist, sampleDist, this->eps, countLen);
            PipeBarrier<PIPE_V>();
            Div(localValueCast, localValueCast, sampleDist, countLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(localValueCast, localValueCast, localValueCast, countLen, true); // 基础API 返回最大值和最大值索引
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            float value = localValueCast.GetValue(0);
            float index = localValueCast.GetValue(1);
            auto tempIndex = *reinterpret_cast<uint32_t*>(&index);
            if (value > maxValue){
                maxIndex = indexOffset + tempIndex;
            }
            maxValue = maxValue > value ? maxValue : value;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            gmOffset += innerCopyLen;
            indexOffset += countLen;
        }
    }

    template <typename DTYPE>
    __aicore__ inline void SoftMaxAll(
        int32_t rowId, uint32_t kCount, const GlobalTensor<DTYPE>& src, bool ifQSampleCompute, uint32_t& maxIndex,
        bool ifCopyOut)
    {
        float rowMax{0};
        float reduceSumMax{0};
        if constexpr (std::is_same<DTYPE, float>::value) { //  判断logits是否为float32？
            rowMax = src[rowId * this->rowLen].GetValue(0);
            SoftMaxFstCompute(rowId, rowMax, &reduceSumMax);
            SoftMaxAndQSampleCompute(rowId, reduceSumMax, maxIndex);
        } else {
            GetRowMax(rowId, &rowMax, src, ifCopyOut); 
            SoftMaxFstCompute(
                rowId, rowMax, &reduceSumMax); // softFst做softmax中的exp(搬到临时GM中存储) 和 s（reduceSumMax）
            if (ifQSampleCompute) {
                SoftMaxAndQSampleCompute(rowId, reduceSumMax, maxIndex);
            }
            else {
                SoftMaxAndTopkCompute(rowId, reduceSumMax, kCount);
            }
        }
    }

    __aicore__ inline void SortOneTime(
        LocalTensor<float>&& srcData, LocalTensor<uint32_t>&& srcIndex, uint32_t dataLen, LocalTensor<float>& sortedLocal)
    {
        LocalTensor<float> concatLocal;
        LocalTensor<float> sortTmpLocal = buf5.Get<float>();
        Concat(concatLocal, srcData, sortTmpLocal, dataLen / SIXTEEN);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(
            sortedLocal, concatLocal, srcIndex.ReinterpretCast<uint32_t>(), sortTmpLocal, dataLen / THIRTY_TWO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Sort32Block(
        LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex, uint32_t dataLen, bool mrg)
    {
        auto sort32LoopNum = SafeCeil(dataLen, SORT32_PER_LEN);
        auto sort32LoopTailNum = dataLen % SORT32_PER_LEN;
        auto compareLen = SORT32_PER_LEN;
        auto compareLenPad = SORT32_PER_LEN;
        LocalTensor<float> msgSortPart1 = buf0.Get<float>();
        LocalTensor<float> msgSortPart2 = buf4.Get<float>();
        LocalTensor<float> msgSortLocal[SORT32_MAX_LOOP] = {
            msgSortPart1, msgSortPart1[SORT32_PER_LEN * NUM_TWO], msgSortPart2, msgSortPart2[SORT32_PER_LEN * NUM_TWO]};
        uint32_t queLen[SORT32_MAX_LOOP] = {0};

        for (uint32_t sort32LoopCount = 0; sort32LoopCount < sort32LoopNum; ++sort32LoopCount) {
            uint32_t sort32BlkOfs = sort32LoopCount * SORT32_PER_LEN;
            if (sort32LoopTailNum > 0 && sort32LoopCount == (sort32LoopNum - 1)) {
                compareLen = sort32LoopTailNum;
                compareLenPad = SafeCeil(compareLen, THIRTY_TWO) * THIRTY_TWO;
                if (compareLenPad > compareLen) {
                    for (uint32_t i = compareLen; i < compareLenPad; ++i) {
                        srcData[sort32BlkOfs].SetValue(i, FLOAT_MIN);
                    }
                    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
                }
            }
            SortOneTime(srcData[sort32BlkOfs], srcIndex[sort32BlkOfs], compareLenPad, msgSortLocal[sort32LoopCount]);
            queLen[sort32LoopCount] = compareLen;
        }

        uint16_t elementLengths[MRGSORT_PER_NUM] = {TOPK_MAX, TOPK_MAX, TOPK_MAX, TOPK_MAX};
        uint16_t valueBits[5] = {0b11, 0b11, 0b11, 0b111, 0b1111};
        MrgSort4Info localParams = {};
        LocalTensor<float> msgSortLocalResult[3] = {buf2.Get<float>(), buf3.Get<float>(), buf1.Get<float>()};
        LocalTensor<float> msgSortLocalResultTemp = {};
        uint32_t qid = 0;
        if (mrg) {
            qid = 1;
            msgSortLocalResultTemp = msgSortLocalResult[NUM_ONE];
        } else {
            qid = 0;
            msgSortLocalResultTemp = msgSortLocalResult[NUM_ZERO];
        }
        if (sort32LoopNum <= 1) {
            // DataCopy demands 32bytes alignment, immediate call using calCount aligned to 32bytes is valid ONLY WHEN msgSortLocal HAS BEEN ALIGNED to 32bytes yet.   
            DataCopy(msgSortLocalResultTemp, msgSortLocal[NUM_ZERO], static_cast<uint32_t>(Align(queLen[NUM_ZERO] * NUM_TWO, EIGHT)));
            PipeBarrier<PIPE_V>();
            this->queLenGlobal[qid] = queLen[NUM_ZERO];
        } else {
            this->queLenGlobal[qid] = TOPK_MAX;
            for (uint32_t i = 0; i < sort32LoopNum; i++) {
                elementLengths[i] = queLen[i] > TOPK_MAX ? TOPK_MAX : queLen[i];
            }
            localParams = {elementLengths, false, valueBits[sort32LoopNum], 1};
            MrgSort<float>(
                msgSortLocalResultTemp, {msgSortLocal[NUM_ZERO], msgSortLocal[NUM_ONE], msgSortLocal[NUM_TWO], msgSortLocal[NUM_THREE]}, localParams);
            PipeBarrier<PIPE_V>();
        }

        if (mrg) {
            elementLengths[NUM_ZERO] = static_cast<uint16_t>(this->queLenGlobal[NUM_ZERO]);
            elementLengths[NUM_ONE] = static_cast<uint16_t>(this->queLenGlobal[NUM_ONE]);
            localParams = {elementLengths, false, valueBits[MRGSORT_PER_NUM_LAST], 1};
            MrgSort<float>(
                msgSortLocalResult[NUM_TWO],
                {msgSortLocalResult[NUM_ZERO], msgSortLocalResult[NUM_ONE], msgSortLocalResult[NUM_ONE], msgSortLocalResult[NUM_ONE]}, localParams);
            PipeBarrier<PIPE_V>();
            DataCopy(msgSortLocalResult[NUM_ZERO], msgSortLocalResult[NUM_TWO], SORT32_PERGROUP_LEN);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Sort32BlockNext(
        LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex, uint32_t dataLen, LocalTensor<float>& mrgSortResult,
        uint32_t kCount)
    {
        uint32_t firstTopkIndex = (kCount - 1) * NUM_TWO;
        uint64_t rsvdCnt = 0;
        uint64_t oriRsvdCnt = 0;
        GatherMaskParams gatherMaskParams{1, 1, 8, 8};
        LocalTensor<uint8_t> compareIndexMask = buf5.Get<uint8_t>();

        float firstTopk = mrgSortResult.GetValue(firstTopkIndex);
        CompareScalar(compareIndexMask, srcData, firstTopk, CMPMODE::GT, SafeCeil(dataLen, SIXTY_FOUR) * SIXTY_FOUR);
        PipeBarrier<PIPE_V>();
        GatherMask(
            srcData, srcData, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMask(
            srcIndex, srcIndex, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        if (rsvdCnt > 0) {
            uint64_t oriRsvCnt = rsvdCnt;
            rsvdCnt = SafeCeil(rsvdCnt, THIRTY_TWO) * THIRTY_TWO;
            if (rsvdCnt > oriRsvCnt) {
                for (uint32_t i = oriRsvCnt; i < rsvdCnt; ++i) {
                    srcData.SetValue(i, FLOAT_MIN);
                }
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            }
            Sort32Block(srcData, srcIndex, rsvdCnt, true);
        }
    }

    __aicore__ inline void TopKLoopProcess(int32_t rowId, uint32_t kCount)
    {
        LocalTensor<T> localValue = buf0.Get<T>();
        LocalTensor<uint32_t> localIndex = buf1.Get<uint32_t>(); // 8 * 1024 * 4
        LocalTensor<float> localSortResult = buf2.Get<float>();
        LocalTensor<float> localValueCast = buf3.Get<float>();

        uint32_t startIndex = rowId * this->rowLen;
        uint32_t indexOffset = 0;
        uint32_t innerCopyLen = INNER_LOOP_ELE; // 8 * 1024
        uint32_t countLen = INNER_LOOP_ELE;
        uint32_t gmOffset = startIndex;

        for (uint32_t innerLoopCount = 0; innerLoopCount < this->innerLoopTime; ++innerLoopCount) {
            if (this->innerLoopEleTail > 0 && innerLoopCount == this->innerLoopTime - 1) {
                innerCopyLen = this->innerLoopEleTailPad;
                countLen = this->innerLoopEleTail;
            }
            CreateVecIndex(localIndex[NUM_ZERO].ReinterpretCast<int32_t>(), static_cast<int32_t>(indexOffset), countLen);
            PipeBarrier<PIPE_V>();
            DataCopy(localValue, logitsGlobal[gmOffset], innerCopyLen);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Cast(localValueCast, localValue, RoundMode::CAST_NONE, countLen); // half/bfloat16 -> float32
            PipeBarrier<PIPE_V>();
            if (innerLoopCount == 0) {
                Sort32Block(localValueCast, localIndex, countLen, false);
            } else {
                Sort32BlockNext(localValueCast, localIndex, countLen, localSortResult, kCount);
            }

            gmOffset += innerCopyLen;
            indexOffset += countLen;
        }
        Extract(localValueRs, localIndexRs, localSortResult, TOPK_MAX / THIRTY_TWO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftMaxCompute(uint32_t count, LocalTensor<float>& sampleSrc)
    {
        LocalTensor<float> tempLocalSoftmax = buf4.Get<float>();
        auto maxLogits = sampleSrc.GetValue(static_cast<uint32_t>(0));
        Duplicate(sampleLogitsLocal, maxLogits, count);
        PipeBarrier<PIPE_V>();
        Sub(sampleLogitsLocal, sampleSrc, sampleLogitsLocal, count);
        PipeBarrier<PIPE_V>();
        Exp(sampleLogitsLocal, sampleLogitsLocal, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(tempLocalSoftmax, sampleLogitsLocal, tempLocalSoftmax, count);
        PipeBarrier<PIPE_V>();
        auto topKExpSum = tempLocalSoftmax.GetValue(static_cast<uint32_t>(0));
        Duplicate(tempLocalSoftmax, topKExpSum, count);
        PipeBarrier<PIPE_V>();
        Div(sampleLogitsLocal, sampleLogitsLocal, tempLocalSoftmax, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TopPCompute(int32_t rowId, uint32_t count, float fp32P)
    {
        float cumSum = 0.0;
        for (uint32_t index = 0; index < count; index++) {
            cumSum += sampleLogitsLocal.GetValue(index); // sampleLogitsLocal is softmax results
            if (cumSum > fp32P) {
                this->topPNum = index + 1;
                this->ifFind = 1;
                break;
            }
        }
        if (this->ifFind == 0) {
            this->topPNum = count;
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    }

    __aicore__ inline void QSampleCompute(int32_t rowId, uint32_t copyCnt, uint32_t calCnt)
    {
        LocalTensor<float> tempLocalQsample = buf5.Get<float>();
        DataCopyEx(tempLocalQsample, qGlobal[(rowId) * this->rowLen], copyCnt, 1, false);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Abs(tempLocalQsample, tempLocalQsample, calCnt);
        PipeBarrier<PIPE_V>();
        Adds(tempLocalQsample, tempLocalQsample, this->eps, calCnt);
        PipeBarrier<PIPE_V>();
        Div(sampleLogitsLocal, sampleLogitsLocal, tempLocalQsample, calCnt);
        PipeBarrier<PIPE_V>();
    }

    // argmax and gather
    __aicore__ inline void ArgMaxAndGather(uint32_t calCnt, uint32_t& maxIndex)
    {
        ReduceMax(sampleLogitsLocal, sampleLogitsLocal, sampleLogitsLocal, calCnt, true);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float maxIndexSrc = sampleLogitsLocal.GetValue(1);
        maxIndex = *reinterpret_cast<uint32_t*>(&maxIndexSrc);
    }

    template <typename GDTYPE>
    __aicore__ inline void CopyOut(int32_t rowId, uint32_t calCnt, GDTYPE& dist, uint32_t maxIndex)
    {   
        LocalTensor<uint64_t> tempLocalCopy = buf0.Get<uint64_t>();
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(srcIndexGlobalUser[rowId * this->rowLen]);
        this->outputIdx = static_cast<uint64_t>(dist.GetValue(maxIndex));
        tempLocalCopy.SetValue(0, this->outputIdx);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyIndexOutParams{1, (uint32_t)sizeof(uint64_t), 0, 0, 0};
        DataCopyPad(dstIndexGlobal[rowId], tempLocalCopy, copyIndexOutParams);
        // gather the scatter result of selected logits
        if (this->isNeedLogits) {
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            uint32_t startIndex = rowId * this->rowLen;
            uint32_t maxNum = GM_COPY_PER_FLOAT_MAX; // 65535/4=16383
            uint32_t rowLenOri = this->rowLen;
            LocalTensor<T> localValue = buf0.Get<T>();
            LocalTensor<float> localValueCast = buf2.Get<float>();
            LocalTensor<uint32_t> indexValue = buf4.Get<uint32_t>();
            LocalTensor<float> topKPLogitsLocal = buf0.Get<float>();
            DataCopyExtParams copyLogitsOutParams{1, (uint32_t)sizeof(float), 0, 0, 0};
            uint32_t startOffset = 0;

            while (rowLenOri > 0) {
                auto dataLen = rowLenOri > maxNum ? maxNum : rowLenOri;
                DataCopyPad(
                    localValue, logitsGlobal[startIndex + startOffset],
                    {1, static_cast<uint16_t>(sizeof(T) * dataLen), 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(localValueCast, localValue, RoundMode::CAST_NONE, dataLen);
                uint32_t calCntOri = calCnt;
                uint32_t indexStartOffset = 0;
                while (calCntOri > 0) {
                    auto indexDataLen = calCntOri > maxNum ? maxNum : calCntOri;
                    if constexpr (std::is_same<GDTYPE, LocalTensor<uint32_t>>::value) {
                        DataCopy(indexValue, dist[indexStartOffset], this->k_pad);  // TOPP_K_FIX
                        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    } else {
                        DataCopyPad(
                            indexValue, dist[indexStartOffset],
                            {1, static_cast<uint16_t>(SIZEOF_UINT32 * indexDataLen), 0, 0}, {false, 0, 0, 0});
                        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    }
                    for (uint32_t num = 0; num < indexDataLen; ++num) {
                        auto index = indexValue.GetValue(num);
                        int32_t valIdx = index - startOffset;   // get local index in current global index section
                        if (valIdx < dataLen && valIdx >= 0) {
                            uint32_t offset = startIndex + index;   // adding origin column index on to GM offset of current core batch for dstlogitsIdx  
                            auto val = localValueCast.GetValue(valIdx);
                            topKPLogitsLocal.SetValue(0, val);
                            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                            DataCopyPad(dstLogitsGlobal[offset], topKPLogitsLocal, copyLogitsOutParams);
                        }
                    }
                    calCntOri -= indexDataLen;
                    indexStartOffset += indexDataLen;
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                }
                rowLenOri -= dataLen;
                startOffset += dataLen;
            }
        }
    }

    __aicore__ inline void CopyOut(int32_t rowId, uint32_t copyCnt, uint32_t maxIndex, bool ifGetIndex, bool ifCopyOutValue)
    {
        // copyOut
        LocalTensor<uint64_t> sampleIndexLocal = buf4.Get<uint64_t>();
        if (ifGetIndex)
            this->outputIdx = static_cast<uint64_t>(localIndexRs.GetValue(maxIndex));
        else
            this->outputIdx = static_cast<uint64_t>(maxIndex);
        sampleIndexLocal.SetValue(0, this->outputIdx);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyIndexOutParams{1, (uint32_t)sizeof(uint64_t), 0, 0, 0};
        DataCopyPad(dstIndexGlobal[rowId], sampleIndexLocal, copyIndexOutParams);
        // copy out scatter result of the selected logits
        if (ifCopyOutValue) {
            uint32_t startIndex = rowId * this->rowLen;
            LocalTensor<float> topKPLogitsLocal = buf5.Get<float>();
            DataCopy(topKPLogitsLocal, localValueRs, copyCnt);
            DataCopyExtParams copyLogitsOutParams{1, (uint32_t)sizeof(float), 0, 0, 0};
            for (uint32_t num = 0; num < this->topPNum; num++) { // 排好序的前topPNum个数，分别拷贝到dstLogitsGlobal中，offset保证放置的位置和源Logits位置一致，未放置位置数值为0？
                topKPLogitsLocal.SetValue(0, topKPLogitsLocal.GetValue(num));
                uint32_t offset = startIndex + localIndexRs.GetValue(num);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                // Refresh dst GM cache before read-Scalar to write-MTE.
                DataCopyPad(dstLogitsGlobal[offset], topKPLogitsLocal, copyLogitsOutParams);
            }
        }
    }

private:
    TPipe* pipe;

    TBuf<TPosition::VECCALC> buf0; // 32k
    TBuf<TPosition::VECCALC> buf1; // 32k
    TBuf<TPosition::VECCALC> buf2; // 32k
    TBuf<TPosition::VECCALC> buf3; // 32k
    TBuf<TPosition::VECCALC> buf4; // 32k
    TBuf<TPosition::VECCALC> buf5; // 32k

    GlobalTensor<T> logitsGlobal;
    GlobalTensor<float> dstLogitsGlobal;
    GlobalTensor<uint64_t> dstIndexGlobal;
    GlobalTensor<uint32_t> topKGlobal;
    GlobalTensor<T> topPGlobal;
    GlobalTensor<float> qGlobal;

    GlobalTensor<float> logitsGlobalUser;
    GlobalTensor<float> sortPartGlobalUser;
    GlobalTensor<float> sortAllGlobalUser;
    GlobalTensor<uint32_t> srcIndexGlobalUser;

    LocalTensor<float> localValueRs;
    LocalTensor<uint32_t> localIndexRs;
    LocalTensor<float> sampleLogitsLocal;

    uint32_t rowNum{0};
    uint32_t rowLen{0};
    uint32_t headCoreNum{0};
    uint32_t innerLoopEle{0};
    uint32_t innerLoopTime{0};
    uint32_t innerLoopEleTail{0};
    uint32_t innerLoopEleTailPad{0};

    uint32_t softmaxLoopTime{0};
    uint32_t softmaxLoopEleTail{0};
    uint32_t softmaxLoopEleTailPad{0};

    uint32_t startTaskId{0};    // start batch index offest of current kernel
    uint32_t rtCoreRowNum{0};   // batch amount processed by this kernel
    uint32_t k{0};
    uint32_t k_pad{0};
    uint32_t topPNum{0};
    uint32_t outputIdx{0};
    float eps{.0};
    uint32_t mrgMode{0};
    T p{0};
    uint32_t isNeedLogits = 0;
    uint32_t topKGuess{TOPP_K_FIX};
    bool ifQSampleCompute = false;
    uint32_t queLenGlobal[NUM_TWO]{0};
    // rowId*List here ought to be local index instead of global batch index
    uint32_t rowIdToppList[ROW_LEN_MAX]{0};
    uint32_t rowIdTopkList[ROW_LEN_MAX]{0};
    uint32_t ifFind{0};

    const float* FP32_NEG_INF_PTR = reinterpret_cast<const float*>(&FP32_NEG_INF_BITS);
    const float SEL_LOGITS_DEF_VAL = *FP32_NEG_INF_PTR;
}; // namespace TopKTopPSample
};
#endif // TOP_K_TOP_P_SAMPLE_H