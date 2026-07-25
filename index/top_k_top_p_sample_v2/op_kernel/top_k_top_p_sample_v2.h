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
 * \file top_k_top_p_sample_v2.h
 * \brief
 */

#ifndef TOP_K_TOP_P_SAMPLE_V2_H
#define TOP_K_TOP_P_SAMPLE_V2_H

#include "top_k_top_p_sample_v2_comm.h"
#include "top_k_top_p_sample_v2_sort_cumsum.h"
#if __NPU_ARCH__ == 3510
#include "c_api/asc_simd.h"
#endif

namespace TopKTopPSampleV2 {

constexpr uint32_t INNER_LOOP_ELE = 8 * 1024;
constexpr uint32_t TOPP_K_FIX = 32;
constexpr uint32_t TOPK_MAX = 1 * 1024;
constexpr uint32_t SORT32_PER_LEN = 2 * 1024;
constexpr uint32_t SORT32_MAX_LOOP = INNER_LOOP_ELE / SORT32_PER_LEN;
constexpr uint32_t MRGSORT_PER_NUM = 4;
constexpr uint32_t MRGSORT_PER_NUM_LAST = 2;
constexpr uint32_t SORT32_PERGROUP_LEN = 2 * TOPK_MAX;
constexpr float TOPP_MAX = 1.0f;
constexpr uint32_t FP32_NEG_INF_BITS = 0b11111111100000000000000000000000;

template <typename T>
class TopKTopPSampleV2Kernel {
public:
    TOPKPParams params{};
    TopKTopPSampleV2SortKernel<T> sortOp;

    __aicore__ inline TopKTopPSampleV2Kernel(){};
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR minPs,
                                GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKPSelect, GM_ADDR logitsIdx,
                                GM_ADDR logitsSortMasked, GM_ADDR workspace,
                                const TopKTopPSampleV2TilingData tilingData, TPipe* tPipe)
    {
        if ASCEND_IS_AIV {
            InitTilingParams(tilingData);
            InitAndSetBuffer(logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKPSelect, logitsIdx,
                             logitsSortMasked, workspace, tPipe);
        }
    }

    __aicore__ inline void Process()
    {
        if (g_coreType == AscendC::AIC) {
            return;
        }

        for (int32_t processRowCount = 0; processRowCount < this->rtCoreRowNum; processRowCount++) {
            uint32_t rowId = processRowCount;
            int32_t kCount = 0;
            float oriMaxValue = 0.0f;
            LocalTensor<T> topPtempLocal = buf2.Get<T>();
            LocalTensor<T> minPtempLocal = buf4.Get<T>();
            LocalTensor<int32_t> topKtempLocal = buf5.Get<int32_t>();
            DataCopyPad(topPtempLocal, topPGlobal[rowId], {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            this->p = topPtempLocal.GetValue(0);
            float fp32P = 0;
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                fp32P = ToFloat(this->p);
            } else {
                fp32P = static_cast<float>(this->p);
            }
            DataCopyPad(topKtempLocal, topKGlobal[rowId], {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            this->k = topKtempLocal.GetValue(0);
            bool hasMinP = false;
            float fp32MinP{0.0f};
            if (this->isInputMinPs) {
                DataCopyPad(minPtempLocal, minPGlobal[rowId], {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                this->minP = minPtempLocal.GetValue(0);
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    fp32MinP = ToFloat(this->minP);
                } else {
                    fp32MinP = static_cast<float>(this->minP);
                }
                hasMinP = fp32MinP > 0 ? true : false;
            }
            params.topp = fp32P;
            this->ifFind = 0;
            this->topPNum = 0;
            uint32_t maxIndex{0};
            float maxValue{FLOAT_MIN};
            params.softmaxLoopTime = this->softmaxLoopTime;
            params.softmaxLoopEleTail = this->softmaxLoopEleTail;
            params.softmaxLoopEleTailPad = this->softmaxLoopEleTailPad;
            const bool doTopKSample = (this->k > 0 && this->k <= AscendC::Std::min(this->rowLen, this->ksMAX));
            const bool doTopPSample = (fp32P > 0 && fp32P < TOPP_MAX);
            noTopKPMinP = (!doTopKSample && !doTopPSample && !hasMinP);
            const bool doGuessTopK = !doTopKSample && doTopPSample;
            bool isInLocalTensor = false;
            hasCopyOutLogits = false;
#if __NPU_ARCH__ == 3510
            asc_dcci_entire_out();
#endif

            if (doTopKSample || fp32P <= 0) {
                kCount = this->k;
                if (fp32P <= 0) {
                    kCount = 1;
                }
                this->topPNum = kCount;
                this->k_pad = Align(kCount, EIGHT); // logits转成float32后，需满足32字节对齐，k值必须为8的整倍数
                TopKLoopProcess(
                    rowId, kCount,
                    logitsGlobal); // 排序且取前k个，logits类型转换成float32，存在了localValueRs（buf0）和localIndexRs（buf1）中
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                maxValue = localValueRs.GetValue(static_cast<uint32_t>(0));
                maxIndex = localIndexRs.GetValue(static_cast<uint32_t>(0));
                oriMaxValue = maxValue;
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                isInLocalTensor = true;
            }

            if (this->inputIsLogits) {
                if (doTopKSample || fp32P <= 0) {
                    SoftMaxCompute(kCount, localValueRs, maxValue); // softmax后结果保存在sampleLogitsLocal（buf3）
                    maxValue = sampleLogitsLocal.GetValue(static_cast<uint32_t>(0));
                } else if (!doTopPSample) {
                    // 从logitsGlobal copyin进来，cast完后，SoftMax一整行，存放在logitsGlobalUser里
                    float rowMax{FLOAT_MIN};
                    float reduceSumMax{0};
                    GetRowMax(rowId, &rowMax, logitsGlobal); // 将cast后未做排序的logits存入logitsGlobalUser，更新rowMax
                    oriMaxValue = rowMax;
                    SoftMaxFstCompute(rowId, rowMax, &reduceSumMax,
                                      true); // 将exp(x - max)结果存储到logitsGlobalUser里，更新reduceSumMax
                    SoftMaxFullCompute(rowId, reduceSumMax, maxIndex); // 将softmax结果存储到logitsGlobalUser里
                    if (noTopKPMinP && this->ifQSampleCompute) {
                        LocalTensor<uint64_t> sampleIndexLocal = buf4.Get<uint64_t>();
                        sampleIndexLocal.SetValue(0, maxIndex);
                        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                        DataCopyExtParams copyIndexOutParams{1, (uint32_t)sizeof(uint64_t), 0, 0, 0};
                        DataCopyPad(dstIndexGlobal[rowId], sampleIndexLocal, copyIndexOutParams);
                        continue;
                    }
                } else {
                    LogitsCast(rowId);
                }
            } else {
                if (doTopKSample || fp32P <= 0) {
                    // localValueRs前kCount个数据copy到samplelogitslocal
                    PipeBarrier<PIPE_V>();
                    DataCopy(sampleLogitsLocal, localValueRs, this->k_pad);
                    PipeBarrier<PIPE_V>();
                } else {
                    // 从logitsGlobal copyin进来，cast完后copyout到logitsGlobalUser里
                    LogitsCast(rowId);
                }
            }

            if (doTopKSample && doTopPSample) {                // 做topktopp
                TopPCompute(sampleLogitsLocal, kCount, fp32P); // 拿到toppNum,方便后续sample
                maxValue = sampleLogitsLocal.GetValue(0);
                maxIndex = localIndexRs.GetValue(0);
                isInLocalTensor = true;
            } else if (doGuessTopK) {
                kCount = this->topKGuess > this->rowLen ? this->rowLen : this->topKGuess;
                this->k_pad = Align(kCount, EIGHT);
                if (kCount > 0) {
                    TopKLoopProcess(rowId, kCount, logitsGlobalUser);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    if (this->inputIsLogits) {
                        maxValue = localValueRs.GetValue(0);
                        oriMaxValue = maxValue;
                        SetFlag<HardEvent::S_V>(EVENT_ID0);
                        WaitFlag<HardEvent::S_V>(EVENT_ID0);

                        float reduceSumMax{0};
                        SoftMaxFstCompute(rowId, maxValue, &reduceSumMax, false);
                        SoftMaxComputeLocal(kCount, localValueRs, maxValue, reduceSumMax);

                        SetFlag<HardEvent::V_S>(EVENT_ID0);
                        WaitFlag<HardEvent::V_S>(EVENT_ID0);
                        TopPCompute(sampleLogitsLocal, kCount, fp32P);
                        maxValue = sampleLogitsLocal.GetValue(0);
                    } else {
                        TopPCompute(localValueRs, kCount, fp32P);
                        maxValue = localValueRs.GetValue(0);
                        oriMaxValue = maxValue;
                        DataCopy(sampleLogitsLocal, localValueRs, this->k_pad);
                        PipeBarrier<PIPE_V>();
                    }
                    maxIndex = localIndexRs.GetValue(0);
                    isInLocalTensor = true;
                }
                if (!this->ifFind) { // GuessKFailed: 没猜到
                    isInLocalTensor = false;
                    params.tensor0 = buf0.Get<float>();
                    params.tensor1 = buf5.Get<float>();
                    params.rowNum = this->rowNum;
                    params.rowLen = this->rowLen;
                    params.rowId = rowId;
                    params.sortOnly = false; // 主链路走 softmax/cumsum，显式置 false（勿依赖零初始化）

                    float rowMax{FLOAT_MIN};
                    float reduceSumMax{0};
                    GetRowMax(rowId, &rowMax, logitsGlobal); // 将cast后未做排序的logits存入logitsGlobalUser，更新rowMax
                    oriMaxValue = rowMax;
                    SoftMaxFstCompute(rowId, rowMax, &reduceSumMax,
                                      false); // 将exp(x - max)结果存储到logitsGlobalUser里，更新reduceSumMax
                    params.inputIsLogits = this->inputIsLogits;
                    params.reduceSumMax = reduceSumMax;
                    params.rowMax = rowMax;
                    // 全排序，拿到排序结果，也会做topP，拿到toppNum
                    sortOp.SortAll(
                        params, logitsGlobalUser, sortPartGlobalUser,
                        sortAllGlobalUser, // 从logitsGlobalUser取做了softmax但是没排序的值做sort，然后求topPNum，sort的value结果存在了logitsGlobalUser中，index结果放在了srcIndexGlobalUser中
                        srcIndexGlobalUser);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                    maxValue = logitsGlobalUser.GetValue(rowId * this->rowLen);
                    maxIndex = srcIndexGlobalUser.GetValue(rowId * this->rowLen);
                    this->topPNum = params.toppNum;
                }
            } else if (!doTopKSample && fp32P >= TOPP_MAX) {
                // 求一行最大值，赋值maxValue
                // 生成下标，存放到srcIndexGlobalUser
                isInLocalTensor = false;
                this->topPNum = rowLen;
                GetRowMaxandIndex(rowId, &maxValue, maxIndex);
                if (!this->inputIsLogits) {
                    oriMaxValue = maxValue;
                }
            }

            while (hasMinP) {
                float minPThd = fp32MinP * maxValue;
                if (fp32MinP >= 1.0f) {
                    localIndexRs.SetValue(0, maxIndex);
                    localValueRs.SetValue(0, oriMaxValue);
                    sampleLogitsLocal.SetValue(0, maxValue);
                    this->topPNum = 1;
                    isInLocalTensor = true;
                    break;
                }
                if (isInLocalTensor) {
                    // 从samplelogitsLocal取出value，localIndexRs取出index，用gathermask取出有效logits，存回samplelogitsLocal和localIndexRs
                    minPComputeLocal(sampleLogitsLocal, localIndexRs, minPThd);
                } else {
                    // 从logitsGlobalUser取出value，srcIndexGlobalUser取出index，用gathermask取出有效logits，存回logitsGlobalUser和srcIndexGlobalUser
                    minPComputeGloabal(rowId, logitsGlobalUser, srcIndexGlobalUser, minPThd);
                }
                break;
            }

            // 本地路径(isInLocalTensor)的 logits 回填依赖片上 buf0(localValueRs)/buf1(localIndexRs)，
            // 而下面 QSample/CopyOutIndex 块会覆写 buf0 —— 故本地 logits 必须在该块之前完成（原始次序）。
            if (this->isNeedLogits && isInLocalTensor) {
                // 占比门控：选中占比 >70% 时逐元素散点的标量/MTE3 小包开销高，改走片上排序+连续段批量；
                // 否则维持原 CopyOutLogits 散点回填，保证低占比(稀疏 topk)场景不劣化。
                if (this->topPNum * 100 >= this->rowLen * 70) {
                    CopyOutLogitsLocalOptimized(rowId, this->topPNum);
                } else {
                    // 根据localIndevRs和toppnum，logitsGlobal cast成float后 copyout
                    CopyOutLogits(rowId, this->topPNum);
                }
            }

            // sample_result / QSample / index 输出：!isInLocalTensor 下是 logitsGlobalUser、
            // srcIndexGlobalUser 的最后消费者。做完后这两块 workspace region 即空闲，供下面
            // 非本地路径 CopyOutLogitsOptimized 复用为排序 scratch（不再额外开辟 idxVal/idxPos region）。
            if (this->isNeedSampleResult) {
                /* 将logitsGlobalUser/samplelogitslocal srcIndexGlobalUser/localIndexRs数据写回GM */
                CopyOutMiddle(rowId, this->topPNum, isInLocalTensor);
            } else if (!this->ifQSampleCompute) {
                // 使用maxIndex作为输出，copyout
                CopyOutIndex(rowId, maxIndex);
            } else if (this->ifQSampleCompute) {
                if (isInLocalTensor) {
                    QSampleCompute(rowId, Align(this->topPNum, SIZEOF_FP32), this->topPNum);
                    ArgMaxAndGather(this->topPNum, maxIndex);
                } else {
                    QSampleGlobal(rowId, maxIndex);
                }
                CopyOutIndex(rowId, maxIndex);
            }

            // 非本地路径(!isInLocalTensor)的 logits 回填从 GM 重新装载，不依赖上面被覆写的片上 buf；
            // 且优化分支需复用已空闲的 logitsGlobalUser 作排序 scratch —— 故放在 QSample/index 块之后。
            if (this->isNeedLogits && !isInLocalTensor) {
                // 根据srcIndexGlobalUser和toppnum，logitsGlobal cast成float后 copyout
                if (!hasCopyOutLogits) {
                    // 占比门控：选中占比 > 60% 时散点搬运的标量空转/MTE3 小包开销高，
                    // 改走连续段批量搬运；否则维持原散点回填。
                    if (this->topPNum * 100 >= this->rowLen * 60) {
                        CopyOutLogitsOptimized(rowId, this->topPNum);
                    } else {
                        CopyOutLogitsGlobal(rowId, this->topPNum, srcIndexGlobalUser);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTilingParams(const TopKTopPSampleV2TilingData tilingData)
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
        this->ksMAX = tilingData.ksMAX;
        this->inputIsLogits = tilingData.inputIsLogits;
        this->isNeedSampleResult = tilingData.isNeedSampleResult;
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

    __aicore__ inline void InitAndSetBuffer(GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR minPs,
                                            GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKPSelect, GM_ADDR logitsIdx,
                                            GM_ADDR logitsSortMasked, GM_ADDR workspace, TPipe* tPipe)
    {
        // InitGlobalBuffer as '(type *)<ADDR_OFFSET>' where <ELEMENT AMOUNT> is not recommended.
        uint64_t gmOffset = static_cast<uint64_t>(this->startTaskId) *
                            this->rowLen; // element offset by row-offset * col_size (B_offset * S)

        // Read-only tensors
        logitsGlobal.SetGlobalBuffer((__gm__ T*)logits + gmOffset);             // logits in [B, S]
        topKGlobal.SetGlobalBuffer((__gm__ int32_t*)topKs + this->startTaskId); // top_k in [B, ]
        topPGlobal.SetGlobalBuffer((__gm__ T*)topPs + this->startTaskId);       // top_p in [B, ]

        uint32_t coreEle_ = this->rtCoreRowNum == 0 ? this->rowLen : this->rtCoreRowNum * this->rowLen;
        // Write tensors shall aligned to their batch offset
        dstLogitsGlobal.SetGlobalBuffer((__gm__ float*)logitsTopKPSelect + gmOffset,
                                        coreEle_); // logits_top_kp_select in [B, S]
        dstIndexGlobal.SetGlobalBuffer((__gm__ uint64_t*)logitsSelectIdx +
                                       this->startTaskId); // logits_select_idx in [B, ]
        dstLogitsSortMaskedGlobal.SetGlobalBuffer((__gm__ float*)logitsSortMasked + gmOffset,
                                                  coreEle_); // logits_sort_masked in [B, S]
        dstLogitsIdxGlobal.SetGlobalBuffer((__gm__ uint64_t*)logitsIdx + gmOffset, coreEle_); // logits_idx in [B, S]

        if (q != nullptr) {
            qGlobal.SetGlobalBuffer((__gm__ float*)q + gmOffset); // q in [B, S]
            this->ifQSampleCompute = true;
        } else {
            this->ifQSampleCompute = false;
        }

        if (minPs != nullptr) {
            minPGlobal.SetGlobalBuffer((__gm__ T*)minPs + this->startTaskId);
            this->isInputMinPs = true;
        } else {
            this->isInputMinPs = false;
        }

        // the workspace seems to be a shared cache for intermediate results from the GM data
        uint64_t count = static_cast<uint64_t>(this->rowNum) * this->rowLen;
        uint64_t offset = 0;             // offsets for different cache blocks shared by ALL kernels
        uint64_t wsKernelOfs = gmOffset; // Kernel addr offset in current workspace block

        // If there is NO inter-kernel communications, local workspace shall start from the GM offset of current core.
        logitsGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs); // global Logits workspace in [B, S]
        offset += count;
        wsKernelOfs *= MRG_PER_ELE; // Kernel Addr offset in workspace block is sized by column factor for ONLY ONCE
        sortPartGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs + offset);
        offset += count * MRG_PER_ELE;
        sortAllGlobalUser.SetGlobalBuffer((__gm__ float*)workspace + wsKernelOfs + offset);
        offset += count * MRG_PER_ELE;
        srcIndexGlobalUser.SetGlobalBuffer((__gm__ uint32_t*)workspace + gmOffset + offset);

        // CopyOutLogitsOptimized 连续段回填复用 SortAll 排序列索引：
        // 不再额外开辟 scratch —— srcGlobal 复用 logitsGlobalUser（本行 sample_result/QSample 已消费完），
        // 归并 ping-pong 复用 sortPartGlobalUser/sortAllGlobalUser，sortIndex 用 srcIndexGlobalUser 占位
        // （sortOnly 下从不写）。workspace 保持 6×count。

        // Init top_kp_selected_logits GM to zeros when if isNeedLogits
        if (this->isNeedLogits) {
            if (this->inputIsLogits) {
                AscendC::InitGlobalMemory(dstLogitsGlobal, this->rtCoreRowNum * this->rowLen,
                                          static_cast<float>(SEL_LOGITS_DEF_VAL));
            } else {
                AscendC::InitGlobalMemory(dstLogitsGlobal, this->rtCoreRowNum * this->rowLen, static_cast<float>(0.0f));
            }
            // Sync single core I/O steps
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            // Prevent dirty write under multi-core scenarios
            SyncAll();
        }

        if (this->isNeedSampleResult) {
            AscendC::InitGlobalMemory(dstLogitsSortMaskedGlobal, this->rtCoreRowNum * this->rowLen,
                                      static_cast<float>(0.0f));
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
    }

    __aicore__ inline void LogitsCast(int32_t rowId)
    {
        LocalTensor<T> localValue = buf0.Get<T>();
        LocalTensor<float> localValueCast = buf3.Get<float>();

        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t innerCopyLen = INNER_LOOP_ELE; // 8 * 1024
        uint32_t countLen = INNER_LOOP_ELE;
        uint64_t gmOffset = startIndex;

        for (uint32_t innerLoopCount = 0; innerLoopCount < this->innerLoopTime; ++innerLoopCount) {
            if (this->innerLoopEleTail > 0 && innerLoopCount == this->innerLoopTime - 1) {
                innerCopyLen = this->innerLoopEleTailPad;
                countLen = this->innerLoopEleTail;
            }
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(localValueCast, logitsGlobal[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            } else {
                DataCopyPad(localValue, logitsGlobal[gmOffset], {1, (uint32_t)(countLen * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(localValueCast, localValue, RoundMode::CAST_NONE, countLen); // half/bfloat16 -> float32
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            }
            DataCopyPad(logitsGlobalUser[gmOffset], localValueCast, {1, (uint32_t)(countLen * SIZEOF_FP32), 0, 0, 0});
            if (noTopKPMinP && this->isNeedLogits) {
                DataCopyPad(dstLogitsGlobal[gmOffset], localValueCast,
                            {1, (uint32_t)(countLen * SIZEOF_FP32), 0, 0, 0});
                hasCopyOutLogits = true;
            }
            gmOffset += innerCopyLen;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }

    __aicore__ inline void GetRowMaxAndIndexInner(LocalTensor<float>& src, LocalTensor<float>& reduceMaxMiddle,
                                                  uint32_t countLen, float* rowMax, uint32_t& rowMaxIdx,
                                                  uint32_t indexOffset, const GlobalTensor<float>& dist)
    {
        ReduceMax(reduceMaxMiddle, src, reduceMaxMiddle, countLen, true);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto temp = reduceMaxMiddle.GetValue(0);
        if (temp > *rowMax) {
            float index = reduceMaxMiddle.GetValue(1);
            auto tempIndex = *reinterpret_cast<uint32_t*>(&index);
            rowMaxIdx = indexOffset + tempIndex;
        }
        *rowMax = *rowMax > temp ? *rowMax : temp;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void GetRowMaxandIndex(uint32_t rowId, float* rowMax, uint32_t& rowMaxIdx)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN;
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t indexOffset = 0;
        LocalTensor<uint32_t> localIndex = buf1.Get<uint32_t>();
        LocalTensor<float> valueLocalCast = buf2.Get<float>();
        LocalTensor<float> reduceMaxMiddle = buf4.Get<float>();
        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; innerLoopCount++) {
            uint64_t gmOffset = startIndex + innerLoopCount * innerCopyLen;
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            CreateVecIndex(localIndex[NUM_ZERO].ReinterpretCast<int32_t>(), static_cast<int32_t>(indexOffset),
                           countLen);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(srcIndexGlobalUser[gmOffset], localIndex,
                        {1, static_cast<uint32_t>(countLen * SIZEOF_FP32), 0, 0, 0});
            DataCopyPad(valueLocalCast, logitsGlobalUser[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            GetRowMaxAndIndexInner(valueLocalCast, reduceMaxMiddle, countLen, rowMax, rowMaxIdx, indexOffset,
                                   logitsGlobalUser[gmOffset]);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            indexOffset += countLen;
            gmOffset += innerCopyLen;
        }
    }

    __aicore__ inline void minPComputeLocal(LocalTensor<float>& samplelogitsLocal, LocalTensor<uint32_t>& localIndexRs,
                                            float& minPThd)
    {
        for (int32_t index = this->topPNum - 1; index >= 0; index--) {
            float samplelogitsLocalValue = samplelogitsLocal.GetValue(index);
            if (samplelogitsLocalValue >= minPThd) {
                this->topPNum = index + 1;
                break;
            }
        }
    }

    __aicore__ inline void minPComputeGloabal(uint32_t rowId, GlobalTensor<float>& logitsGlobalUser,
                                              GlobalTensor<uint32_t>& srcIndexGlobalUser, float& minPThd)
    {
        uint32_t countLen = INNER_LOOP_ELE;
        uint32_t countLenPad = INNER_LOOP_ELE;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint64_t gmOffset = 0;
        uint32_t indexOffset = 0;
        uint32_t copyOutOffset = 0;
        uint64_t rsvdCnt = 0;
        // 根据topPNum确定循环次数
        auto minPLoopNum = SafeCeil(this->topPNum, INNER_LOOP_ELE);
        auto minPLoopTailNum = this->topPNum % INNER_LOOP_ELE;
        LocalTensor<float> logitsLocal = buf0.Get<float>();
        LocalTensor<uint32_t> srcIndexLocal = buf1.Get<uint32_t>();
        LocalTensor<uint8_t> compareIndexMask = buf5.Get<uint8_t>();

        for (uint32_t minPLoopCount = 0; minPLoopCount < minPLoopNum; minPLoopCount++) {
            if (minPLoopTailNum > 0 && minPLoopCount == (minPLoopNum - 1)) {
                countLen = minPLoopTailNum;
                countLenPad = Align(countLen, EIGHT);
            }
            // 搬运至UB
            DataCopyPad(logitsLocal, logitsGlobalUser[startIndex + gmOffset],
                        {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(srcIndexLocal, srcIndexGlobalUser[startIndex + indexOffset],
                        {1, (uint32_t)(countLen * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            // 与minp比较
            GatherMaskParams gatherMaskParams{1, 1, 8, 8};
            CompareScalar(compareIndexMask, logitsLocal, minPThd, CMPMODE::GE,
                          SafeCeil(countLen, SIXTY_FOUR) * SIXTY_FOUR); // 大于等于，256字节对齐
            PipeBarrier<PIPE_V>();

            // 取出
            GatherMask(logitsLocal, logitsLocal, compareIndexMask.ReinterpretCast<uint32_t>(), true, countLen,
                       gatherMaskParams, rsvdCnt);
            GatherMask(srcIndexLocal, srcIndexLocal, compareIndexMask.ReinterpretCast<uint32_t>(), true, countLen,
                       gatherMaskParams, rsvdCnt);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            // copy to workspace, 由rsvdCnt更新偏移
            DataCopyExtParams valueCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>((rsvdCnt) * sizeof(float)), 0, 0, 0};
            DataCopyExtParams indexCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>((rsvdCnt) * sizeof(uint32_t)), 0, 0, 0};
            DataCopyPad(logitsGlobalUser[startIndex + copyOutOffset], logitsLocal, valueCopyParams);
            DataCopyPad(srcIndexGlobalUser[startIndex + copyOutOffset], srcIndexLocal, indexCopyParams);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            // 更新偏移
            gmOffset += countLen;
            indexOffset += countLen;
            copyOutOffset += rsvdCnt;
        }

        this->topPNum = copyOutOffset; // 最后的copyOutOffset等于有效logits的数量
    }

    __aicore__ inline void GetRowMaxInner(LocalTensor<float>& src, LocalTensor<float>& reduceMaxMiddle,
                                          uint32_t countLen, float* rowMax, const GlobalTensor<float>& dist)
    {
        ReduceMax(reduceMaxMiddle, src, reduceMaxMiddle, countLen, false);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto temp = reduceMaxMiddle.GetValue(0);
        *rowMax = *rowMax > temp ? *rowMax : temp;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        DataCopyPad(dist, src,
                    {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0}); // 将cast后未做排序的logits存入globalUser
    }

    template <typename DTYPE>
    __aicore__ inline void GetRowMax(uint32_t rowId, float* rowMax, const GlobalTensor<DTYPE>& src)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN;
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        LocalTensor<T> valueLocal = buf0.Get<T>();
        LocalTensor<float> valueLocalCast = buf2.Get<float>();
        LocalTensor<float> reduceMaxMiddle = buf4.Get<float>();
        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; innerLoopCount++) {
            uint64_t gmOffset = startIndex + innerLoopCount * innerCopyLen;
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            if constexpr (std::is_same<DTYPE, float>::value) {
                DataCopyPad(valueLocalCast, src[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                DataCopyPad(valueLocal, src[gmOffset], {1, (uint32_t)(countLen * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(valueLocalCast, valueLocal, RoundMode::CAST_NONE, countLen);
                PipeBarrier<PIPE_V>();
            }
            GetRowMaxInner(valueLocalCast, reduceMaxMiddle, countLen, rowMax, logitsGlobalUser[gmOffset]);
            if (noTopKPMinP && this->isNeedLogits) {
                DataCopyPad(dstLogitsGlobal[gmOffset], valueLocalCast,
                            {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0});
                hasCopyOutLogits = true;
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void SoftMaxFstCompute(uint32_t rowId, float rowMax, float* reduceSumMax, bool needCopyOut)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN; // 8 * 1024 * 2 个元素，每个元素2字节，满足32k
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        LocalTensor<float> valueLocalCast = buf3.Get<float>();
        LocalTensor<float> valueLocalMiddle = buf2.Get<float>();
        LocalTensor<float> reduceSumVal = buf4.Get<float>();

        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; innerLoopCount++) {
            uint64_t gmOffset = startIndex + innerLoopCount * innerCopyLen;
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            DataCopyPad(valueLocalCast, logitsGlobalUser[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0}); // 从globalUser里将GetRowMax函数中存储的未排序的做了cast的logits拿出来
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
            if (needCopyOut) {
                DataCopyPad(logitsGlobalUser[gmOffset], valueLocalMiddle,
                            {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0});
                // 将EXP结果存储到globalUser里
            }

            ReduceSum(reduceSumVal, valueLocalMiddle, reduceSumVal, countLen); // 基础API
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            *reduceSumMax = *reduceSumMax + reduceSumVal.GetValue(0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void SoftMaxSecCompute( // 做softmax中的exp/S的步骤
        LocalTensor<float>& valueLocalCast, float reduceSumMax, uint64_t gmOffset, uint32_t innerCopyLen,
        uint32_t countLen)
    {
        LocalTensor<float> valueRs = buf0.Get<float>();
        DataCopyPad(valueRs, logitsGlobalUser[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0}); // 将softmaxFst中算的exp临时存储在了GM中，现在搬到local中
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(valueLocalCast, reduceSumMax, countLen);
        PipeBarrier<PIPE_V>();
        Div(valueLocalCast, valueRs, valueLocalCast, countLen); // softmax计算结果存储在valueLocalCast中
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftMaxFullCompute(uint32_t rowId, float reduceSumMax, uint32_t& maxIndextemp)
    {
        uint32_t innerCopyLen = INNER_LOOP_ELE;
        uint32_t countLen = INNER_LOOP_ELE;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t indexOffset = 0;
        float maxValueTemp{FLOAT_MIN};

        LocalTensor<float> localValueCast = buf3.Get<float>();
        LocalTensor<float> sampleDistTemp = buf4.Get<float>();
        uint64_t gmOffset = startIndex;
        for (int32_t innerLoopCount = 0; innerLoopCount < this->innerLoopTime; ++innerLoopCount) {
            if (this->innerLoopEleTail > 0 && innerLoopCount == this->innerLoopTime - 1) {
                // tail block
                innerCopyLen = this->innerLoopEleTailPad;
                countLen = this->innerLoopEleTail;
            }
            SoftMaxSecCompute(localValueCast, reduceSumMax, gmOffset, innerCopyLen,
                              countLen); // localValueCast存放归一化后的softmax值
            if (!(noTopKPMinP && this->ifQSampleCompute)) {
                SetWaitFlag<HardEvent::V_MTE3>(
                    HardEvent::V_MTE3); // It should be ok to use SetWaitFlag for flows share the same EVENT_ID.
                DataCopyPad(logitsGlobalUser[gmOffset], localValueCast,
                            {1, static_cast<uint32_t>(countLen * SIZEOF_FP32), 0, 0, 0});
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            } else {
                DataCopyPad(sampleDistTemp, qGlobal[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Abs(sampleDistTemp, sampleDistTemp, countLen);
                PipeBarrier<PIPE_V>();
                Adds(sampleDistTemp, sampleDistTemp, this->eps, countLen);
                PipeBarrier<PIPE_V>();
                Div(localValueCast, localValueCast, sampleDistTemp, countLen);
                PipeBarrier<PIPE_V>();
                ReduceMax(localValueCast, localValueCast, localValueCast, countLen, true);
                PipeBarrier<PIPE_V>();
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                float valueLocalFP = localValueCast.GetValue(0);
                float indexLocalFP = localValueCast.GetValue(1);
                auto tempIndex = *reinterpret_cast<uint32_t*>(&indexLocalFP);
                if (valueLocalFP > maxValueTemp) {
                    maxIndextemp = indexOffset + tempIndex;
                }
                maxValueTemp = maxValueTemp > valueLocalFP ? maxValueTemp : valueLocalFP;
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                indexOffset += countLen;
            }
            // 将softmax结果存储到logitsGlobalUser里
            gmOffset += innerCopyLen;
        }
    }

    __aicore__ inline void SortOneTime(LocalTensor<float>&& srcData, LocalTensor<uint32_t>&& srcIndex, uint32_t dataLen,
                                       LocalTensor<float>& sortedLocal)
    {
        LocalTensor<float> concatLocal;
        LocalTensor<float> sortTmpLocal = buf5.Get<float>();
        Concat(concatLocal, srcData, sortTmpLocal, dataLen / SIXTEEN);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sortedLocal, concatLocal, srcIndex.ReinterpretCast<uint32_t>(), sortTmpLocal,
                          dataLen / THIRTY_TWO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Sort32Block(LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex, uint32_t dataLen,
                                       bool mrg)
    {
        auto sort32LoopNum = SafeCeil(dataLen, SORT32_PER_LEN);
        auto sort32LoopTailNum = dataLen % SORT32_PER_LEN;
        auto compareLen = SORT32_PER_LEN;
        auto compareLenPad = SORT32_PER_LEN;
        LocalTensor<float> msgSortPart1 = buf0.Get<float>();
        LocalTensor<float> msgSortPart2 = buf4.Get<float>();
        LocalTensor<float> msgSortLocal[SORT32_MAX_LOOP] = {msgSortPart1, msgSortPart1[SORT32_PER_LEN * NUM_TWO],
                                                            msgSortPart2, msgSortPart2[SORT32_PER_LEN * NUM_TWO]};
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
            // DataCopy demands 32bytes alignment, immediate call using calCount aligned to 32bytes is valid ONLY WHEN
            // msgSortLocal HAS BEEN ALIGNED to 32bytes yet.
            DataCopy(msgSortLocalResultTemp, msgSortLocal[NUM_ZERO],
                     static_cast<uint32_t>(Align(queLen[NUM_ZERO] * NUM_TWO, EIGHT)));
            PipeBarrier<PIPE_V>();
            this->queLenGlobal[qid] = queLen[NUM_ZERO];
        } else {
            this->queLenGlobal[qid] = TOPK_MAX;
            for (uint32_t i = 0; i < sort32LoopNum; i++) {
                elementLengths[i] = queLen[i] > TOPK_MAX ? TOPK_MAX : queLen[i];
            }
            localParams = {elementLengths, false, valueBits[sort32LoopNum], 1};
            MrgSort<float>(
                msgSortLocalResultTemp,
                {msgSortLocal[NUM_ZERO], msgSortLocal[NUM_ONE], msgSortLocal[NUM_TWO], msgSortLocal[NUM_THREE]},
                localParams);
            PipeBarrier<PIPE_V>();
        }

        if (mrg) {
            elementLengths[NUM_ZERO] = static_cast<uint16_t>(this->queLenGlobal[NUM_ZERO]);
            elementLengths[NUM_ONE] = static_cast<uint16_t>(this->queLenGlobal[NUM_ONE]);
            localParams = {elementLengths, false, valueBits[MRGSORT_PER_NUM_LAST], 1};
            MrgSort<float>(msgSortLocalResult[NUM_TWO],
                           {msgSortLocalResult[NUM_ZERO], msgSortLocalResult[NUM_ONE], msgSortLocalResult[NUM_ONE],
                            msgSortLocalResult[NUM_ONE]},
                           localParams);
            PipeBarrier<PIPE_V>();
            DataCopy(msgSortLocalResult[NUM_ZERO], msgSortLocalResult[NUM_TWO], SORT32_PERGROUP_LEN);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Sort32BlockNext(LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex,
                                           uint32_t dataLen, LocalTensor<float>& mrgSortResult, uint32_t kCount)
    {
        uint32_t firstTopkIndex = (kCount - 1) * NUM_TWO;
        uint64_t rsvdCnt = 0;
        uint64_t oriRsvdCnt = 0;
        GatherMaskParams gatherMaskParams{1, 1, 8, 8};
        LocalTensor<uint8_t> compareIndexMask = buf5.Get<uint8_t>();

        float firstTopk = mrgSortResult.GetValue(firstTopkIndex);
        CompareScalar(compareIndexMask, srcData, firstTopk, CMPMODE::GT, SafeCeil(dataLen, SIXTY_FOUR) * SIXTY_FOUR);
        PipeBarrier<PIPE_V>();
        GatherMask(srcData, srcData, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams,
                   rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMask(srcIndex, srcIndex, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams,
                   rsvdCnt);
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

    template <typename DTYPE>
    __aicore__ inline void TopKLoopProcess(int32_t rowId, uint32_t kCount, const GlobalTensor<DTYPE>& src)
    {
        LocalTensor<T> localValue = buf0.Get<T>();
        LocalTensor<uint32_t> localIndex = buf1.Get<uint32_t>(); // 8 * 1024 * 4
        LocalTensor<float> localSortResult = buf2.Get<float>();
        LocalTensor<float> localValueCast = buf3.Get<float>();

        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t indexOffset = 0;
        uint32_t innerCopyLen = INNER_LOOP_ELE; // 8 * 1024
        uint32_t countLen = INNER_LOOP_ELE;
        uint64_t gmOffset = startIndex;

        for (uint32_t innerLoopCount = 0; innerLoopCount < this->innerLoopTime; ++innerLoopCount) {
            if (this->innerLoopEleTail > 0 && innerLoopCount == this->innerLoopTime - 1) {
                innerCopyLen = this->innerLoopEleTailPad;
                countLen = this->innerLoopEleTail;
            }
            CreateVecIndex(localIndex[NUM_ZERO].ReinterpretCast<int32_t>(), static_cast<int32_t>(indexOffset),
                           countLen);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<DTYPE, float>::value) {
                DataCopyPad(localValueCast, src[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            } else {
                DataCopyPad(localValue, src[gmOffset], {1, (uint32_t)(countLen * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(localValueCast, localValue, RoundMode::CAST_NONE, countLen); // half/bfloat16 -> float32
                PipeBarrier<PIPE_V>();
            }
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

    __aicore__ inline void SoftMaxCompute(uint32_t count, LocalTensor<float>& sampleSrc, float maxValue)
    {
        LocalTensor<float> tempLocalSoftmax = buf4.Get<float>();
        Duplicate(sampleLogitsLocal, maxValue, count);
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

    __aicore__ inline void SoftMaxComputeLocal(uint32_t count, LocalTensor<float>& sampleSrc, float maxValue,
                                               float reduceSumMax)
    {
        LocalTensor<float> tempLocalSoftmax = buf4.Get<float>();
        Duplicate(sampleLogitsLocal, maxValue, count);
        Duplicate(tempLocalSoftmax, reduceSumMax, count);
        PipeBarrier<PIPE_V>();
        Sub(sampleLogitsLocal, sampleSrc, sampleLogitsLocal, count);
        PipeBarrier<PIPE_V>();
        Exp(sampleLogitsLocal, sampleLogitsLocal, count);
        PipeBarrier<PIPE_V>();
        Div(sampleLogitsLocal, sampleLogitsLocal, tempLocalSoftmax, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TopPCompute(LocalTensor<float>& logitsTensor, uint32_t count, float fp32P)
    {
        float cumSum = 0.0;
        for (uint32_t index = 0; index < count; index++) {
            cumSum += logitsTensor.GetValue(index); // sampleLogitsLocal is softmax results
            if (cumSum >= fp32P) {
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

    __aicore__ inline void QSampleGlobal(uint32_t rowId, uint32_t& maxIndex)
    {
        uint32_t innerCopyLen = SOFTMAX_PER_LEN;
        uint32_t countLen = SOFTMAX_PER_LEN;
        uint32_t indexOffset = 0;
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint64_t gmOffset = startIndex;
        LocalTensor<float> localValueCast = buf2.Get<float>();
        LocalTensor<float> sampleDist = buf4.Get<float>();
        float maxValue{FLOAT_MIN};
        params.softmaxLoopTime = SafeCeil(this->topPNum,
                                          SOFTMAX_PER_LEN); //(params.toppNum + SOFTMAX_PER_LEN - 1) / SOFTMAX_PER_LEN;
        params.softmaxLoopEleTail = this->topPNum % SOFTMAX_PER_LEN;
        params.softmaxLoopEleTailPad = (params.softmaxLoopEleTail + THIRTY_ONE) / THIRTY_TWO * THIRTY_TWO;
        for (int32_t innerLoopCount = 0; innerLoopCount < params.softmaxLoopTime; ++innerLoopCount) {
            if (params.softmaxLoopEleTail > 0 && innerLoopCount == params.softmaxLoopTime - 1) {
                countLen = params.softmaxLoopEleTail;
                innerCopyLen = params.softmaxLoopEleTailPad;
            }
            DataCopyPad(localValueCast, logitsGlobalUser[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0});
            DataCopyPad(sampleDist, qGlobal[gmOffset], {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Abs(sampleDist, sampleDist, countLen);
            PipeBarrier<PIPE_V>();
            Adds(sampleDist, sampleDist, this->eps, countLen);
            PipeBarrier<PIPE_V>();
            Div(localValueCast, localValueCast, sampleDist, countLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(localValueCast, localValueCast, localValueCast, countLen, true);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            float value = localValueCast.GetValue(0);
            float index = localValueCast.GetValue(1);
            auto tempIndex = *reinterpret_cast<uint32_t*>(&index);
            if (value > maxValue) {
                maxIndex = indexOffset + tempIndex;
            }
            maxValue = maxValue > value ? maxValue : value;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            gmOffset += innerCopyLen;
            indexOffset += countLen;
        }
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(srcIndexGlobalUser[rowId * this->rowLen]);
        maxIndex = srcIndexGlobalUser.GetValue(startIndex + maxIndex);
    }

    __aicore__ inline void CopyOutIndex(int32_t rowId, uint32_t maxIndex)
    {
        LocalTensor<uint64_t> tempLocalCopy = buf0.Get<uint64_t>();
        this->outputIdx = static_cast<uint64_t>(maxIndex);
        tempLocalCopy.SetValue(0, this->outputIdx);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyIndexOutParams{1, (uint32_t)sizeof(uint64_t), 0, 0, 0};
        DataCopyPad(dstIndexGlobal[rowId], tempLocalCopy, copyIndexOutParams);
    }

    __aicore__ inline void CopyOutLogits(int32_t rowId, uint32_t copyCnt)
    {
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        LocalTensor<float> topKPLogitsLocal = buf5.Get<float>();
        DataCopy(topKPLogitsLocal, localValueRs, Align(copyCnt, EIGHT));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        DataCopyExtParams copyLogitsOutParams{1, (uint32_t)sizeof(float), 0, 0, 0};
        for (uint32_t num = 0; num < this->topPNum; num++) {
            topKPLogitsLocal.SetValue(0, topKPLogitsLocal.GetValue(num));
            uint32_t offset = startIndex + localIndexRs.GetValue(num);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            // Refresh dst GM cache before read-Scalar to write-MTE.
            DataCopyPad(dstLogitsGlobal[offset], topKPLogitsLocal, copyLogitsOutParams);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    template <typename GDTYPE>
    __aicore__ inline void CopyOutLogitsGlobal(int32_t rowId, uint32_t calCnt, GDTYPE& dist)
    {
        // gather the scatter result of selected logits
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t maxNum = GM_COPY_PER_FLOAT_MAX;
        uint32_t rowLenOri = this->rowLen;
        LocalTensor<T> localValue = buf0.Get<T>();
        LocalTensor<float> localValueCast = buf2.Get<float>();
        LocalTensor<uint32_t> indexValue = buf4.Get<uint32_t>();
        LocalTensor<float> topKPLogitsLocal = buf0.Get<float>();
        DataCopyExtParams copyLogitsOutParams{1, (uint32_t)sizeof(float), 0, 0, 0};
        uint32_t startOffset = 0;

        while (rowLenOri > 0) {
            auto dataLen = rowLenOri > maxNum ? maxNum : rowLenOri;
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(localValueCast, logitsGlobal[startIndex + startOffset],
                            {1, (uint32_t)(dataLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            } else {
                DataCopyPad(localValue, logitsGlobal[startIndex + startOffset],
                            {1, (uint32_t)(dataLen * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(localValueCast, localValue, RoundMode::CAST_NONE, dataLen); // half/bfloat16 -> float32
                PipeBarrier<PIPE_V>();
            }
            uint32_t calCntOri = calCnt;
            uint32_t indexStartOffset = 0;
            while (calCntOri > 0) {
                auto indexDataLen = calCntOri > maxNum ? maxNum : calCntOri;
                DataCopyPad(indexValue, dist[startIndex + indexStartOffset],
                            {1, static_cast<uint16_t>(SIZEOF_UINT32 * indexDataLen), 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                for (uint32_t num = 0; num < indexDataLen; ++num) {
                    auto index = indexValue.GetValue(num);
                    int32_t valIdx = index - startOffset; // get local index in current global index section
                    if (valIdx < dataLen && valIdx >= 0) {
                        uint32_t offset = startIndex + index; // adding origin column index on to GM offset of current
                                                              // core batch for dstlogitsIdx
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

    // ============================================================
    // 连续段批量回填（高选中占比场景，替代 CopyOutLogitsGlobal 的散点搬运）
    // 思路：升序排序选中的列索引，使相邻 index 连续 -> 整段连续 MTE2 搬入 + MTE3 搬出，
    //       大幅减少 MTE3 小包次数与标量空转判断。
    // ============================================================

    // 将 srcIndexGlobalUser 前 copyCnt 个选中列索引 (uint32) cast 成 float 写入 logitsGlobalUser，
    // 作为后续 SortAll 的排序“值”。索引 < rowLen < 2^24，fp32 可精确表示。
    __aicore__ inline void PrepareIdxForSort(int32_t rowId, uint32_t copyCnt)
    {
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        LocalTensor<uint32_t> uintBuf = buf1.Get<uint32_t>();
        LocalTensor<float> floatBuf = buf2.Get<float>();
        uint32_t maxNum = GM_COPY_PER_FLOAT_MAX;
        uint32_t remain = copyCnt;
        uint32_t off = 0;
        while (remain > 0) {
            uint32_t curLen = remain > maxNum ? maxNum : remain;
            DataCopyPad(uintBuf, srcIndexGlobalUser[startIndex + off],
                        {1, (uint32_t)(curLen * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Cast(floatBuf, uintBuf.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, curLen); // int32 -> float
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(logitsGlobalUser[startIndex + off], floatBuf, {1, (uint32_t)(curLen * sizeof(float)), 0, 0, 0});
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            remain -= curLen;
            off += curLen;
        }
    }

    // 连续段搬运：从原始 logitsGlobal 连续搬入 [segStart, segStart+segLen) 段，cast 成 float 后连续搬出。
    // 工作 buffer 由调用方传入：!isInLocalTensor 路径用 buf2/buf3；isInLocalTensor 路径需保留 buf3
    // (sampleLogitsLocal 后续 CopyOutMiddle/QSample 仍要用)，改传 buf2/buf4。
    __aicore__ inline void FlushSegment(uint64_t startIndex, uint32_t segStart, uint32_t segLen)
    {
        LocalTensor<T> segBuf = buf2.Get<T>();
        LocalTensor<float> segBufCast = buf3.Get<float>();
        FlushSegment(startIndex, segStart, segLen, segBuf, segBufCast);
    }

    __aicore__ inline void FlushSegment(uint64_t startIndex, uint32_t segStart, uint32_t segLen, LocalTensor<T>& segBuf,
                                        LocalTensor<float>& segBufCast)
    {
        uint32_t maxNum = GM_COPY_PER_FLOAT_MAX;
        uint32_t doneLen = 0;
        while (segLen > 0) {
            uint32_t curLen = segLen > maxNum ? maxNum : segLen;
            uint64_t gmOffset = startIndex + segStart + doneLen;
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(segBufCast, logitsGlobal[gmOffset], {1, (uint32_t)(curLen * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            } else {
                DataCopyPad(segBuf, logitsGlobal[gmOffset], {1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(segBufCast, segBuf, RoundMode::CAST_NONE, curLen); // half/bfloat16 -> float32
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            }
            DataCopyPad(dstLogitsGlobal[gmOffset], segBufCast, {1, (uint32_t)(curLen * sizeof(float)), 0, 0, 0});
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            segLen -= curLen;
            doneLen += curLen;
        }
    }

    __aicore__ inline void CopyOutLogitsOptimized(int32_t rowId, uint32_t copyCnt)
    {
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;

        // 1) 准备待排索引：srcIndexGlobalUser(uint32) -> logitsGlobalUser(float)（只读 srcIndexGlobalUser，不改动）
        PrepareIdxForSort(rowId, copyCnt);

        // 2) 复用 SortAll 对索引做降序排序，结果就地写回 logitsGlobalUser。
        //    不做 padding：直接以 copyCnt 走 guessK 已验证的尾段路径（eightKPartTail 处理不足 1024 的尾段），
        //    使 SortAll 各 region（logitsGlobalUser/sortPart/sortAll）写入范围 ≤ copyCnt ≤ rowLen，
        //    绝不溢出 1×count 的 scratch（rowLen 非 1024 整数倍时 Align 会超出 rowLen 而踩内存）。
        //    绕开 softmax(inputIsLogits=false) 与 cumsum 截断(topp 设极大值)。
        //    复用已有 workspace：logitsGlobalUser 作 srcGlobal（本行 sample_result/QSample 已消费完，可覆写），
        //    sortPart/sortAll 作归并 ping-pong，srcIndexGlobalUser 作 sortIndex 占位（sortOnly 下从不写）。
        TOPKPParams sp = this->params; // 局部副本，避免污染 this->params（后续行的 guessK 路径仍需原值）
        sp.rowId = rowId;
        sp.rowNum = this->rowNum;
        sp.rowLen = this->rowLen;
        sp.inputIsLogits = false;
        sp.topp = 1e30f;
        sp.tensor0 = buf0.Get<float>();
        sp.tensor1 = buf5.Get<float>();
        sp.eightKPartNum = SafeCeil(copyCnt, S_PART_MIN_LEN);
        sp.eightKPartTail = copyCnt % S_PART_MIN_LEN;
        sp.eightKPartTailPad = SafeCeil(sp.eightKPartTail, THIRTY_TWO) * THIRTY_TWO;
        sp.sortOnly = true;
        sortOp.SortAll(sp, logitsGlobalUser, sortPartGlobalUser, sortAllGlobalUser, srcIndexGlobalUser);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        // 3) 扫描降序排序后的索引，识别连续段并整段搬运。
        //    无 padding，直接扫 copyCnt 个真实索引；降序连续: curIdx == prevIdx - 1；segStart 跟踪段的低端列索引。
        LocalTensor<float> idxChunk = buf1.Get<float>();
        uint32_t maxNum = GM_COPY_PER_FLOAT_MAX;
        uint32_t remain = copyCnt;
        uint32_t off = 0;
        int32_t segStart = 0;
        uint32_t segLen = 0;
        int32_t prevIdx = 0;
        bool first = true;
        while (remain > 0) {
            uint32_t curLen = remain > maxNum ? maxNum : remain;
            DataCopyPad(idxChunk, logitsGlobalUser[startIndex + off], {1, (uint32_t)(curLen * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (uint32_t i = 0; i < curLen; i++) {
                int32_t curIdx = static_cast<int32_t>(static_cast<int64_t>(idxChunk.GetValue(i)));
                if (first) {
                    segStart = curIdx;
                    segLen = 1;
                    first = false;
                } else if (curIdx == prevIdx - 1) {
                    segStart = curIdx; // 降序，段低端下移
                    segLen++;
                } else {
                    FlushSegment(startIndex, segStart, segLen);
                    segStart = curIdx;
                    segLen = 1;
                }
                prevIdx = curIdx;
            }
            remain -= curLen;
            off += curLen;
        }
        if (segLen > 0) {
            FlushSegment(startIndex, segStart, segLen); // 收尾段
        }
    }

    // ============================================================
    // isInLocalTensor 高选中占比场景：连续段批量回填（替代 CopyOutLogits 的逐元素散点搬运）
    // 数据已在 local：localValueRs(值) / localIndexRs(列索引)，topPNum<=ksMAX<=1024，整批可片上排序。
    // 直接用高阶 Sort API(复用 SortOneTime+Extract) 对列索引降序排序，使相邻列索引连续，
    // 再复用 FlushSegment 从 logitsGlobal 整段回填（本路径恒有 localValueRs==cast(logitsGlobal[列索引])）。
    // ============================================================
    __aicore__ inline void CopyOutLogitsLocalOptimized(int32_t rowId, uint32_t copyCnt)
    {
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t alignedCnt = Align(copyCnt, THIRTY_TWO); // Sort 需 32 对齐

        // 构造排序键=列索引(float)，payload=列索引(uint32)；尾部 pad 成 FLOAT_MIN，
        // 降序排序后聚到末尾，扫描只取前 copyCnt。localValueRs(buf0) 本路径后续不再需要，可复用。
        LocalTensor<float> keyLocal = buf4.Get<float>();
        LocalTensor<uint32_t> idxPayload = buf0.Get<uint32_t>();
        Duplicate(keyLocal, FLOAT_MIN, alignedCnt);
        PipeBarrier<PIPE_V>();
        Cast(keyLocal, localIndexRs.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, copyCnt); // 列索引 -> float
        DataCopy(idxPayload, localIndexRs, alignedCnt); // payload 携带列索引(uint32)
        PipeBarrier<PIPE_V>();

        // 复用本类 SortOneTime(Concat+Sort<float,true> 降序，内部用 buf5 作 tmp)，结果交错存 buf2。
        // 前两参数为右值引用，用下标临时量绑定(与 Sort32Block 中调用惯用法一致)。
        LocalTensor<float> sortedLocal = buf2.Get<float>();
        SortOneTime(keyLocal[0], idxPayload[0], alignedCnt, sortedLocal);
        LocalTensor<float> sortedKey = buf4.Get<float>();       // 丢弃(仅 Extract 需要一个 value 出口)
        LocalTensor<uint32_t> sortedIdx = buf0.Get<uint32_t>(); // 排序后列索引(降序)
        Extract(sortedKey, sortedIdx, sortedLocal, alignedCnt / THIRTY_TWO);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S); // Extract(V) -> sortedIdx.GetValue(S)
        // 传递保证 V(Extract 读 buf2 / 写 buf4) -> FlushSegment 首个 MTE2(写 buf2/buf4)，避免 WAR。
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);

        // 扫描降序列索引识别连续段并整段回填；FlushSegment 传 buf2/buf4，避开需保留的 buf3(sampleLogitsLocal)。
        LocalTensor<T> segBuf = buf2.Get<T>();
        LocalTensor<float> segBufCast = buf4.Get<float>();
        int32_t segStart = 0;
        uint32_t segLen = 0;
        int32_t prevIdx = 0;
        bool first = true;
        for (uint32_t i = 0; i < copyCnt; i++) {
            int32_t curIdx = static_cast<int32_t>(sortedIdx.GetValue(i));
            if (first) {
                segStart = curIdx;
                segLen = 1;
                first = false;
            } else if (curIdx == prevIdx - 1) {
                segStart = curIdx; // 降序，段低端下移
                segLen++;
            } else {
                FlushSegment(startIndex, segStart, segLen, segBuf, segBufCast);
                segStart = curIdx;
                segLen = 1;
            }
            prevIdx = curIdx;
        }
        if (segLen > 0) {
            FlushSegment(startIndex, segStart, segLen, segBuf, segBufCast); // 收尾段
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void CopyOutMiddle(int32_t rowId, uint32_t calCnt, bool isInLocalTensor)
    {
        // gather the scatter result of selected logits
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        uint64_t startIndex = static_cast<uint64_t>(rowId) * this->rowLen;
        uint32_t countLen = SOFTMAX_PER_LEN / 2;
        uint32_t loopTime = SafeCeil(calCnt, SOFTMAX_PER_LEN / 2);
        uint32_t lastLoopLen = calCnt - (loopTime - 1) * (SOFTMAX_PER_LEN / 2);
        LocalTensor<int64_t> middleIndexLocalInt64 = buf4.Get<int64_t>();
        LocalTensor<float> middleValueLocal = buf5.Get<float>();
        uint32_t startOffset = 0;
        for (int32_t innerLoopCount = 0; innerLoopCount < loopTime; ++innerLoopCount) {
            if (lastLoopLen > 0 && innerLoopCount == loopTime - 1) {
                countLen = lastLoopLen;
            }
            if (!isInLocalTensor) {
                DataCopyPad(sampleLogitsLocal, logitsGlobalUser[startIndex + startOffset],
                            {1, (uint32_t)(countLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(localIndexRs, srcIndexGlobalUser[startIndex + startOffset],
                            {1, (uint32_t)(countLen * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                DataCopyExtParams dstLogitsSortMaskedParams{1, static_cast<uint32_t>(countLen * sizeof(float)), 0, 0,
                                                            0};
                DataCopyPad(dstLogitsSortMaskedGlobal[startIndex + startOffset], sampleLogitsLocal,
                            dstLogitsSortMaskedParams);
                Cast(middleIndexLocalInt64, localIndexRs.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, countLen);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams dstLogitsIdxParams{1, static_cast<uint32_t>(countLen * sizeof(int64_t)), 0, 0, 0};
                DataCopyPad(dstLogitsIdxGlobal[startIndex + startOffset],
                            middleIndexLocalInt64.ReinterpretCast<uint64_t>(), dstLogitsIdxParams);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            } else {
                DataCopyExtParams dstLogitsSortMaskedParams{1, static_cast<uint32_t>(countLen * sizeof(float)), 0, 0,
                                                            0};
                DataCopyPad(dstLogitsSortMaskedGlobal[startIndex + startOffset], sampleLogitsLocal[startOffset],
                            dstLogitsSortMaskedParams);
                Cast(middleIndexLocalInt64, localIndexRs.ReinterpretCast<int32_t>()[startOffset], RoundMode::CAST_NONE,
                     countLen);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams dstLogitsIdxParams{1, static_cast<uint32_t>(countLen * sizeof(int64_t)), 0, 0, 0};
                DataCopyPad(dstLogitsIdxGlobal[startIndex + startOffset],
                            middleIndexLocalInt64.ReinterpretCast<uint64_t>(), dstLogitsIdxParams);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
            startOffset += countLen;
        }
    }

    // argmax and gather
    __aicore__ inline void ArgMaxAndGather(uint32_t calCnt, uint32_t& maxIndex)
    {
        ReduceMax(sampleLogitsLocal, sampleLogitsLocal, sampleLogitsLocal, calCnt, true);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float maxIndexSrc = sampleLogitsLocal.GetValue(1);
        maxIndex = *reinterpret_cast<uint32_t*>(&maxIndexSrc);
        maxIndex = localIndexRs.GetValue(maxIndex);
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
    GlobalTensor<float> dstLogitsSortMaskedGlobal;
    GlobalTensor<uint64_t> dstLogitsIdxGlobal;
    GlobalTensor<int32_t> topKGlobal;
    GlobalTensor<T> topPGlobal;
    GlobalTensor<float> qGlobal;
    GlobalTensor<T> minPGlobal;

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

    uint32_t startTaskId{0};  // start batch index offest of current kernel
    uint32_t rtCoreRowNum{0}; // batch amount processed by this kernel
    int32_t k{0};
    uint32_t k_pad{0};
    uint32_t topPNum{0};
    uint32_t outputIdx{0};
    float eps{.0};
    uint32_t mrgMode{0};
    T p{0};
    T minP{0};
    uint32_t isNeedLogits = 0;
    int32_t topKGuess{TOPP_K_FIX};
    bool ifQSampleCompute = false;
    uint32_t queLenGlobal[NUM_TWO]{0};
    uint32_t ifFind{0};
    uint32_t ksMAX{1024};
    uint32_t inputIsLogits{0};
    bool isInputMinPs = false;
    uint32_t isNeedSampleResult{0};
    bool hasCopyOutLogits = false;
    bool noTopKPMinP = false;

    const float* FP32_NEG_INF_PTR = reinterpret_cast<const float*>(&FP32_NEG_INF_BITS);
    const float SEL_LOGITS_DEF_VAL = *FP32_NEG_INF_PTR;
}; // namespace TopKTopPSampleV2
}; // namespace TopKTopPSampleV2
#endif // TOP_K_TOP_P_SAMPLE_V2_H
