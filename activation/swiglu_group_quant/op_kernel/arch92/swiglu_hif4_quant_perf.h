/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_HIF4_QUANT_PERF_H
#define SWIGLU_HIF4_QUANT_PERF_H

#include "kernel_operator.h"
#include "../arch35/swiglu_group_quant_base.h"
#include "swiglu_hif4_quant_common.h"

namespace SwigluGroupQuant {
using namespace AscendC;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
template <typename T0, typename T1, typename T2, bool outputOrigin = false>
class SwigluHiF4QuantPerf {
public:
    __aicore__ inline SwigluHiF4QuantPerf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                                GM_ADDR yOrigin, GM_ADDR workspace, const SwigluGroupQuantTilingData* tilingDataPtr,
                                TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        scaleGm.SetGlobalBuffer((__gm__ T2*)scale);

        pipe->InitBufPool(tBufPool, tilingData->ubSize);
        ProcessGroupIndexTiling(groupIndex, tilingData, tBufPool, groupIndexQue, groupIndexSumBuf, groupIndexGm,
                                groupSumLocal, hasGroupIndex_, usedCoreNums, rowOfFormerBlock, rowOfTailBlock,
                                rowLoopOfFormerBlock, rowLoopOfTailBlock, tailRowFactorOfFormerBlock,
                                tailRowFactorOfTailBlock);

        if (topkWeight != nullptr) {
            hasTopkWeight_ = true;
            topkWeightGm.SetGlobalBuffer((__gm__ float*)topkWeight);
            tBufPool.InitBuffer(topkWeightQue, DOUBLE_BUFFER_NUM,
                                RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        }
        if constexpr (outputOrigin) {
            yOriginGm.SetGlobalBuffer((__gm__ T0*)yOrigin);
            tBufPool.InitBuffer(yOriginQue, DOUBLE_BUFFER_NUM,
                                tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        }
        // hif4量化时，一次需要读取4个VL，也就是4 * VL = 1024Byte = 512 * sizeof(bfloat16_t),
        // 为了防止越界，实际处理的个数需要做512对齐
        int64_t swigluElemCnt = CeilAlign(tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor),
                                          4 * REPEAT_SIZE / sizeof(T0));
        tBufPool.InitBuffer(xQue, DOUBLE_BUFFER_NUM,
                            tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0) * DIGIT_TWO);

        tBufPool.InitBuffer(yQue, DOUBLE_BUFFER_NUM,
                            RoundUp<uint8_t>(CeilDiv(swigluElemCnt, FP4_PACK_NUM)) * sizeof(int8_t));

        tBufPool.InitBuffer(scaleQue, DOUBLE_BUFFER_NUM,
                            RoundUp<uint32_t>(CeilDiv(swigluElemCnt, PER_HIF4)) * sizeof(T2));

        tBufPool.InitBuffer(swigluBuf, swigluElemCnt * sizeof(T0));
        tBufPool.InitBuffer(eABuf, RoundUp<uint8_t>(CeilDiv(swigluElemCnt, PER_HIF4)) * sizeof(uint8_t));
        tBufPool.InitBuffer(eBBuf, RoundUp<uint8_t>(CeilDiv(swigluElemCnt, PER_HIF4)) * sizeof(uint8_t));
        tBufPool.InitBuffer(eCBuf, RoundUp<uint16_t>(CeilDiv(swigluElemCnt, PER_HIF4 / DIGIT_TWO)) * sizeof(uint16_t));

        swigluLocal = swigluBuf.Get<bfloat16_t>();
        eALocal = eABuf.Get<uint8_t>();
        eBLocal = eBBuf.Get<uint8_t>();
        eCLocal = eCBuf.Get<uint16_t>();

        hasClampValue_ = (tilingData->hasClampLimit == 1);
        clampValue_ = tilingData->clampLimit;
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        if (tilingData->bs == 0 || tilingData->splitD == 0) {
            // Empty input (zero token count or zero last dim of x): nothing to compute.
            return;
        }
        if (GetBlockIdx() >= usedCoreNums) {
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop = (curBlockIdx == usedCoreNums - 1) ? rowLoopOfTailBlock : rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums - 1) ? tailRowFactorOfTailBlock :
                                                                    tailRowFactorOfFormerBlock;
        int64_t x0GmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->d;
        int64_t x1GmBaseOffset = x0GmBaseOffset + tilingData->splitD;
        int64_t yGmBaseOffset = curBlockIdx * rowOfFormerBlock * (tilingData->splitD / FP4_PACK_NUM);
        int64_t scaleGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->scaleCol;
        int64_t topkWeightGmBaseOffset = curBlockIdx * rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            if (hasTopkWeight_) {
                topkWeightLocal = topkWeightQue.template AllocTensor<float>();
                CopyIn(topkWeightGm[topkWeightGmBaseOffset + rowOuterIdx * tilingData->rowFactor], topkWeightLocal, 1,
                       curRowFactor);
                topkWeightQue.template EnQue(topkWeightLocal);
                topkWeightLocal = topkWeightQue.template DeQue<float>();
            }
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor = (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor :
                                                                           tilingData->dFactor;
                int64_t scaleDFactor = CeilDiv(curDFactor, PER_HIF4);
                int64_t xBaseOffset = rowOuterIdx * tilingData->rowFactor * tilingData->d +
                                      dLoopIdx * tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T0>();
                CopyIn(xGm[x0GmBaseOffset + xBaseOffset], xLocal, curRowFactor, curDFactor, tilingData->d - curDFactor);

                CopyIn(xGm[x1GmBaseOffset + xBaseOffset], xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                       curRowFactor, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T0>();
                // yOrigin keeps the pre-weight swiglu result and is copied out through the queue
                // (VF->MTE3 sync), while quantization always consumes the swiglu result from the
                // dedicated swiglu buffer.
                if constexpr (outputOrigin) {
                    yOriginLocal = yOriginQue.template AllocTensor<T0>();
                }
                if (hasTopkWeight_) {
                    if (hasClampValue_) {
                        VFProcessSwiglu<T0, bfloat16_t, true, true, outputOrigin>(
                            swigluLocal, yOriginLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                            topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwiglu<T0, bfloat16_t, true, false, outputOrigin>(
                            swigluLocal, yOriginLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                            topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    }
                } else {
                    if (hasClampValue_) {
                        VFProcessSwiglu<T0, bfloat16_t, false, true, outputOrigin>(
                            swigluLocal, yOriginLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                            topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwiglu<T0, bfloat16_t, false, false, outputOrigin>(
                            swigluLocal, yOriginLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                            topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    }
                }

                yLocal = yQue.template AllocTensor<uint8_t>();

                VFProcessSwigluHiF4InvScaleAndQuant(
                    yLocal.template ReinterpretCast<int32_t>(), swigluLocal.template ReinterpretCast<float>(),
                    eALocal.template ReinterpretCast<uint8_t>(), eBLocal.template ReinterpretCast<uint32_t>(),
                    eCLocal.template ReinterpretCast<uint8_t>(),
                    CeilAlign(curRowFactor * RoundUp<T0>(curDFactor), HIF4_QUANT_ALIGN_NUM));
                xQue.template FreeTensor(xLocal);

                yQue.template EnQue(yLocal);
                scaleLocal = scaleQue.template AllocTensor<T2>();

                VFProcessSwigluGroupQuantScale(scaleLocal.template ReinterpretCast<uint16_t>(), eALocal, eBLocal,
                                               eCLocal,
                                               CeilAlign(curRowFactor * RoundUp<T0>(curDFactor), HIF4_QUANT_ALIGN_NUM));

                scaleQue.template EnQue(scaleLocal);
                scaleLocal = scaleQue.template DeQue<T2>();

                CopyOut<T2, AscendC::PaddingMode::Compact>(
                    scaleLocal,
                    scaleGm[scaleGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->scaleCol +
                            dLoopIdx * CeilDiv(tilingData->dFactor, PER_MX_FP16)],
                    curRowFactor, scaleDFactor, tilingData->scaleCol - scaleDFactor);
                scaleQue.template FreeTensor(scaleLocal);

                yLocal = yQue.template DeQue<uint8_t>();

                CopyOut(yLocal.template ReinterpretCast<int8_t>(),
                        yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor * (tilingData->splitD / FP4_PACK_NUM) +
                            dLoopIdx * (tilingData->dFactor / FP4_PACK_NUM)],
                        curRowFactor, CeilDiv(curDFactor, FP4_PACK_NUM),
                        tilingData->splitD / FP4_PACK_NUM - CeilDiv(curDFactor, FP4_PACK_NUM));
                yQue.template FreeTensor(yLocal);

                // copy yOrigin to gm
                if constexpr (outputOrigin) {
                    int64_t yOriginGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->splitD;
                    yOriginQue.template EnQue(yOriginLocal);
                    yOriginLocal = yOriginQue.template DeQue<T0>();
                    CopyOut(yOriginLocal,
                            yOriginGm[yOriginGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->splitD +
                                      dLoopIdx * tilingData->dFactor],
                            curRowFactor, curDFactor, tilingData->splitD - curDFactor);
                    yOriginQue.template FreeTensor(yOriginLocal);
                }
            }
            if (hasTopkWeight_) {
                topkWeightQue.template FreeTensor(topkWeightLocal);
            }
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupQuantTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int64_t> groupIndexGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<T2> scaleGm;
    GlobalTensor<T0> yOriginGm;
    GlobalTensor<float> topkWeightGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> scaleQue;
    TQue<QuePosition::VECOUT, 1> yOriginQue;

    TBuf<QuePosition::VECCALC> swigluBuf;
    TBuf<QuePosition::VECCALC> eABuf;
    TBuf<QuePosition::VECCALC> eBBuf;
    TBuf<QuePosition::VECCALC> eCBuf;
    TBuf<QuePosition::VECCALC> testBuf;
    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TQue<QuePosition::VECIN, 1> topkWeightQue;
    TBufPool<QuePosition::VECCALC, TBUF_POOL_NUM> tBufPool;

    LocalTensor<T0> xLocal;
    LocalTensor<uint8_t> yLocal;
    LocalTensor<T2> scaleLocal;
    LocalTensor<T0> yOriginLocal;
    LocalTensor<bfloat16_t> swigluLocal;
    LocalTensor<uint8_t> eALocal;
    LocalTensor<uint8_t> eBLocal;
    LocalTensor<uint16_t> eCLocal;
    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;
    LocalTensor<float> topkWeightLocal;
    LocalTensor<bfloat16_t> testLocal;

    uint16_t maxValue = 0;
    float clampValue_ = 448.0f;
    bool hasClampValue_ = false;
    bool hasGroupIndex_ = false;
    bool hasTopkWeight_ = false;

    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};
#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)

} // namespace SwigluGroupQuant

#endif
