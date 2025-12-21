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
 * \file conv_group_common.h
 * \brief
 */

#ifndef CONV_GROUP_COMMON_H
#define CONV_GROUP_COMMON_H

#include "conv_common.h"

template <class CONV, class RUN_INFO>
class ConvGroupCommon : public ConvCommon<CONV, RUN_INFO> {
public:
    __aicore__ __forceinline__ void CalcStartAddrFmapHW();

    __aicore__ __forceinline__ void CalcStartAddrOriGroup(const uint64_t din, const uint64_t dout, const uint32_t kd,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);
    __aicore__ __forceinline__ void CalcStartAddrOriGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart);
    __aicore__ __forceinline__ void CalcStartAddrOriGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart);
    __aicore__ __forceinline__ void ConvKernelImplOriGroup();

    __aicore__ __forceinline__ void UpdateRealCoutOptGroup();

    __aicore__ __forceinline__ void CalcStartAddrOptGroup(const uint64_t din, const uint64_t dout, const uint32_t kd,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);
    __aicore__ __forceinline__ void CalcStartAddrOptGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart, const uint32_t kd);
    __aicore__ __forceinline__ void CalcStartAddrOptGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart, const uint32_t kd);
    __aicore__ __forceinline__ void ConvKernelImplOptGroup();
    __aicore__ __forceinline__ void DealFixpiepParams(uint64_t groupIter, uint64_t coPerGroup);
    __aicore__ __forceinline__ void SetOptGroupTail();

private:
    using ConvCommon<CONV, RUN_INFO>::convOps;
    using ConvCommon<CONV, RUN_INFO>::convRunInfo;
    using ConvCommon<CONV, RUN_INFO>::hasScale;
    using ConvCommon<CONV, RUN_INFO>::fmStartAddr;
    using ConvCommon<CONV, RUN_INFO>::weightStartAddr;
    using ConvCommon<CONV, RUN_INFO>::biasStartAddr;
    using ConvCommon<CONV, RUN_INFO>::scaleStartAddr;
    using ConvCommon<CONV, RUN_INFO>::outputStartAddr;

    uint64_t fmapOneGroupSize = 0;
    uint64_t weightOneCoSize = 0;
    uint64_t weightOneGroupSize = 0;
    uint64_t outputOneGroupSize = 0;

    uint32_t enlargeTail = 0;
    uint64_t hwIn = 0;
    uint64_t hwOut = 0;
    uint64_t dhwOut = 0;
    uint64_t dhwIn = 0;
};

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrFmapHW()
{
    if constexpr (!CONV::isMMode) {
        int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convRunInfo->strideH) -
                                static_cast<int64_t>(convRunInfo->padTop);
        convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
        if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
            fmStartAddr += convOps->singleCoreHiStartPos * convRunInfo->win * convRunInfo->cin;
        } else {
            fmStartAddr += convOps->singleCoreHiStartPos * convRunInfo->win;
        }
        
        // Conv2D Split W
        convOps->singleCoreHiStartPos = hiStartPosTmp;
        if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
            int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convRunInfo->strideW) -
                                    static_cast<int64_t>(convRunInfo->padLeft);
            convOps->singleCoreWiStartPos = wiStartPosTmp < 0 ? 0 : wiStartPosTmp;
            if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
                fmStartAddr += convOps->singleCoreWiStartPos * convRunInfo->cin;
            } else {
                fmStartAddr += convOps->singleCoreWiStartPos;
            }
            convOps->singleCoreWiStartPos = wiStartPosTmp;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOriGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart)
{
    // fmap: Batch -> Di -> HiWi -> Group -> Ci_perg
    // weight: Kd -> KhKw -> Ci_perg -> group -> Co_perg
    // output: Batch -> Do -> HoWo -> Group -> Co_perg
    fmStartAddr += convOps->batchIdxStart * convOps->fmapOneBatchSize +
                   convOps->groupIdxStart * convOps->ciPerGroup;
    CalcStartAddrFmapHW();
    weightStartAddr = convOps->groupIdxStart * convOps->coPerGroup +
                      convOps->nIdxStart;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart * convRunInfo->cout;
    } else {
        outputStartAddr += convOps->hoIdxStart * convRunInfo->wout * convRunInfo->cout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
            outputStartAddr += convOps->woIdxStart * convRunInfo->cout;
        }
    }
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convRunInfo->cin;
        outputStartAddr += doIdxStart * hwOut * convRunInfo->cout;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOriGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart)
{
    // fmap: Batch -> Group -> Ci_perg -> Di -> HiWi
    // weight: Group -> Co_perg -> Ci_perg -> Kd -> KhKw
    // output: Batch -> Group -> Co_perg -> Do -> HoWo
    fmStartAddr += convOps->batchIdxStart * convOps->fmapOneBatchSize +
                   convOps->groupIdxStart * fmapOneGroupSize;
    CalcStartAddrFmapHW();
    weightStartAddr = convOps->groupIdxStart * weightOneGroupSize +
                      convOps->nIdxStart * weightOneCoSize;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * outputOneGroupSize +
                      convOps->nIdxStart * dhwOut;
    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart;
    } else {
        outputStartAddr += convOps->hoIdxStart * convRunInfo->wout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
            outputStartAddr += convOps->woIdxStart;
        }
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOriGroup(const uint64_t din, const uint64_t dout, const uint32_t kd, const uint64_t doIdxStart,
                          const int64_t diIdxStart)
{
    hwIn = convRunInfo->hin * convRunInfo->win;
    hwOut = convRunInfo->hout * convRunInfo->wout;
    dhwOut = dout * hwOut;
    dhwIn = din * hwIn;
    convOps->fmapOneBatchSize = dhwIn * convRunInfo->cin;
    convOps->outputOneBatchSize = dhwOut * convRunInfo->cout;

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
        fmapOneGroupSize = convOps->ciPerGroup;
        weightOneGroupSize = convOps->coPerGroup;
        outputOneGroupSize = convOps->coPerGroup;
        CalcStartAddrOriGroupHWC(doIdxStart, diIdxStart);
    } else {
        fmapOneGroupSize = convOps->ciPerGroup * dhwIn;
        weightOneCoSize = convOps->ciPerGroup * kd * convRunInfo->kh * convRunInfo->kw;
        weightOneGroupSize = convOps->coPerGroup * weightOneCoSize;
        outputOneGroupSize = convOps->coPerGroup * dhwOut;
        CalcStartAddrOriGroupCHW(doIdxStart, diIdxStart);
    }

    if (convRunInfo->hasBias) {
        biasStartAddr = convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    DealFixpiepParams(uint64_t groupIter, uint64_t coPerGroup)
{
    if constexpr (CONV::isQuant || CONV::IS_EXTEND_CONV2D) {
        if (hasScale) {
            if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
                Extendconv2dFixpipeParams fixpipeParamsCopy(convOps->fixpipeParams);
                if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    fixpipeParamsCopy.scale0 = convOps->fixpipeParams.scale0[groupIter * coPerGroup];
                }
                if constexpr (CONV::IS_EXTEND_CONV2D) {
                    if (convOps->dualOutput) {
                        if (convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            fixpipeParamsCopy.scale1 = convOps->fixpipeParams.scale1[groupIter * coPerGroup];
                        }
                    }
                }
                if constexpr (CONV::IS_EXTEND_CONV2D) {
                    if (convOps->dualOutput) {
                        if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
                            convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                        }
                    } else if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                    }
                } else if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                }
            } else {
                convOps->conv.SetScale(convOps->scaleGm[groupIter * coPerGroup]);
            }
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    ConvKernelImplOriGroup()
{
    for (uint64_t groupIter = 0; groupIter < convOps->singleGroups; ++groupIter) {
        convOps->conv.SetWeight(convOps->filterGm[groupIter * weightOneGroupSize]);
        convOps->conv.SetFmap(convOps->fmapGm[groupIter * fmapOneGroupSize]);
        if (convRunInfo->hasBias) {
            convOps->conv.SetBias(convOps->biasGm[groupIter * convOps->coPerGroup]);
        }

        DealFixpiepParams(groupIter, convOps->coPerGroup);

        if constexpr (CONV::IS_EXTEND_CONV2D) {
            if (convOps->dualOutput) {
                convOps->conv.IterateAll(convOps->outputGm[groupIter * outputOneGroupSize],
                                         convOps->output1Gm[groupIter * outputOneGroupSize]);
            } else {
                convOps->conv.IterateAll(convOps->outputGm[groupIter * outputOneGroupSize]);
            }
        } else {
            convOps->conv.IterateAll(convOps->outputGm[groupIter * outputOneGroupSize]);
        }
        convOps->conv.End();
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    UpdateRealCoutOptGroup()
{
    convOps->singleCoreN = convOps->singleCoOpt;

    // update singleGroups : real groups nums in singlecore
    enlargeTail = convRunInfo->groups % convRunInfo->enlarge;

    if (unlikely(convOps->isGroupDimTail && enlargeTail != 0)) {
        uint64_t realCout = enlargeTail * convOps->coPerGroup;
        if (unlikely(convOps->nIdxStart + convOps->singleCoOpt > realCout)) {
            if (realCout <= convOps->nIdxStart) {
                convOps->singleCoOpt = 0;
            } else {
                convOps->singleCoOpt = realCout - convOps->nIdxStart;
            }
        }
    }

    if (unlikely(convOps->isGroupDimTail)) {
        convOps->singleGroups = enlargeTail == 0 ? convRunInfo->enlarge : enlargeTail;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOptGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart, const uint32_t kd)
{
    // fmap: Batch -> Group(enlarge) -> Ci_opt -> Di -> HiWi
    // weight: Group(enlarge) -> Co_opt -> Ci_opt -> Kd -> KhKw
    // output: Batch -> Group(enlarge) -> Co_opt -> Do -> HoWo
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize +
                  convOps->groupIdxStart * fmapOneGroupSize;
    CalcStartAddrFmapHW();
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z) {
        weightOneGroupSize = convRunInfo->coutOpt * convRunInfo->cinOpt * kd * convRunInfo->kh * convRunInfo->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize + convOps->nIdxStart * convOps->k0;
    } else {
        weightOneGroupSize = convRunInfo->coutOpt * convOps->ciPerGroup * kd * convRunInfo->kh * convRunInfo->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize;
    }
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * outputOneGroupSize +
                      convOps->nIdxStart * dhwOut;

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }

    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart;
    } else {
        outputStartAddr += convOps->hoIdxStart * convRunInfo->wout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
            outputStartAddr += convOps->woIdxStart;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOptGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart, const uint32_t kd)
{
    // fmap: Batch -> Di -> HiWi -> Group(enlarge) -> Ci_opt
    // weight: Kd -> KhKw -> Ci_opt -> Group(enlarge) -> Co_opt
    // output: Batch -> Do -> HoWo -> Group(enlarge) -> Co_opt
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize +
                  convOps->groupIdxStart * convRunInfo->cinOpt;
    CalcStartAddrFmapHW();
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z) {
        weightOneGroupSize = convRunInfo->coutOpt * convRunInfo->cinOpt * kd * convRunInfo->kh * convRunInfo->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize + convOps->nIdxStart * convOps->k0;
    } else {
        weightOneGroupSize = convRunInfo->coutOpt;
        weightStartAddr = convOps->groupIdxStart * convRunInfo->coutOpt;
    }
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * convRunInfo->coutOpt +
                      convOps->nIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convRunInfo->cin;
        outputStartAddr += doIdxStart * hwOut * convRunInfo->cout;
    }

    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart * convRunInfo->cout;
    } else {
        outputStartAddr += convOps->hoIdxStart * convRunInfo->wout * convRunInfo->cout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
            outputStartAddr += convOps->woIdxStart * convRunInfo->cout;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    CalcStartAddrOptGroup(const uint64_t din, const uint64_t dout, const uint32_t kd, const uint64_t doIdxStart,
                          const int64_t diIdxStart)
{
    hwIn = convRunInfo->hin * convRunInfo->win;
    hwOut = convRunInfo->hout * convRunInfo->wout;
    dhwOut = dout * hwOut;
    dhwIn = din * hwIn;
    convOps->fmapOneBatchSize = convRunInfo->cin * dhwIn;
    convOps->outputOneBatchSize = convRunInfo->cout * dhwOut;
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
        fmapOneGroupSize = convRunInfo->cinOpt;
        outputOneGroupSize = convRunInfo->coutOpt;
        CalcStartAddrOptGroupHWC(doIdxStart, diIdxStart, kd);
    } else {
        fmapOneGroupSize = convRunInfo->cinOpt * dhwIn;
        outputOneGroupSize = convRunInfo->coutOpt * dhwOut;
        CalcStartAddrOptGroupCHW(doIdxStart, diIdxStart, kd);
    }

    if (convRunInfo->hasBias) {
        biasStartAddr = convOps->groupIdxStart * convRunInfo->coutOpt + convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->groupIdxStart * convRunInfo->coutOpt + convOps->nIdxStart;
        }
    }

    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
            convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scaleStartAddr = convOps->groupIdxStart * convRunInfo->coutOpt + convOps->nIdxStart;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    ConvKernelImplOptGroup()
{
    convOps->conv.SetWeightStartPosition(convOps->nIdxStart);
    for (uint64_t groupOptIter = 0; groupOptIter < convOps->singleGroupOpt; ++groupOptIter) {
        if (unlikely(convOps->isGroupDimTail && enlargeTail != 0 &&  groupOptIter == convOps->singleGroupOpt - 1)) {
            if (unlikely(convOps->singleCoOpt == 0)) {
                break;
            }

            SetOptGroupTail();
        }

        convOps->conv.SetIterIndex(groupOptIter);

        convOps->conv.SetFmap(convOps->fmapGm[groupOptIter * fmapOneGroupSize]);
        convOps->conv.SetWeight(convOps->filterGm[groupOptIter * weightOneGroupSize]);

        if (convRunInfo->hasBias) {
            convOps->conv.SetBias(convOps->biasGm[groupOptIter * convRunInfo->coutOpt]);
        }

        DealFixpiepParams(groupOptIter, convRunInfo->coutOpt);

        if constexpr (CONV::IS_EXTEND_CONV2D) {
            if (convOps->dualOutput) {
                convOps->conv.IterateAll(convOps->outputGm[groupOptIter * outputOneGroupSize],
                                         convOps->output1Gm[groupOptIter * outputOneGroupSize]);
            } else {
                convOps->conv.IterateAll(convOps->outputGm[groupOptIter * outputOneGroupSize]);
            }
        } else {
            convOps->conv.IterateAll(convOps->outputGm[groupOptIter * outputOneGroupSize]);
        }
        convOps->conv.End();
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, RUN_INFO>::
    SetOptGroupTail()
{
    if (convOps->singleCoOpt != convOps->singleCoreN) {
        if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW || CONV::A_FORMAT == ConvFormat::NDHWC) {
            if constexpr (CONV::isMMode) {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreDout, convOps->singleCoreM,
                                                   convOps->singleCoreBatch);
            } else {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreDout, convOps->singleCoreHo,
                                                   convRunInfo->wout, convOps->singleCoreBatch);
            }
        } else {
            if constexpr (CONV::isMMode) {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreM, convOps->singleCoreBatch);
            } else {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreHo, convOps->singleCoreWo,
                                                   convOps->singleCoreBatch);
            }
        }
    }

    convOps->conv.SetOptGroupParams(convOps->singleGroups, convOps->singleGroupOpt);
}

#endif // CONV_GROUP_COMMON_H