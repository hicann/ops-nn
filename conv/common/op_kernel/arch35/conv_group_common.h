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

template <class CONV, class CONV_TILING>
class ConvGroupCommon : public ConvCommon<CONV, CONV_TILING> {
public:
    __aicore__ __forceinline__ void CalcStartAddrFmapHW();

    __aicore__ __forceinline__ void CalcStartAddrOriGroup(const uint64_t din, const uint64_t dout, const uint32_t kd,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);
    __aicore__ __forceinline__ void CalcStartAddrOriGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart);
    __aicore__ __forceinline__ void CalcStartAddrOriGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart);
    __aicore__ __forceinline__ void ConvKernelImplOriGroup();
    __aicore__ __forceinline__ void ConvKernelImplOptGroupPreload();

    __aicore__ __forceinline__ void UpdateRealCoutOptGroup();

    __aicore__ __forceinline__ void CalcOutputStartAddrOptGroup(const uint64_t doIdxStart);
    __aicore__ __forceinline__ void CalcStartAddrOptGroup(const uint64_t din, const uint64_t dout, const uint32_t kd,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);
    __aicore__ __forceinline__ void CalcStartAddrOptGroupCHW(const uint64_t doIdxStart, const int64_t diIdxStart,
                                                             const uint32_t kd);
    __aicore__ __forceinline__ void CalcStartAddrOptGroupHWC(const uint64_t doIdxStart, const int64_t diIdxStart,
                                                             const uint32_t kd);
    __aicore__ __forceinline__ void ConvKernelImplOptGroup();
    __aicore__ __forceinline__ void DealFixpiepParams(uint64_t groupIter, uint64_t coPerGroup);
    __aicore__ __forceinline__ void SetOptGroupTail();

private:
    using ConvCommon<CONV, CONV_TILING>::convOps;
    // 2d and 3d is conflict, need to fix in future
    using ConvCommon<CONV, CONV_TILING>::convTilingData;
    using ConvCommon<CONV, CONV_TILING>::hasScale;
    using ConvCommon<CONV, CONV_TILING>::fmStartAddr;
    using ConvCommon<CONV, CONV_TILING>::weightStartAddr;
    using ConvCommon<CONV, CONV_TILING>::biasStartAddr;
    using ConvCommon<CONV, CONV_TILING>::scaleStartAddr;
    using ConvCommon<CONV, CONV_TILING>::reluWeightStartAddr;
    using ConvCommon<CONV, CONV_TILING>::outputStartAddr;

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

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrFmapHW()
{
    if constexpr (!CONV::isMMode) {
        int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convTilingData->strideH) -
                                static_cast<int64_t>(convTilingData->padTop);
        convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
        if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
            fmStartAddr += convOps->singleCoreHiStartPos * convTilingData->win * convTilingData->cin;
        } else {
            fmStartAddr += convOps->singleCoreHiStartPos * convTilingData->win;
        }

        // Conv2D Split W
        convOps->singleCoreHiStartPos = hiStartPosTmp;
        if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
            int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convTilingData->strideW) -
                                    static_cast<int64_t>(convTilingData->padLeft);
            convOps->singleCoreWiStartPos = wiStartPosTmp < 0 ? 0 : wiStartPosTmp;
            if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
                fmStartAddr += convOps->singleCoreWiStartPos * convTilingData->cin;
            } else {
                fmStartAddr += convOps->singleCoreWiStartPos;
            }
            convOps->singleCoreWiStartPos = wiStartPosTmp;
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOriGroupHWC(const uint64_t doIdxStart,
                                                                                             const int64_t diIdxStart)
{
    // fmap: Batch -> Di -> HiWi -> Group -> Ci_perg
    // weight: Kd -> KhKw -> Ci_perg -> group -> Co_perg
    // output: Batch -> Do -> HoWo -> Group -> Co_perg
    fmStartAddr += convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->groupIdxStart * convOps->ciPerGroup;
    CalcStartAddrFmapHW();
    weightStartAddr = convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart * convTilingData->cout;
    } else {
        outputStartAddr += convOps->hoIdxStart * convTilingData->wout * convTilingData->cout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
            outputStartAddr += convOps->woIdxStart * convTilingData->cout;
        }
    }
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convTilingData->cin;
        outputStartAddr += doIdxStart * hwOut * convTilingData->cout;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOriGroupCHW(const uint64_t doIdxStart,
                                                                                             const int64_t diIdxStart)
{
    // fmap: Batch -> Group -> Ci_perg -> Di -> HiWi
    // weight: Group -> Co_perg -> Ci_perg -> Kd -> KhKw
    // output: Batch -> Group -> Co_perg -> Do -> HoWo
    fmStartAddr += convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->groupIdxStart * fmapOneGroupSize;
    CalcStartAddrFmapHW();
    weightStartAddr = convOps->groupIdxStart * weightOneGroupSize + convOps->nIdxStart * weightOneCoSize;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->groupIdxStart * outputOneGroupSize + convOps->nIdxStart * dhwOut;
    if constexpr (CONV::isMMode) {
        outputStartAddr += convOps->mIdxStart;
    } else {
        outputStartAddr += convOps->hoIdxStart * convTilingData->wout;
        if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
            outputStartAddr += convOps->woIdxStart;
        }
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOriGroup(
    const uint64_t din, const uint64_t dout, const uint32_t kd, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    hwIn = convTilingData->hin * convTilingData->win;
    hwOut = convTilingData->hout * convTilingData->wout;
    dhwOut = dout * hwOut;
    dhwIn = din * hwIn;
    convOps->fmapOneBatchSize = dhwIn * convTilingData->cin;
    convOps->outputOneBatchSize = dhwOut * convTilingData->cout;

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
        fmapOneGroupSize = convOps->ciPerGroup;
        weightOneGroupSize = convOps->coPerGroup;
        outputOneGroupSize = convOps->coPerGroup;
        CalcStartAddrOriGroupHWC(doIdxStart, diIdxStart);
    } else {
        fmapOneGroupSize = convOps->ciPerGroup * dhwIn;
        weightOneCoSize = convOps->ciPerGroup * kd * convTilingData->kh * convTilingData->kw;
        weightOneGroupSize = convOps->coPerGroup * weightOneCoSize;
        outputOneGroupSize = convOps->coPerGroup * dhwOut;
        CalcStartAddrOriGroupCHW(doIdxStart, diIdxStart);
    }

    if (convTilingData->hasBias) {
        biasStartAddr = convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->groupIdxStart * convOps->coPerGroup + convOps->nIdxStart;
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::DealFixpiepParams(uint64_t groupIter,
                                                                                      uint64_t coPerGroup)
{
    if constexpr (CONV::isQuant || CONV::IS_EXTEND_CONV2D) {
        if (hasScale) {
            if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
                Extendconv2dFixpipeParams fixpipeParamsCopy(convOps->fixpipeParams);
                if (convOps->convTilingData->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    fixpipeParamsCopy.scale0 = convOps->fixpipeParams.scale0[groupIter * coPerGroup];
                }
                if constexpr (CONV::IS_EXTEND_CONV2D) {
                    if (convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                        fixpipeParamsCopy.reluWeight0 = convOps->fixpipeParams.reluWeight0[groupIter * coPerGroup];
                    }
                    if (convTilingData->dualOutput) {
                        if (convTilingData->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            fixpipeParamsCopy.scale1 = convOps->fixpipeParams.scale1[groupIter * coPerGroup];
                        }
                        if (convTilingData->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                            fixpipeParamsCopy.reluWeight1 = convOps->fixpipeParams.reluWeight1[groupIter * coPerGroup];
                        }
                        if (convTilingData->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
                            convTilingData->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
                            convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ||
                            convTilingData->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                            convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                        }
                    } else if (convOps->convTilingData->quantMode0 ==
                               static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                    }
                } else if (convOps->convTilingData->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    convOps->conv.SetFixpipeParams(fixpipeParamsCopy);
                }
            } else {
                convOps->conv.SetScale(convOps->scaleGm[groupIter * coPerGroup]);
            }
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::ConvKernelImplOriGroup()
{
    for (uint64_t groupIter = 0; groupIter < convOps->singleGroups; ++groupIter) {
        convOps->conv.SetWeight(convOps->filterGm[groupIter * weightOneGroupSize]);
        convOps->conv.SetFmap(convOps->fmapGm[groupIter * fmapOneGroupSize]);
        if (convTilingData->hasBias) {
            convOps->conv.SetBias(convOps->biasGm[groupIter * convOps->coPerGroup]);
        }

        DealFixpiepParams(groupIter, convOps->coPerGroup);

        if constexpr (CONV::IS_EXTEND_CONV2D) {
            if (this->convTilingData->dualOutput) {
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

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::UpdateRealCoutOptGroup()
{
    convOps->singleCoreN = convOps->singleCoOpt;

    // update singleGroups : real groups nums in singlecore
    enlargeTail = convTilingData->groups % convTilingData->enlarge;

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
        convOps->singleGroups = enlargeTail == 0 ? convTilingData->enlarge : enlargeTail;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcOutputStartAddrOptGroup(
    const uint64_t doIdxStart)
{
    if constexpr (CONV::C_FORMAT == ConvFormat::NDHWC || CONV::C_FORMAT == ConvFormat::NHWC) {
        outputOneGroupSize = convTilingData->coutOpt;
        outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                          convOps->groupIdxStart * convTilingData->coutOpt + convOps->nIdxStart;

        if constexpr (CONV::C_FORMAT == ConvFormat::NDHWC) {
            outputStartAddr += doIdxStart * hwOut * convTilingData->cout;
        }

        if constexpr (CONV::isMMode) {
            outputStartAddr += convOps->mIdxStart * convTilingData->cout;
        } else {
            outputStartAddr += convOps->hoIdxStart * convTilingData->wout * convTilingData->cout;
            if constexpr (CONV::C_FORMAT == ConvFormat::NHWC) {
                outputStartAddr += convOps->woIdxStart * convTilingData->cout;
            }
        }
    } else if constexpr (CONV::C_FORMAT == ConvFormat::NCDHW || CONV::C_FORMAT == ConvFormat::NCHW) {
        outputOneGroupSize = convTilingData->coutOpt * dhwOut;
        outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                          convOps->groupIdxStart * outputOneGroupSize + convOps->nIdxStart * dhwOut;

        if constexpr (CONV::C_FORMAT == ConvFormat::NCDHW) {
            outputStartAddr += doIdxStart * hwOut;
        }

        if constexpr (CONV::isMMode) {
            outputStartAddr += convOps->mIdxStart;
        } else {
            outputStartAddr += convOps->hoIdxStart * convTilingData->wout;
            if constexpr (CONV::C_FORMAT == ConvFormat::NCHW) {
                outputStartAddr += convOps->woIdxStart;
            }
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOptGroupCHW(const uint64_t doIdxStart,
                                                                                             const int64_t diIdxStart,
                                                                                             const uint32_t kd)
{
    // fmap: Batch -> Group(enlarge) -> Ci_opt -> Di -> HiWi
    // weight: Group(enlarge) -> Co_opt -> Ci_opt -> Kd -> KhKw
    // output: Batch -> Group(enlarge) -> Co_opt -> Do -> HoWo
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->groupIdxStart * fmapOneGroupSize;
    CalcStartAddrFmapHW();
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z) {
        weightOneGroupSize = convTilingData->coutOpt * convTilingData->cinOpt * kd * convTilingData->kh *
                             convTilingData->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize + convOps->nIdxStart * convOps->k0;
    } else {
        weightOneGroupSize = convTilingData->coutOpt * convOps->ciPerGroup * kd * convTilingData->kh *
                             convTilingData->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize;
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
    }

    CalcOutputStartAddrOptGroup(doIdxStart);
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOptGroupHWC(const uint64_t doIdxStart,
                                                                                             const int64_t diIdxStart,
                                                                                             const uint32_t kd)
{
    // fmap: Batch -> Di -> HiWi -> Group(enlarge) -> Ci_opt
    // weight: Kd -> KhKw -> Ci_opt -> Group(enlarge) -> Co_opt
    // output: Batch -> Do -> HoWo -> Group(enlarge) -> Co_opt
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->groupIdxStart * convTilingData->cinOpt;
    CalcStartAddrFmapHW();
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z) {
        weightOneGroupSize = convTilingData->coutOpt * convTilingData->cinOpt * kd * convTilingData->kh *
                             convTilingData->kw;
        weightStartAddr = convOps->groupIdxStart * weightOneGroupSize + convOps->nIdxStart * convOps->k0;
    } else {
        weightOneGroupSize = convTilingData->coutOpt;
        weightStartAddr = convOps->groupIdxStart * convTilingData->coutOpt;
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convTilingData->cin;
    }

    CalcOutputStartAddrOptGroup(doIdxStart);
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::CalcStartAddrOptGroup(
    const uint64_t din, const uint64_t dout, const uint32_t kd, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    hwIn = convTilingData->hin * convTilingData->win;
    hwOut = convTilingData->hout * convTilingData->wout;
    dhwOut = dout * hwOut;
    dhwIn = din * hwIn;
    convOps->fmapOneBatchSize = convTilingData->cin * dhwIn;
    convOps->outputOneBatchSize = convTilingData->cout * dhwOut;
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC || CONV::A_FORMAT == ConvFormat::NHWC) {
        fmapOneGroupSize = convTilingData->cinOpt;
        CalcStartAddrOptGroupHWC(doIdxStart, diIdxStart, kd);
    } else {
        fmapOneGroupSize = convTilingData->cinOpt * dhwIn;
        CalcStartAddrOptGroupCHW(doIdxStart, diIdxStart, kd);
    }

    if (convTilingData->hasBias) {
        biasStartAddr = convOps->groupIdxStart * convTilingData->coutOpt + convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->groupIdxStart * convTilingData->coutOpt + convOps->nIdxStart;
        }
    }

    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->convTilingData->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
            convOps->convTilingData->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scaleStartAddr = convOps->groupIdxStart * convTilingData->coutOpt + convOps->nIdxStart;
        }
        if (convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ||
            convTilingData->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            reluWeightStartAddr = convOps->groupIdxStart * convTilingData->coutOpt + convOps->nIdxStart;
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::ConvKernelImplOptGroup()
{
    convOps->conv.SetWeightStartPosition(convOps->nIdxStart);
    if (CONV::IS_OPTGROUP_PRELOAD) {
        ConvKernelImplOptGroupPreload();
        return;
    }
    for (uint64_t groupOptIter = 0; groupOptIter < convOps->singleGroupOpt; ++groupOptIter) {
        if (unlikely(convOps->isGroupDimTail && enlargeTail != 0 && groupOptIter == convOps->singleGroupOpt - 1)) {
            if (unlikely(convOps->singleCoOpt == 0)) {
                break;
            }

            SetOptGroupTail();
        }

        convOps->conv.SetIterIndex(groupOptIter);

        convOps->conv.SetFmap(convOps->fmapGm[groupOptIter * fmapOneGroupSize]);
        convOps->conv.SetWeight(convOps->filterGm[groupOptIter * weightOneGroupSize]);

        if (convTilingData->hasBias) {
            convOps->conv.SetBias(convOps->biasGm[groupOptIter * convTilingData->coutOpt]);
        }

        DealFixpiepParams(groupOptIter, convTilingData->coutOpt);

        if constexpr (CONV::IS_EXTEND_CONV2D) {
            if (this->convTilingData->dualOutput) {
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

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::ConvKernelImplOptGroupPreload()
{
    uint64_t groupOptIter = 0;

    if (convOps->isGroupDimTail && enlargeTail != 0) {
        if (unlikely(convOps->singleCoOpt == 0 && convOps->singleGroupOpt == 1)) {
            return;
        }
        convOps->conv.SetOptGroupParams(enlargeTail, convOps->singleGroupOpt, convOps->singleCoOpt);
    } else {
        convOps->conv.SetOptGroupParams(convTilingData->enlarge, convOps->singleGroupOpt, convOps->singleCoOpt);
    }

    convOps->conv.SetIterIndex(groupOptIter);

    convOps->conv.SetFmap(convOps->fmapGm);
    convOps->conv.SetWeight(convOps->filterGm);

    if (convTilingData->hasBias) {
        convOps->conv.SetBias(convOps->biasGm);
    }

    // quant need, current not care
    DealFixpiepParams(groupOptIter, convTilingData->coutOpt);

    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->dualOutput) {
            convOps->conv.IterateAll(convOps->outputGm, convOps->output1Gm);
        } else {
            convOps->conv.IterateAll(convOps->outputGm);
        }
    } else {
        convOps->conv.IterateAll(convOps->outputGm);
    }
    convOps->conv.End();
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvGroupCommon<CONV, CONV_TILING>::SetOptGroupTail()
{
    if (convOps->singleCoOpt != convOps->singleCoreN) {
        if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW || CONV::A_FORMAT == ConvFormat::NDHWC) {
            if constexpr (CONV::isMMode) {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreDout, convOps->singleCoreM,
                                                   convOps->singleCoreBatch);
            } else {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreDout, convOps->singleCoreHo,
                                                   convTilingData->wout, convOps->singleCoreBatch);
            }
        } else {
            if constexpr (CONV::isMMode) {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreM,
                                                   convOps->singleCoreBatch);
            } else {
                convOps->conv.SetSingleOutputShape(convOps->singleCoOpt, convOps->singleCoreHo, convOps->singleCoreWo,
                                                   convOps->singleCoreBatch);
            }
        }
    }

    convOps->conv.SetOptGroupParams(convOps->singleGroups, convOps->singleGroupOpt, convOps->singleCoOpt);
}

#endif // CONV_GROUP_COMMON_H