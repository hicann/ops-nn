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
 * \file conv_common_func.h
 * \brief
 */

#ifndef CONV_COMMON_FUNC_H
#define CONV_COMMON_FUNC_H

#include <cstring>
#include "conv_config.h"
#include "conv_framework_util.h"
#include "conv_iterate_impl.h"
#include "conv_opt_group_impl.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace conv;

CONV_DECLARE_REG_IMPL(SetFmap);
CONV_DECLARE_REG_IMPL(SetWeight);
CONV_DECLARE_REG_IMPL(SetBias);
CONV_DECLARE_REG_IMPL(SetScale);
CONV_DECLARE_REG_IMPL(SetFixpipeParams);
CONV_DECLARE_REG_IMPL(ConvPreProcess);
CONV_DECLARE_REG_IMPL(ConvPostProcess);
CONV_DECLARE_REG_IMPL(IterateAll);
CONV_DECLARE_REG_IMPL(GetTensorC);
CONV_DECLARE_REG_IMPL(End);

using TypeFalse = struct {
    __uint128_t _[1024];
};

template <class Intf, uint32_t ImplType>
struct SetFmap {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::FmapT> &fmap)
    {
        self->ctx.agm.SetGlobalBuffer(fmap.GetPhyAddr(0), fmap.GetSize());
        self->ctx.isFirstIterate = true;
    }
};

template <class Intf, uint32_t ImplType>
struct SetWeight {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::WeightT> &weight)
    {
        self->ctx.bgm.SetGlobalBuffer(weight.GetPhyAddr(0), weight.GetSize());
        self->ctx.isFirstIterate = true;
    }
};

template <class Intf, uint32_t ImplType>
struct SetBias {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::BiasT> &bias)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (!Intf::isDeQuantFlag) {
                self->ctx.biasgm.SetGlobalBuffer(bias.GetPhyAddr(0), bias.GetSize());
                self->ctx.isFirstIterate = true;
                self->ctx.enableBias = true;
            }
        }

        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::isDeQuantFlag) {
                self->ctx.biasgm.SetGlobalBuffer(bias.GetPhyAddr(0), bias.GetSize());
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetScale {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::ScaleT> &scale)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (!Intf::isDeQuantFlag) {
                self->ctx.scalegm.SetGlobalBuffer(scale.GetPhyAddr(0), scale.GetSize());
                self->ctx.isFirstIterate = true;
                self->ctx.enableVectorQuant = true;
            }
        }

        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::isDeQuantFlag) {
                self->ctx.scalegm.SetGlobalBuffer(scale.GetPhyAddr(0), scale.GetSize());
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetFixpipeParams {
    static __aicore__ inline void call(Intf *self, const Extendconv2dFixpipeParams<typename Intf::ScaleT, typename Intf::ReluWeightT, typename Intf::ClipValue0T, typename Intf::ClipValue1T>& fixpipeParams)
    {
        if ASCEND_IS_AIC_CONV {
            if (self->ctx.convTiling->quantMode0 != static_cast<uint8_t>(QuantModeType::NO_QUANT)) {
                self->ctx.scalegm.SetGlobalBuffer(fixpipeParams.scale0.GetPhyAddr(0), fixpipeParams.scale0.GetSize());
                if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                    self->ctx.deqScalar0 = fixpipeParams.scale0.GetValue(0);
                }
            }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
            if (self->ctx.convTiling->reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                float m2 = fixpipeParams.reluWeight0.GetValue(0);
                self->ctx.preReluScalar0 = reinterpret_cast<uint64_t&>(m2);
            }
#endif
            if constexpr (Intf::isExtendConv2d) {
                if (self->ctx.convTiling->dualOutput &&
                    self->ctx.convTiling->quantMode1 != static_cast<uint8_t>(QuantModeType::NO_QUANT)) {
                    self->ctx.scale1gm.SetGlobalBuffer(fixpipeParams.scale1.GetPhyAddr(0), fixpipeParams.scale1.GetSize());
                    if (self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                        self->ctx.deqScalar1 = fixpipeParams.scale1.GetValue(0);
                    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                    if (self->ctx.convTiling->reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                        float m2 = fixpipeParams.reluWeight1.GetValue(0);
                        self->ctx.preReluScalar1 = reinterpret_cast<uint64_t&>(m2);
                    }
#endif
                }
                if (self->ctx.convTiling->quantMode0 != static_cast<uint8_t>(QuantModeType::NO_QUANT) ||
                    self->ctx.convTiling->quantMode1 != static_cast<uint8_t>(QuantModeType::NO_QUANT)) {
                    self->ctx.isFirstIterate = true;
                }
                if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
                    self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.enableVectorQuant = true;
                }
            } else {
                if (self->ctx.convTiling->quantMode0 != static_cast<uint8_t>(QuantModeType::NO_QUANT)) {
                    self->ctx.isFirstIterate = true;
                }
                if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.enableVectorQuant = true;
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct GetTensorC {
    template <template <typename> class TensorTypeT, const FixpipeConfig &config, bool sync = true>
    static __aicore__ inline void call(
        Intf *self, const TensorTypeT<typename Intf::OutputT> &output, CopyUbInfo* ubInfo = nullptr, bool enSequentialWrite = false)
    {
        GlobalTensor<typename Intf::Output1T> output1;
        GetTensorC<Intf, ImplType>::call<TensorTypeT, config>(self, output, output1, ubInfo);
    }

    template <template <typename> class TensorTypeT, const FixpipeConfig &config, bool sync = true>
    static __aicore__ inline void call(
        Intf *self, const TensorTypeT<typename Intf::OutputT> &output0, const GlobalTensor<typename Intf::Output1T> &output1,
        CopyUbInfo* ubInfo = nullptr, bool enSequentialWrite = false)
    {
        if ASCEND_IS_AIC_CONV {
            self->ctx.copyOutIns.template CopyOut<TensorTypeT, config>(output0, ubInfo);
            if constexpr (Intf::isExtendConv2d) {
                if (self->ctx.convTiling->dualOutput) {
                    self->ctx.copyOutIns1.template CopyOut<GlobalTensor, config>(output1, ubInfo);
                }
            }

            if constexpr (Intf::isInnerBatchFlag) {
                self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
            }

            if (self->ctx.enableVectorQuant && !self->ctx.convTiling->fixpParamsFullLoadFlag) {
                self->ctx.queueScaleL1.FreeTensor(self->ctx.scaleL1);
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct ConvPreProcess {
    static __aicore__ inline void call(Intf *self)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (!Intf::isDeQuantFlag) {
                if (self->ctx.convTiling->biasFullLoadFlag && self->ctx.enableBias) {
                    self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                    self->ctx.loadBiasL1Ins.LoadChannelWiseL1FullLoad(self->ctx.biasL1, self->ctx.biasgm,
                        self->ctx.singleCoreCo, 0);
                    self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                    self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
                }
                if (self->ctx.convTiling->fixpParamsFullLoadFlag && self->ctx.enableVectorQuant) {
                    if constexpr (Intf::groupType != static_cast<int8_t>(ConvGroupType::NORMAL_CONV)) {
                        event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::FIX_MTE2));
                        SetFlag<HardEvent::FIX_MTE2>(eventId);
                        WaitFlag<HardEvent::FIX_MTE2>(eventId);
                    }
                    self->ctx.scaleL1 = self->ctx.queueScaleL1.template AllocTensor<typename Intf::ScaleT>();
                    if constexpr (Intf::isExtendConv2d) {
                        if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            self->ctx.loadScaleL1Ins.LoadChannelWiseL1FullLoad(self->ctx.scaleL1, self->ctx.scalegm,
                                self->ctx.singleCoreCo, 0);
                        }
                        if (self->ctx.convTiling->dualOutput &&
                            self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            self->ctx.loadScaleL1Ins.LoadChannelWiseL1FullLoad(self->ctx.scaleL1[self->ctx.scale1L1offset],
                                self->ctx.scale1gm, self->ctx.singleCoreCo, 0);
                        }
                    } else {
                        self->ctx.loadScaleL1Ins.LoadChannelWiseL1FullLoad(self->ctx.scaleL1, self->ctx.scalegm,
                            self->ctx.singleCoreCo, 0);
                    }
                    self->ctx.queueScaleL1.EnQue(self->ctx.scaleL1);
                    self->ctx.scaleL1 = self->ctx.queueScaleL1.template DeQue<typename Intf::ScaleT>();
                }
            }
        }

        self->ctx.cl0PingPongFlag = 0;
    }
};

template <class Intf, uint32_t ImplType>
struct ConvPostProcess {
    static __aicore__ inline void call(Intf *self)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (!Intf::isDeQuantFlag) {
                if (self->ctx.convTiling->biasFullLoadFlag && self->ctx.enableBias) {
                    self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
                }
                if (self->ctx.convTiling->fixpParamsFullLoadFlag && self->ctx.enableVectorQuant) {
                    self->ctx.queueScaleL1.FreeTensor(self->ctx.scaleL1);
                }
            }

            self->ctx.isFirstIterate = true;
        }
    }
};

template <class Intf, uint32_t ImplType>
struct IterateAll {
    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        GlobalTensor<typename Intf::Output1T> output1; // fake output1 tensor
        return IterateAll<Intf, ImplType>::call(self, output, output1, enPartialSum);
    }

    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output0,
        const GlobalTensor<typename Intf::Output1T> &output1, bool enPartialSum = false)
    {
        ConvPreProcess<Intf, ImplType>::call(self);

        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::isDeQuantFlag) {
                self->ctx.dequantUB2GmTools.SetOutputGm(output0);
            }
        }

        while (Iterate<Intf, ImplType>::call(self, enPartialSum)) {
            if ASCEND_IS_AIC_CONV {
                if constexpr (Intf::isDeQuantFlag) {
                    self->ctx.dequantL0C2UBTools.CopyOut();
                    self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
                } else if constexpr (Intf::isExtendConv2d) {
                    IterateAll<Intf, ImplType>::GetExtendTensorC(self, output0, output1);
                } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
                    if constexpr (Intf::isFixedPoint) {
                        GetTensorC<Intf, ImplType>::template
                            call<GlobalTensor, CFG_ROW_MAJOR_FIXED_POINT, sync>(self, output0);
                    } else {
                        GetTensorC<Intf, ImplType>::template
                            call<GlobalTensor, CFG_ROW_MAJOR, sync>(self, output0);
                    }
                } else {
                    if constexpr (Intf::isFixedPoint) {
                        GetTensorC<Intf, ImplType>::template
                            call<GlobalTensor, CFG_COLUMN_MAJOR_FIXED_POINT, sync>(self, output0);
                    } else {
                        GetTensorC<Intf, ImplType>::template
                            call<GlobalTensor, CFG_COLUMN_MAJOR, sync>(self, output0);
                    }
                }
            }
        }

        ConvPostProcess<Intf, ImplType>::call(self);

        return false;
    }

    template <bool sync = true>
    static __aicore__ inline void GetExtendTensorC(Intf *self, const GlobalTensor<typename Intf::OutputT> &output0,
                                                   const GlobalTensor<typename Intf::Output1T> &output1) {
        if constexpr (Intf::formatOutput == ConvFormat::NHWC) {
            if constexpr (Intf::isFixedPoint) {
                if (self->ctx.convTiling->dualOutput) {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_ROW_MAJOR_FIXED_POINT, sync>(self, output0, output1);
                } else {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_ROW_MAJOR_FIXED_POINT, sync>(self, output0);
                }
            } else {
                if (self->ctx.convTiling->dualOutput) {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_ROW_MAJOR, sync>(self, output0, output1);
                } else {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_ROW_MAJOR, sync>(self, output0);
                }
            }
        } else {
            if constexpr (Intf::isFixedPoint) {
                if (self->ctx.convTiling->dualOutput) {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_COLUMN_MAJOR_FIXED_POINT, sync>(self, output0, output1);
                } else {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_COLUMN_MAJOR_FIXED_POINT, sync>(self, output0);
                }
            } else {
                if (self->ctx.convTiling->dualOutput) {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_COLUMN_MAJOR, sync>(self, output0, output1);
                } else {
                    GetTensorC<Intf, ImplType>::template
                        call<GlobalTensor, CFG_COLUMN_MAJOR, sync>(self, output0);
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct End {
    static __aicore__ inline void call(Intf *self)
    {
        if ASCEND_IS_AIC_CONV {
            self->ctx.queueAL1.FreeAllEvent();
            self->ctx.queueBL1.FreeAllEvent();
            if constexpr (!Intf::isDeQuantFlag) {
                self->ctx.queueBiasL1.FreeAllEvent();
                self->ctx.queueScaleL1.FreeAllEvent();
                self->ctx.queueBiasBT.FreeAllEvent();
            }

            if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
                self->ctx.queueCL0.FreeAllEvent();
            }
        }
    }
};

template <class Intf>
__aicore__ inline void InitBufferWithDoubleBuf(Intf *self)
{
    self->ctx.pipe.InitBuffer(self->ctx.al0Buf, L0A_SIZE);
    self->ctx.pipe.InitBuffer(self->ctx.bl0Buf, L0B_SIZE);
    self->ctx.wholeAl0Tensor = self->ctx.al0Buf.template Get<typename Intf::FmapT>();
    self->ctx.wholeBl0Tensor = self->ctx.bl0Buf.template Get<typename Intf::WeightT>();

    int8_t cl0db = (self->ctx.convTiling->pBufferFlag & 0x04) >> 2;
    int8_t al1db = (self->ctx.convTiling->pBufferFlag & 0x08) >> 3;
    int8_t bl1db = (self->ctx.convTiling->pBufferFlag & 0x10) >> 4;

    if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
        uint64_t cl0Spacesize = self->ctx.convTiling->mStep * self->ctx.convTiling->nL0;
        if constexpr (Intf::isInnerBatchFlag) {
            cl0Spacesize *= self->ctx.convTiling->innerBatch;
        }
        if (!cl0db) {
            self->ctx.pipe.InitBuffer(self->ctx.queueCL0, 1, cl0Spacesize * Intf::sizeOfL0c);
        } else {
            self->ctx.pipe.InitBuffer(self->ctx.queueCL0, DOUBLE_BUF, cl0Spacesize * Intf::sizeOfL0c);
        }
    } else {
        self->ctx.pipe.InitBuffer(self->ctx.l0cBuf, L0C_SIZE);
        self->ctx.wholeCl0Tensor = self->ctx.l0cBuf.template Get<typename Intf::L0cT>();
    }

    if (!al1db) {
        self->ctx.pipe.InitBuffer(self->ctx.queueAL1, 1, self->ctx.convTiling->aL1SpaceSize);
    } else {
        self->ctx.pipe.InitBuffer(self->ctx.queueAL1, DOUBLE_BUF, self->ctx.convTiling->aL1SpaceSize);
    }

    if constexpr (!Intf::bL1DBFlag) {
        self->ctx.pipe.InitBuffer(
            self->ctx.queueBL1, 1, self->ctx.bL1SpaceSize * Intf::sizeOfWeight);
    } else {
        if constexpr (Intf::weightUbTrans) {
            self->ctx.pipe.InitBuffer(self->ctx.bL1TBuf, self->ctx.bL1SpaceSize * DOUBLE_BUF * Intf::sizeOfWeight);
            self->ctx.bL1Tensor = self->ctx.bL1TBuf.template Get<typename Intf::WeightT>();
        } else {
            self->ctx.pipe.InitBuffer(self->ctx.queueBL1, DOUBLE_BUF, self->ctx.bL1SpaceSize * Intf::sizeOfWeight);
        }
    }

    if constexpr (Intf::isDeQuantFlag) {
        self->ctx.pipe.InitBuffer(self->ctx.mmadResUbBuf,
            self->ctx.convTiling->mUB * self->ctx.convTiling->nUB * Intf::sizeOfL0c);
        self->ctx.mmadResUbTensor = self->ctx.mmadResUbBuf.template Get<typename Intf::L0cT>();
    }
}

template <class Intf>
__aicore__ inline void InitBuffer(Intf *self)
{
    self->ctx.bL1SpaceSize = self->ctx.convTiling->nBL1 * self->ctx.convTiling->kBL1;

    InitBufferWithDoubleBuf<Intf>(self);
    
    if constexpr (!Intf::isDeQuantFlag) {
        if (self->ctx.convTiling->hasBias) {
            uint64_t biasl1Spacesize = self->ctx.convTiling->biasFullLoadFlag ? AlignB(
                self->ctx.singleCoreCo * Intf::sizeOfBias, BLOCK_L0_N * Intf::sizeOfBias) :
                self->ctx.convTiling->nL0 * Intf::sizeOfBias;
            uint64_t biasBTSpacesize = self->ctx.convTiling->nL0;
            self->ctx.pipe.InitBuffer(self->ctx.queueBiasL1, 1, AlignB(biasl1Spacesize, C0_SIZE));
            self->ctx.pipe.InitBuffer(
                self->ctx.queueBiasBT, 1, AlignB(biasBTSpacesize * Intf::sizeOfL0c, BT_SIZE));
        }
    }

    if constexpr (Intf::isExtendConv2d) {
        uint64_t scaleL1SpaceSize = 0;
        uint64_t scale0L1Size = self->ctx.convTiling->fixpParamsFullLoadFlag ? AlignB(
            self->ctx.singleCoreCo * Intf::sizeOfScale, BLOCK_L0_N * Intf::sizeOfScale) :
            self->ctx.convTiling->nL0 * Intf::sizeOfScale;
        if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scaleL1SpaceSize += scale0L1Size;
        }
        if (self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            self->ctx.scale1L1offset = scaleL1SpaceSize / Intf::sizeOfScale;
            scaleL1SpaceSize += scale0L1Size;
        }
        if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
            self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            self->ctx.pipe.InitBuffer(self->ctx.queueScaleL1, 1, AlignB(scaleL1SpaceSize, C0_SIZE));
        }
    } else if constexpr (Intf::isQuantScene) {
        if (self->ctx.convTiling->hasScale != 0) {
            uint64_t scaleL1SpaceSize = self->ctx.convTiling->fixpParamsFullLoadFlag ? AlignB(
                self->ctx.singleCoreCo * Intf::sizeOfScale, BLOCK_L0_N * Intf::sizeOfScale) :
                self->ctx.convTiling->nL0 * Intf::sizeOfScale;
            self->ctx.pipe.InitBuffer(self->ctx.queueScaleL1, 1, AlignB(scaleL1SpaceSize, C0_SIZE));
        }
    }
}

template <class Intf>
__aicore__ inline void InitHf32Mode(Intf *self)
{
    if (self->ctx.convTiling->hf32Enable) {
        SetHF32Mode(self->ctx.convTiling->hf32Enable);
        SetHF32TransMode(self->ctx.convTiling->hf32TransMode);
    } else {
        SetHF32Mode(0);
        SetHF32TransMode(0);
    }
}

template <class Intf>
__aicore__ inline void InitSubApiParams(Intf *self)
{
    self->ctx.loadBL1Ins.SetParams(self);
    self->ctx.loadAl1Ins.SetParams(self);
    self->ctx.loadAL0Ins.SetParams(self);
    self->ctx.loadBL0Ins.SetParams(self);
    self->ctx.madIns.SetParams(self);
    if constexpr (!Intf::isDeQuantFlag) {
        self->ctx.loadBiasL1Ins.SetParams(self);
        self->ctx.loadScaleL1Ins.SetParams(self);
        self->ctx.loadBiasBTIns.SetParams(self);
        self->ctx.copyOutIns.SetParams(self);
        if constexpr (Intf::isExtendConv2d) {
            self->ctx.copyOutIns1.SetParams(self);
        }
    } else {
        self->ctx.dequantL0C2UBTools.SetParams(self);
    }
}

}  // namespace ConvFunc
#endif