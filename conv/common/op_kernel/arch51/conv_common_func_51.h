/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_common_func_51.h
 * \brief
 */

#ifndef CONV_COMMON_FUNC_5162_H
#define CONV_COMMON_FUNC_5162_H

#include "../arch35/conv_common_func_arch35.h"

namespace ConvFunc {

template <class Intf, uint32_t ImplType>
template <template <typename> class TensorTypeT, const FixpipeConfig& config, bool sync>
__aicore__ inline void GetTensorC<Intf, ImplType>::call(Intf* self, const TensorTypeT<typename Intf::OutputT>& output,
                                                        IterateParam& iterParam, CopyUbInfo* ubInfo,
                                                        bool enSequentialWrite)
{
    GlobalTensor<typename Intf::Output1T> output1;
    GetTensorC<Intf, ImplType>::call<TensorTypeT, config>(self, output, output1, iterParam, ubInfo);
}

template <class Intf, uint32_t ImplType>
template <template <typename> class TensorTypeT, const FixpipeConfig& config, bool sync>
__aicore__ inline void GetTensorC<Intf, ImplType>::call(Intf* self, const TensorTypeT<typename Intf::OutputT>& output0,
                                                        const GlobalTensor<typename Intf::Output1T>& output1,
                                                        IterateParam& iterParam, CopyUbInfo* ubInfo,
                                                        bool enSequentialWrite)
{
    if ASCEND_IS_AIC_CONV {
        self->ctx.copyOutIns.template CopyOut<TensorTypeT, config>(output0, iterParam, ubInfo);
        if constexpr (Intf::isExtendConv2d) {
            if (self->ctx.convTilingData->dualOutput) {
                self->ctx.copyOutIns1.template CopyOut<GlobalTensor, config>(output1, iterParam, ubInfo);
            }
        }

        if constexpr (Intf::isInnerBatchFlag) {
            self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
        }

        if (self->ctx.enableVectorQuant && !self->ctx.convTilingData->fixpParamsFullLoadFlag) {
            self->ctx.queueScaleL1.FreeTensor(self->ctx.scaleL1);
        }

        if (self->ctx.enableVectorRelu && !self->ctx.convTilingData->fixpParamsFullLoadFlag) {
            self->ctx.queueReluWeightL1.FreeTensor(self->ctx.reluWeightL1);
        }
    }
}

template <class Intf, uint32_t ImplType>
template <bool sync>
__aicore__ inline void IterateAll<Intf, ImplType>::IterateProcess(Intf* self,
                                                                  const GlobalTensor<typename Intf::OutputT>& output0,
                                                                  const GlobalTensor<typename Intf::Output1T>& output1,
                                                                  IterateParam& iterParam, bool enPartialSum)
{
    while (Iterate<Intf, ImplType>::call(self, iterParam, enPartialSum)) {
        if ASCEND_IS_AIC_CONV {
            if constexpr (Intf::isDeQuantFlag) {
                self->ctx.dequantL0C2UBTools.CopyOut();
                self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
            } else if constexpr (Intf::isExtendConv2d) {
                IterateAll<Intf, ImplType>::GetExtendTensorC(self, output0, output1, iterParam);
            } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
                if constexpr (Intf::isFixedPoint) {
                    GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_ROW_MAJOR_FIXED_POINT, sync>(
                        self, output0, iterParam);
                } else {
                    GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_ROW_MAJOR, sync>(self, output0,
                                                                                                 iterParam);
                }
            } else {
                if constexpr (Intf::isFixedPoint) {
                    GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_COLUMN_MAJOR_FIXED_POINT, sync>(
                        self, output0, iterParam);
                } else {
                    GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_COLUMN_MAJOR, sync>(self, output0,
                                                                                                    iterParam);
                }
            }
        }
    }
}

template <class Intf, uint32_t ImplType>
template <bool sync>
__aicore__ inline void IterateAll<Intf, ImplType>::GetExtendTensorC(
    Intf* self, const GlobalTensor<typename Intf::OutputT>& output0,
    const GlobalTensor<typename Intf::Output1T>& output1, IterateParam& iterParam)
{
    if constexpr (Intf::formatOutput == ConvFormat::NHWC) {
        if constexpr (Intf::isFixedPoint) {
            GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_ROW_MAJOR_FIXED_POINT, sync>(
                self, output0, output1, iterParam);
        } else {
            GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_ROW_MAJOR, sync>(self, output0, output1,
                                                                                         iterParam);
        }
    } else {
        if constexpr (Intf::isFixedPoint) {
            GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_COLUMN_MAJOR_FIXED_POINT, sync>(
                self, output0, output1, iterParam);
        } else {
            GetTensorC<Intf, ImplType>::template call<GlobalTensor, CFG_COLUMN_MAJOR, sync>(self, output0, output1,
                                                                                            iterParam);
        }
    }
}

} // namespace ConvFunc

#endif // CONV_COMMON_FUNC_5162_H
