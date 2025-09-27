/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file convolution.h
 * \brief
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *Conv2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv2dV2NCHW(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                              op::DataType outputDtype, const aclIntArray *stride, const aclIntArray *padding,
                              const aclIntArray *dilation, int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv2d5HdFp1625HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                       const aclIntArray *stride, const aclIntArray *padding,
                                       const aclIntArray *dilation, int groups, aclOpExecutor *executor);

const aclTensor *Conv2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                 const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                 int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                 const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                 int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                 const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                 int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                   const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                   int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                   const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                   int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                   const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                   int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWHif8(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                   const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                   int groups, aclOpExecutor *executor);

const aclTensor *QuantConv3d6HdInt8To6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                             const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                             const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                             aclOpExecutor *executor);

const aclTensor *QuantConv3d6HdInt8To6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                             const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                             const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                             aclOpExecutor *executor);

const aclTensor *Conv3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdHif8(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdF8e4m3fn(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor);

const aclTensor *ConvTranspose3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);
const aclTensor *ConvTranspose3d6HdHif8(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);
const aclTensor *ConvTranspose3d6HdF8e4m3fn(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

bool IsSupportConv3DToConv3DV2();
bool IsSupportConv1DTransposeTo3D();
bool IsSupportConv2DTransposeTo3D(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                  const aclIntArray *stride, const aclIntArray *padding,
                                  const aclIntArray *dilation, const aclIntArray *outputPadding,
                                  const int64_t groups, aclTensor *output);
}  // namespace l0op

#endif  // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H
