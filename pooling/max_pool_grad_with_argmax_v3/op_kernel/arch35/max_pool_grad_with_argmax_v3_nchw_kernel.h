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
 * \file max_pool_grad_with_argmax_v3_nchw_kernel.h
 * \brief MaxPoolGradWithArgmaxV3 NCHW格式Kernel实现，继承自公共基类
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../pool_grad_common/arch35/max_pool_grad_nchw_scatter_common.h"

namespace MaxPoolGradWithArgmaxV3NCHWNameSpace {

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
class MaxPoolGradWithArgmaxV3NCHWKernel
    : public MaxPoolGradNCHWNameSpace::MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE> {
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3NCHWKernel(void)
        : MaxPoolGradNCHWNameSpace::MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>(){};
};

} // namespace MaxPoolGradWithArgmaxV3NCHWNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_