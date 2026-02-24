/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_softmax_grad_tiling.h
 * \brief
 */

#ifndef LOG_SOFTMAX_GRAD_TILING_BASE_H
#define LOG_SOFTMAX_GRAD_TILING_BASE_H

#include "../../../softmax_grad/op_host/arch35/softmax_grad_tiling.h"

namespace optiling
{
class LogSoftmaxGradTilingBase : virtual public SoftmaxGradTilingBase
{
public:
    explicit LogSoftmaxGradTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~LogSoftmaxGradTilingBase() override = default;

protected:
    ge::graphStatus GetAndCheckDtypes() override;
};

}  // namespace optiling

#endif  // LOG_SOFTMAX_GRAD_TILING_BASE_H
