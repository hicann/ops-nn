/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_softmax_v2_tiling.h
 * \brief
 */

#ifndef LOG_SOFTMAX_V2_TILING_BASE_H_
#define LOG_SOFTMAX_V2_TILING_BASE_H_

#include "../../../softmax_v2/op_host/arch35/softmax_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling
{
class LogSoftmaxV2TilingBase : virtual public SoftmaxV2TilingBase
{
public:
    explicit LogSoftmaxV2TilingBase(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~LogSoftmaxV2TilingBase() override = default;

protected:
    ge::graphStatus GetAndCheckDtypes() override;
};

}  // namespace optiling

#endif  // LOGSOFTMAX_V2_TILING_BASE_H_