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
 * \file log_softmax_v2_tiling_ara_full_load.cpp
 * \brief
 */

#include "log_softmax_v2_tiling.h"

using namespace ge;

namespace optiling
{
class LogSoftmaxV2ARATiling : public LogSoftmaxV2TilingBase, public SoftmaxV2ARATiling
{
public:
    explicit LogSoftmaxV2ARATiling(gert::TilingContext* context)
        : TilingBaseClass(context),
          SoftmaxV2TilingBase(context),
          LogSoftmaxV2TilingBase(context),
          SoftmaxV2ARATiling(context)
    {
    }
    ~LogSoftmaxV2ARATiling() override = default;
};

REGISTER_OPS_TILING_TEMPLATE(LogSoftmaxV2, LogSoftmaxV2ARATiling, TEMPLATE_ARA_FULL_LOAD_PRIORITY);
}  // namespace optiling
