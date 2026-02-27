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
 * \file max_pool_grad_with_argmax_simt_tiling_common.h
 * \brief 
 */
#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMT_TILING_COMMON_H
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMT_TILING_COMMON_H

#include <array>
#include "../op_kernel/max_pool_grad_with_argmax_struct_common.h"
#include "max_pool_grad_with_argmax_tiling_common.h"

namespace optiling {

class MaxPoolGradWithArgmaxSIMTTilingCommon{
public:
    explicit MaxPoolGradWithArgmaxSIMTTilingCommon(MaxPoolGradWithArgmaxInputInfoCommon* input)
            :inputData(input)
    {
    }
    ge::graphStatus DoOpTiling(gert::TilingContext* context_);
    ge::graphStatus PostTiling(gert::TilingContext* context_, MaxPoolGradWithArgmaxHardwareInfo hwinfo);

private: 
    void SetTilingData(gert::TilingContext* context);
    MaxPoolGradWithArgmaxInputInfoCommon* inputData;
    MaxPoolGradWithArgmaxHardwareInfo hardWare;
};

}  // namespace optiling

#endif  