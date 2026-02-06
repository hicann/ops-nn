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
 * \file max_pool_grad_with_argmax_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_TILING_H_

#include "../../max_pool_grad_with_argmax_common/op_host/max_pool_grad_with_argmax_tiling_common.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxTilingData);

class MaxPoolGradWithArgmaxBaseTiling : public MaxPoolGradWithArgmaxTilingCommon {
public:
    explicit MaxPoolGradWithArgmaxBaseTiling(gert::TilingContext* context) : MaxPoolGradWithArgmaxTilingCommon(context) {
    }

    ~MaxPoolGradWithArgmaxBaseTiling() override {
    }

    protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
};
}  // namespace optiling

#endif