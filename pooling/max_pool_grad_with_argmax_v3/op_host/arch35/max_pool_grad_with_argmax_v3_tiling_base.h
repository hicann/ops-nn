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
 * \file max_pool_grad_with_argmax_v3_tiling_base.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_V3_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_V3_TILING_BASE_H_

#include "op_common/op_host/util/platform_util.h"
#include "../../../max_pool_grad_with_argmax_common/op_host/max_pool_grad_with_argmax_tiling_common.h"
#include "../../../max_pool_grad_with_argmax_common/op_kernel/max_pool_grad_with_argmax_struct_common.h"

using namespace std;

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3TilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3TilingData);

class MaxPoolGradWithArgmaxV3BaseTiling : public MaxPoolGradWithArgmaxTilingCommon {
public:
    explicit MaxPoolGradWithArgmaxV3BaseTiling(gert::TilingContext* context) : MaxPoolGradWithArgmaxTilingCommon(context)
    {}

    ~MaxPoolGradWithArgmaxV3BaseTiling() override
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
};
} // namespace optiling
#endif