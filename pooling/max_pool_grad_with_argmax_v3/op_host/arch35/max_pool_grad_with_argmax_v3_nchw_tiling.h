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
 * \file max_pool_grad_with_argmax_v3_nchw_tiling.h
 * \brief MaxPoolGradWithArgmaxV3 NCHW格式Tiling实现，继承自公共基类
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_H_

#include "max_pool_grad_with_argmax_v3_tiling_base.h"
#include "../../../pool_grad_common/op_host/arch35/max_pool_grad_nchw_tiling_common.h"

namespace optiling {

class MaxPoolGradWithArgmaxV3NCHWTiling : public MaxPoolGradWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolGradWithArgmaxV3NCHWTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context), nchwTilingCommon(&inputData)
    {}

    ~MaxPoolGradWithArgmaxV3NCHWTiling() override
    {}

private:
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPoolGradNCHWTilingCommon nchwTilingCommon;
};

} // namespace optiling

#endif