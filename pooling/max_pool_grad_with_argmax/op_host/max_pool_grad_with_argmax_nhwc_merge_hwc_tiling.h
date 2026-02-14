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
 * \file max_pool_grad_with_argmax_nhwc_merge_hwc_tiling.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_NHWC_TILING_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_NHWC_TILING_H_

#include "max_pool_grad_with_argmax_tiling.h"
#include "../../max_pool_grad_with_argmax_common/op_host/max_pool_grad_with_argmax_nhwc_tiling_common.h"

namespace optiling
{
class MaxPoolGradWithArgmaxMergeHWCTiling : public MaxPoolGradWithArgmaxBaseTiling
{
public:
    explicit MaxPoolGradWithArgmaxMergeHWCTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxBaseTiling(context), 
          NHWCBase(new MaxPoolGradWithArgmaxNHWCTilingCommon(&inputData))
    {
    }

    ~MaxPoolGradWithArgmaxMergeHWCTiling() override
    {
        delete NHWCBase;
    }
    
private:
    MaxPoolGradWithArgmaxNHWCTilingCommon* NHWCBase;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
};

}  // namespace optiling

#endif