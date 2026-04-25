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
 * \file max_pool_grad_tiling_simt.h
 * \brief simt imply for max_pool_grad
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_SIMT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_SIMT_TILING_H

#include <array>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "max_pool_grad_tiling.h"
#include "../../pool_grad_common/op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"

using namespace MaxPoolGradWithArgmaxNHWCNameSpace;
namespace optiling {
constexpr int64_t MAX_THREAD_NUM = 256;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
class MaxPoolGradTiling : public MaxPoolGradTilingBase {
public:
    explicit MaxPoolGradTiling(gert::TilingContext* context) : MaxPoolGradTilingBase(context)
    {}

    ~MaxPoolGradTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    MaxPoolGradWithArgmaxSimtTilingCommonData* tilingData_ =
        context_->GetTilingData<MaxPoolGradWithArgmaxSimtTilingCommonData>();
    int64_t outShapeSize = 0;
    int64_t outputDataCount = 0;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
};

} // namespace optiling
#endif // MAX_POOL_GRAD_V2_SIMT_TILING_H