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
 * \file fused_matmul_builtin_tiling.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_H__
#define __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_H__

#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_advanced.h"
#include "platform/platform_ascendc.h"

namespace optiling {
namespace fused_matmul {
using namespace batch_matmul_v3_advanced;
class FusedMatMulBuiltInTiling : public BatchMatMulV3Tiling {
public:
    explicit FusedMatMulBuiltInTiling(gert::TilingContext* context) : BatchMatMulV3Tiling(context){};

    ~FusedMatMulBuiltInTiling() override = default;

    ge::graphStatus DoTiling() override;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetArgs() override;
    ge::graphStatus CheckArgs() override;

private:
    platform_ascendc::SocVersion socVersion_;
};
} // namespace fused_matmul
} // namespace optiling
#endif // __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_H__
