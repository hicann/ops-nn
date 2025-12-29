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
 * \file fused_matmul_basic_streamk_tiling.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_MATMUL_BASIC_STREAMK_TILING_H__
#define __OP_HOST_FUSED_MATMUL_BASIC_STREAMK_TILING_H__

#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.h"

namespace optiling {
namespace fused_matmul {
using matmul_v3_advanced::MatMulV3BasicStreamKTiling;

class FusedMatMulStreamKTiling : public MatMulV3BasicStreamKTiling {
public:
    FusedMatMulStreamKTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : MatMulV3BasicStreamKTiling(context, cfg){};

    ~FusedMatMulStreamKTiling() override = default;

protected:
    bool IsCapable() override;
};
} // namespace fused_matmul
} // namespace optiling
#endif // __OP_HOST_FUSED_MATMUL_ASW_BASIC_TILING_H__