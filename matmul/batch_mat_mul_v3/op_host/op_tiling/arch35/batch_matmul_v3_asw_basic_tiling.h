/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_asw_basic_tiling.h
 * \brief
 */

#ifndef __OP_HOST_BATCH_MATMUL_V3_ASW_BASIC_TILING_H__
#define __OP_HOST_BATCH_MATMUL_V3_ASW_BASIC_TILING_H__

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace matmul_v3_advanced;
class BatchMatMulV3AswBasicTiling : public MatMulV3BaseTiling {
public:
    BatchMatMulV3AswBasicTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3BaseTiling(context, cfg) {};

    ~BatchMatMulV3AswBasicTiling() override {};

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    uint64_t GetBlockDim() const override;

    MatMulV3Model aswtModel_{MatMulV3Model::BASIC};
};
}
}
#endif // __OP_HOST_BATCH_MATMUL_V3_ASW_BAISC_TILING_H__