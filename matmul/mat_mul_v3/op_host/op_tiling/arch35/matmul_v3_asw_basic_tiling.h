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
 * \file matmul_v3_asw_basic_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"
#include "matmul_v3_asw_tiling.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3AswBasicApiTiling : public MatMulV3AswTiling {
public:
    MatMulV3AswBasicApiTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3AswTiling(context, cfg) {};

    ~MatMulV3AswBasicApiTiling() override {};

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__