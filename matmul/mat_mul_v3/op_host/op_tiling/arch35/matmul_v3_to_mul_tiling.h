/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_to_mul_tiling.h
 * \brief
 */
#pragma once

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3ToMulTiling : public MatMulV3BaseTiling {
public:
    MatMulV3ToMulTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3BaseTiling(context, cfg) {};

    ~MatMulV3ToMulTiling() override = default;
protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetTilingData(TilingResult &tiling) const override;
private:
    static constexpr uint64_t BASE_MN = 128;
    static constexpr uint64_t BASE_K = 128;
    static constexpr uint64_t LIMIT_K = 512;
    static constexpr uint64_t ALIGN_NUM = 8;
};
} // namespace matmul_v3
} // namespace optiling

