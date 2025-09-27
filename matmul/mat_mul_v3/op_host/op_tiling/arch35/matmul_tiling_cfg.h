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
 * \file matmul_tiling_cfg.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_TILING_CFG_H__
#define __OP_HOST_MATMUL_TILING_CFG_H__

#include <cstdint>
#include <cstddef>
#include <vector>

namespace optiling {
struct TilingResult {
    uint64_t tilingKey;
    uint64_t blockDim;
    void *tilingData;
    size_t tilingDataSize;
    std::vector<size_t> workspaceSize;
};

class MatMulTilingCfg {
public:
    MatMulTilingCfg(bool needUpdateIn, const void *compileInfoIn, const void *argsIn)
        : needUpdate(needUpdateIn), compileInfo(compileInfoIn), args(argsIn)
    {}

    virtual ~MatMulTilingCfg() {};

    virtual void Update(const TilingResult &result) {
        (void)result;
    };

public:
    const bool needUpdate = false;     // true: Tiling结果通过Update返回； false: Tiling结果填充到context中
    const void *compileInfo = nullptr; // 编译信息，用于生成TilingKey
    const void *args = nullptr;        // 算子参数，用于生成TilingKey
};
} // namespace optiling

#endif // __OP_HOST_MATMUL_TILING_CFG_H__
