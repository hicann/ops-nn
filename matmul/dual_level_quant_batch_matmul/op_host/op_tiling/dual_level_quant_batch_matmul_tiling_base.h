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
 * \file dual_level_quant_batch_matmul_tiling_base.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_BASE_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_BASE_H

#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "dual_level_quant_batch_matmul_tiling_tool.h"
#include "../../op_kernel/dual_level_quant_batch_matmul_tiling_data.h"
#include "dual_level_quant_batch_matmul_checker.h"

namespace optiling {
struct DualLevelQuantBatchMatmulCompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    NpuArch npuArch;
};

class DualLevelQuantBatchMatmulBaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    using TilingBaseClass::Reset;

    explicit DualLevelQuantBatchMatmulBaseTiling(gert::TilingContext* context);

    ~DualLevelQuantBatchMatmulBaseTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;

    bool IsCapable() override;

    void InitCompileInfo();

    bool InitMatmulInfo();

    bool SetPlatformInfoForTiling();

    // 输入信息
    DualLevelQuantBatchMatmulInfo matmulInfo_;
    // 平台相关信息
    DualLevelQuantBatchMatmulCompileInfo compileInfo_;
    bool isCompileInfoInit = false;
    size_t tilingDataSize_ = 0;
};
} // namespace optiling
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_H
