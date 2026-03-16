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
 * \file pp_matmul_int8_tiling.h
 * \brief
 */
#ifndef PP_MATMUL_INT8_TILING_H
#define PP_MATMUL_INT8_TILING_H
#include "util/math_util.h"
#include "../quant_batch_matmul_v3_tiling_base.h"
#include "../../../../transpose_batch_mat_mul/op_host/op_tiling/transpose_batch_mat_mul_einsum_tiling.h"

namespace optiling {

class PpMatmulInt8Tiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit PpMatmulInt8Tiling(gert::TilingContext *context);
    ~PpMatmulInt8Tiling() override = default;

    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void Reset();

protected:
    pp_matmul::PpMatmulDefaultTilingData ppMatmulDefaultTilingData_{};
    PpMatmulTilingData tilingDataSelf_{};
    PpMatmulTilingData &tilingData_;
};
}  // namespace optiling
#endif  // PP_MATMUL_INT8_TILING_H