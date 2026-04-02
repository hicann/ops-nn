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
 * \file gemm_v3_base_tiling.h
 * \brief
 */

#ifndef GEMM_V3_BASE_TILING_H
#define GEMM_V3_BASE_TILING_H
#include "../../../transpose_batch_mat_mul/op_host/op_tiling/pp_matmul_default.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace gemm_v3 {

class GemmV3BaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GemmV3BaseTiling(gert::TilingContext* context)
        : TilingBaseClass(context), tiling_(context), params_(tiling_.matMulInfo_), hwInfo_(tiling_.hardwareInfo_),
          tilingData_(tiling_.ppMatmulDefaultTilingData_)
    {
    }
    ~GemmV3BaseTiling() override {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    bool InitParams();
    ge::graphStatus GetInputDims(const gert::Shape& shapeA, const gert::Shape& shapeB);

private:
    pp_matmul::PpMatMulDefault tiling_;
    pp_matmul::MatMulInfo& params_;
    pp_matmul::HardwareInfo& hwInfo_;
    pp_matmul::PpMatmulDefaultTilingData& tilingData_;
    uint32_t numBatchA_{0};
    uint32_t numBatchB_{0};
    float alpha_{0.0f};
    float beta_{0.0f};
};
} // namespace gemm_v3
} // namespace optiling
#endif // GEMM_V3_BASE_TILING_H
