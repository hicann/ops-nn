/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_compress_tiling_base.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_COMPRESS_TILING_BASE_H__
#define __OP_HOST_MATMUL_COMPRESS_TILING_BASE_H__

#include "register/op_def_registry.h"
#include "../../../transpose_batch_mat_mul/op_host/op_tiling/pp_matmul_default.h"
#include "../../op_kernel/matmul_compress_tiling_data.h"

namespace optiling {
namespace matmul_compress {

class MatmulCompressTilingBase{
public:
    explicit MatmulCompressTilingBase(gert::TilingContext* context) 
        : context_(context),
          tiling_(context),
          params_(tiling_.matMulInfo_),
          hwInfo_(tiling_.hardwareInfo_),
          tilingData_(tiling_.ppMatmulDefaultTilingData_)
    {}
    ~MatmulCompressTilingBase() = default;

    bool GetMatMulInfo();
    bool GetTilingKey();
    bool GetMatMulTilingData();
    ge::graphStatus DoTiling();
    void PrintTiling();
    ge::graphStatus PostTiling();
    gert::TilingContext *context_ = nullptr;

private:
    uint64_t tilingKey_{0};
    MatmulCompressTilingDataArch20 matmulCompressTilingDataArch20_{};
    pp_matmul::PpMatMulDefault tiling_;
    pp_matmul::MatMulInfo& params_;
    pp_matmul::HardwareInfo& hwInfo_;
    pp_matmul::PpMatmulDefaultTilingData& tilingData_;
};
}
}
#endif
