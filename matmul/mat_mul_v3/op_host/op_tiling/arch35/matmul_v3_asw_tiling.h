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
 * \file matmul_v3_asw_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3AswTiling : public MatMulV3BaseTiling {
public:
    MatMulV3AswTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3BaseTiling(context, cfg) {};

    ~MatMulV3AswTiling() override {};

protected:
    bool IsCapable() override
    {
        return true;
    };

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    MatMulV3Model aswtModel_ {MatMulV3Model::BASIC};

private:
    void FormulateBasicBlock();
    void CalcTailBasicBlock();
    void OptimizeEdgeBasicBlock();
    void GetOuterAxisTailCnt(const bool nLoadBalance, uint64_t& baseTailSplitCnt, uint64_t& tailMain);
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_TILING_H__