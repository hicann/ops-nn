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
 * \file matmul_v3_stream_k_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
#define __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3StreamKTiling : public MatMulV3BaseTiling {
public:
    MatMulV3StreamKTiling(gert::TilingContext *context, MatMulTilingCfg &cfg) : MatMulV3BaseTiling(context, cfg) {}

    ~MatMulV3StreamKTiling() override {}

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    std::vector<size_t> GetWorkspaceSize() const override;

    ge::graphStatus GetTilingData(TilingResult &tiling) const override;

private:
    bool CheckStreamKSKTiling() const;
    bool CheckStreamKDPSKTiling() const;
    MatMulV3L0C2Out GetL0C2OutFlag() const;

    uint64_t mCnt_{ 1 };
    uint64_t nCnt_{ 1 };
    uint64_t totalMNCnt_{ 1 };
    MatMulV3L0C2Out l0C2Out_{MatMulV3L0C2Out::ON_THE_FLY};
};
}
}

#endif // __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
