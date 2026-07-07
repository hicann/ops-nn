/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace matmul_v3_advanced;
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;
class BatchMatMulV3IterBatchBroadcastBasicApiTiling : public MatMulV3BaseTiling {
public:
    BatchMatMulV3IterBatchBroadcastBasicApiTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3BaseTiling(context, cfg) {};

    ~BatchMatMulV3IterBatchBroadcastBasicApiTiling() override {};

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    uint64_t GetNumBlocks() const override;

    ge::graphStatus GetTilingData(TilingResult& tiling) const override;

    bool IsContiguousStride(StrideIndexPairs& strideIndexPairs) const;

    bool IsInputNonContiguous(uint32_t inputIdx) const;

private:
    uint64_t alignMValue_{0};
    uint64_t alignNValue_{0};
    uint64_t alignKValue_{0};
    uint64_t iterBatchL0A_{0};
    uint64_t iterBatchL0B_{0};
    uint64_t iterBatchL0C_{0};
    uint64_t iterBatchL1_{0};
    bool l0CanLoadBatch_{false};
    uint64_t broadcastAxisA_{4}; // max batch dim 4
    uint64_t broadcastAxisB_{4}; // max batch dim 4
    uint64_t sizeAOneBatch_{0};
    uint64_t sizeBOneBatch_{0};
    uint64_t sizeCOneBatch_{0};

    bool hasBroadcastAxis();
    bool CheckNonBroadcastAxisMatch() const;
    bool CheckL1IterBatch();
    bool CheckL0IterBatch();
    uint64_t GetABatchDim(int32_t idx) const;
    uint64_t GetBBatchDim(int32_t idx) const;
};
}
}
