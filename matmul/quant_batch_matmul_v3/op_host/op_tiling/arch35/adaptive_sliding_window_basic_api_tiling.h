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
 * \file adaptive_sliding_window_basic_api_tiling.h
 * \brief
 */
#ifndef ADAPTIVE_SLIDING_WINDOW_BASIC_API_TILING_H
#define ADAPTIVE_SLIDING_WINDOW_BASIC_API_TILING_H
#include "util/math_util.h"
#include "../quant_batch_matmul_v3_tiling_base.h"
#include "adaptive_sliding_window_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERGROUP_MODE = 0x1U << 4,
    PERBLOCK_MODE = 0x1U << 5,
};

#pragma pack(push, 8)
struct QuantBatchMatmulV3BasicAPIDataParams {
    uint32_t batchA = 0;
    uint32_t batchB = 0;
    uint32_t batchC = 0;
    uint32_t batchA1 = 0;
    uint32_t batchA2 = 0;
    uint32_t batchA3 = 0;
    uint32_t batchA4 = 0;
    uint32_t batchB1 = 0;
    uint32_t batchB2 = 0;
    uint32_t batchB3 = 0;
    uint32_t batchB4 = 0;
    uint32_t batchC1 = 0;
    uint32_t batchC2 = 0;
    uint32_t batchC3 = 0;
    uint32_t batchC4 = 0;
    uint32_t x1QuantMode = 0;
    uint32_t x2QuantMode = 0;
    uint32_t biasThreeDim = 0;
    uint32_t biasDtype = 0;
    uint32_t groupSizeM = 0;
    uint32_t groupSizeN = 0;
    uint32_t groupSizeK = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct BasicAPICubeTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t scaleKL1 = 0;
    uint32_t kL1 = 0;
    uint16_t usedCoreNum = 0;
    uint8_t scaleFactorA = 0;
    uint8_t scaleFactorB = 0;
    uint8_t isBias = 0;
    uint8_t nBufferNum = 0;
    uint8_t dbL0C = 0;
    uint8_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantBatchMatmulV3BasicAPITilingData {
    QuantBatchMatmulV3BasicAPIDataParams params;
    BasicAPICubeTiling matmulTiling;
    DequantBmm::SlidingWindowParams adaptiveSlidingWin;
};
#pragma pack(pop)

class AdaptiveSlidingWindowBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowBasicAPITiling(gert::TilingContext *context);
    AdaptiveSlidingWindowBasicAPITiling(gert::TilingContext *context, QuantBatchMatmulV3BasicAPITilingData *out);
    ~AdaptiveSlidingWindowBasicAPITiling() override = default;

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    void Reset();
    bool AnalyseSlidingWinInfo();
    void IsAFullLoad();
    void SetTilingData();

    QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    QuantBatchMatmulV3BasicAPITilingData &tilingData_;
};
}
#endif  // ADAPTIVE_SLIDING_WINDOW_BASIC_API_TILING_H