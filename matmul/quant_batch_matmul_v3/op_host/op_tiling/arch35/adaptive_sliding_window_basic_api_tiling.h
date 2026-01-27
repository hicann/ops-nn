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
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class AdaptiveSlidingWindowBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowBasicAPITiling(gert::TilingContext *context);
    AdaptiveSlidingWindowBasicAPITiling(gert::TilingContext *context,
                                        DequantBmm::QuantBatchMatmulV3BasicAPITilingData *out);
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
    bool IsCapable() override;
    void Reset() override;
    bool AnalyseSlidingWinInfo() override;
    void IsAFullLoad() override;
    void SetTilingData() override;
    void CalculateNBufferNum4Perblock();
    
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData &tilingData_;
};
}
#endif  // ADAPTIVE_SLIDING_WINDOW_BASIC_API_TILING_H