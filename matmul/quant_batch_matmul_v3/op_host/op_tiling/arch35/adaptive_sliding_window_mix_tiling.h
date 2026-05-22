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
 * \file adaptive_sliding_window_mix_tiling.h
 * \brief
 */
#pragma once

#include "adaptive_sliding_window_tiling.h"

namespace optiling {

class AdaptiveSlidingWindowMixTiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowMixTiling(gert::TilingContext* context);
    AdaptiveSlidingWindowMixTiling(gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out);
    ~AdaptiveSlidingWindowMixTiling() override = default;

protected:
    bool IsCapable() override;
    bool CheckCoreNum() const override;
    ge::graphStatus GetWorkspaceSize() override;
    bool CalcBasicBlock() override;
    void AnalyseFullLoadInfo() override;
    void CalcTailRoundBasicBlockSplit() override;
    bool CalL1Tiling() override;

private:
    void UpdateAFullLoadStatus();
    bool isSupportS4S4_ = false;
};

} // namespace optiling
