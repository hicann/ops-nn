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
 * \file adaptive_sliding_window_cube_tiling.h
 * \brief
 */
#pragma once

#include "adaptive_sliding_window_tiling.h"

namespace optiling {

class AdaptiveSlidingWindowCubeTiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context);
    AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out);
    ~AdaptiveSlidingWindowCubeTiling() override = default;

protected:
    bool IsCapable() override;
    bool CalL1Tiling() override;
    void AnalyseFullLoadInfo() override;
    virtual void UpdateAFullLoadStatus();
    virtual void UpdateBFullLoadStatus();
    virtual void UpdateABFullLoadStatus();
};

} // namespace optiling
