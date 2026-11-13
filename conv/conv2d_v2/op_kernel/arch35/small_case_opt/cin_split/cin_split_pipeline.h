/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * CinSplitPipeline: entry point for the cin-direction L1-split variant. Wires
 * the spatial scheme module selected by IsHwMode (M-mode / HW-mode) together
 * with the cin-split L1 strategy.
 */
#ifndef CIN_SPLIT_PIPELINE_H
#define CIN_SPLIT_PIPELINE_H

#include "cin_split_processor.h"
#include "../spatial/m_scheme.h"
#include "../spatial/hw_scheme.h"

namespace small_conv {

template <typename FmapT, typename WeightT, typename BiasT, typename OutputT, typename Output1T, bool IsNHWCin,
          bool IsNHWCout, ConvFormat WeightFmt, bool IsHwMode, typename TilingData>
struct CinSplitPipeline {
    using Traits = SmallTraits<FmapT, WeightT, BiasT, OutputT, Output1T>;
    using BlockCtxT = BlockCtx<Traits, IsNHWCin, IsNHWCout, WeightFmt>;
    using Scheme = typename SchemeSelect<IsHwMode, HwScheme<BlockCtxT, TilingData>,
                                         MScheme<BlockCtxT, TilingData>>::Type;

    __aicore__ inline void operator()(const TilingData& tiling, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      const ExtendParams* ep) const
    {
        BlockScheduler<BlockCtxT, TilingData, Scheme> scheduler;
        BlockCtxT blockCtx = scheduler(tiling);
        if (!blockCtx.active || blockCtx.actualCo == 0)
            return;
        CinSplitProcessor<BlockCtxT, TilingData, Scheme> processor;
        processor(blockCtx, tiling, x, filter, bias, y, ep);
    }
};

} // namespace small_conv

#endif // CIN_SPLIT_PIPELINE_H
