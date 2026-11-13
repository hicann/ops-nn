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
 * FmPartloadPipeline: entry point for the FM-partload (weight one-shot)
 * variant. Wires the spatial scheme module selected by IsHwMode (M-mode /
 * HW-mode) together with the fm-partload L1 strategy.
 */
#ifndef FM_PARTLOAD_PIPELINE_H
#define FM_PARTLOAD_PIPELINE_H

#include "fm_partload_processor.h"

namespace small_conv {

template <typename FmapT, typename WeightT, typename BiasT, typename OutputT, typename Output1T, bool IsNHWCin,
          bool IsNHWCout, ConvFormat WeightFmt, bool IsHwMode, typename TilingData>
struct FmPartloadPipeline {
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
        FmPartloadProcessor<BlockCtxT, TilingData, Scheme> processor;
        processor(blockCtx, tiling, x, filter, bias, y, ep);
    }
};

} // namespace small_conv

#endif // FM_PARTLOAD_PIPELINE_H
