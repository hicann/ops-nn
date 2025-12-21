/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_ACT_BLOCK_BLOCK_DECL_H
#define ARCH35_ACT_BLOCK_BLOCK_DECL_H

#include "lib/std/type_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act {
template <
    typename DispatchPolicy, typename TileShapeL1, typename TileShapeL0, typename DtypeA, typename StrideA,
    typename DtypeB, typename StrideB, typename DtypeBias, typename StrideBias, typename DtypeC, typename StrideC>
struct BlockMmad {
    static_assert(!AscendC::Std::is_same_v<DispatchPolicy, DispatchPolicy>, "Unsupported combination");
};

namespace Block {
template <
    typename DispatchPolicy, typename TileShapeL1, typename TileShapeL0, typename AType, typename BType, typename CType,
    typename BiasType, typename TileCopy = void, typename TileMmad = void>
struct BlockMmadAct {
    static_assert(!AscendC::Std::is_same_v<DispatchPolicy, DispatchPolicy>, "Unsupported combination");
};
} // namespace Block
} // namespace WeightQuantBatchMatmulV2::Arch35::Act
#endif