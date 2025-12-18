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
 * \file block_mmad.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_H
#define MATMUL_BLOCK_BLOCK_MMAD_H

#include <type_traits>
#include "../../utils/arch.h"
#include "../../utils/integral_constant.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
* @class BlockMmad
* @brief Block matrix multiplication class for performing block matrix multiplication operations
*
* @param [in] DispatchPolicy: the dispatch policy type, which determines how computation tasks are allocated
* @param [in] L1TileShape: the shape of L1 tile
* @param [in] L0TileShape: the shape of L0 tile
* @param [in] AType: the data type of matrix A
* @param [in] BType: the data type of matrix B
* @param [in] CType: the data type of matrix C
* @param [in] BiasType: the data type of the bias term, defaulting to the same type of CType
* @param [in] TileCopy: the tile copy strategy type
* @param <unnamed> Support specialization via the DispatchPolicy type
*/
template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class BiasType = CType, class TileCopy = void,
          typename = void
          >
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
