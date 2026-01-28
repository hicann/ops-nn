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
 * \file matmul_v3_basic_aswt_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__
#define __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__

#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3BasicAswtTiling : public MatMulV3AswTiling {
public:
    MatMulV3BasicAswtTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3AswTiling(context, cfg) {};
    ~MatMulV3BasicAswtTiling() override = default;
    bool CheckBL1FullLoad91095(uint64_t kAlignedValue, uint64_t nAlignedValue);
    bool CheckAL1FullLoad91095(uint64_t kAlignedValue, uint64_t mAlignedValue);
    void AdjustBL1Tiling91095Basic(uint64_t biasBatchDimAll);
    void AdjustAL1Tiling91095Basic(uint64_t biasBatchDimAll);

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetTilingData(TilingResult& tiling) const override;
    void DoBL1FullLoad(uint64_t aBatchDimAll = 1UL, uint64_t biasBatchDimAll = 1UL);
    void DoAL1FullLoad(uint64_t bBatchDimAll = 1UL, uint64_t biasBatchDimAll = 1UL);

private:
    void FullLoadPre();
    uint64_t GetAFullLoadBasicNL1() const;
    void CalcTailBasicBlockBL1Full();
    void CalcTailBasicBlockAL1Full();
    bool CheckBL1FullLoad();
    bool CheckAL1FullLoad();

    MatMulV3FullLoad fullLoad_ {MatMulV3FullLoad::NONE_FULL_LOAD};
    MatMulV3L0C2Out l0C2Out_ {MatMulV3L0C2Out::ON_THE_FLY};
    uint64_t biasSize_ {0};
    bool isSingleRound_ {false};
    bool isAFullLoad_ {false};
    bool isBFullLoad_ {false};
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__