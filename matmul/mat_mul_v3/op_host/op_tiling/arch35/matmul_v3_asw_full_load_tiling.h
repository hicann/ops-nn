/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_asw_full_load_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__
#define __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__

#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3AswFullLoadTiling : public MatMulV3AswTiling {
public:
    MatMulV3AswFullLoadTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3AswTiling(context, cfg) {};

    ~MatMulV3AswFullLoadTiling() override {};
    bool CheckBL1FullLoadDefault(bool &isKFullLoad, uint64_t kAlignedValue, uint64_t nAlignedValue) const;
    bool CheckBL1FullLoad91095(bool &isKFullLoad, uint64_t kAlignedValue, uint64_t nAlignedValue);
    void AdjustTiling91095Basic(uint64_t biasBatchDimAll);

protected:
    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    void DoBL1FullLoad(bool isKFullLoad, uint64_t aBatchDimAll = 1UL, uint64_t biasBatchDimAll = 1UL);
    void DoAL1FullLoad(bool isKFullLoad, uint64_t bBatchDimAll = 1UL, uint64_t biasBatchDimAll = 1UL);

private:
    void FullLoadPre();
    bool CheckABL1FullLoad() const;
    void DoABL1FullLoad();
    void CalcTailBasicBlockBL1Full();
    bool CheckBL1FullLoad(bool &isKFullLoad);
    bool CheckAL1FullLoad(bool &isKFullLoad) const;
    void AdjustTilingDefault(uint64_t biasBatchDimAll);
    void AdjustTilingCommon(uint64_t aBatchDimAll);
    bool ABL1FullLoadExtraCond(uint64_t al1SingleCoreSize, uint64_t bl1SingleCoreSize) const;
    uint64_t GetStepSmallK(bool isBL1FullLoad) const;

    MatMulV3FullLoad fullLoad_ {MatMulV3FullLoad::NONE_FULL_LOAD};
    MatMulV3L0C2Out l0C2Out_ {MatMulV3L0C2Out::ON_THE_FLY};
    MatMulV3ApiLevel apiLevel_ {MatMulV3ApiLevel::HIGH_LEVEL};
    uint64_t biasSize_ {0};
    bool isSingleRound_ {false};
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_FULL_LOAD_TILING_H__