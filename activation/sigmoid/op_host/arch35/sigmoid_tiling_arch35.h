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
 * \file sigmoid_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SIGMOID_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SIGMOID_H_

#include "register/op_def_registry.h"
#include "tiling_base/tiling_base.h"
#include "op_common/atvoss/elewise/elewise_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SigmoidTilingData)
    TILING_DATA_FIELD_DEF(int64_t, dim0);
    TILING_DATA_FIELD_DEF(int64_t, blockFormer);
    TILING_DATA_FIELD_DEF(int64_t, ubFormer);
    TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
    TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
    TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
    TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
    TILING_DATA_FIELD_DEF(int64_t, elemNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sigmoid, SigmoidTilingData);

struct SigmoidCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SigmoidTiling : public Ops::NN::Optiling::TilingBaseClass {
   public:
    explicit SigmoidTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {}

   protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    std::string ToString(SigmoidTilingData &tilingData_);

   private:
    uint64_t GetOpKey(ge::DataType xDtype, ge::DataType yDtype);
    uint64_t GenerateTilingKey(uint64_t innerKey);
    std::map<uint64_t, Ops::Base::ComputeParams> GetComputeMap(uint64_t opKey_);

    SigmoidTilingData tilingData;
    uint64_t opKey;
    int64_t coreNum;
    int64_t ubSize;
    uint64_t blockNum;
};

}  // namespace optiling

#endif //AIR_CXX_RUNTIME_V2_OP_IMPL_SIGMOID_H_
