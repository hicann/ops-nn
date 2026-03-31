/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file reverse_v2_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_V2_H_
#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling
{
constexpr int64_t MAX_DIM_SIZE = 8;

BEGIN_TILING_DATA_DEF(ReverseV2TilingData4AscendC)
TILING_DATA_FIELD_DEF(int64_t, param0);
TILING_DATA_FIELD_DEF(int64_t, param1);
TILING_DATA_FIELD_DEF(int64_t, param2);
TILING_DATA_FIELD_DEF(int64_t, param3);
TILING_DATA_FIELD_DEF(int64_t, param4);
TILING_DATA_FIELD_DEF(int64_t, param5);
TILING_DATA_FIELD_DEF(int64_t, param6);
TILING_DATA_FIELD_DEF(int64_t, dim0);
TILING_DATA_FIELD_DEF(int64_t, dim1);
TILING_DATA_FIELD_DEF(int64_t, dim2);
TILING_DATA_FIELD_DEF(int64_t, dim3);
TILING_DATA_FIELD_DEF(int64_t, dim4);
TILING_DATA_FIELD_DEF(int64_t, dim5);
TILING_DATA_FIELD_DEF(int64_t, dim6);
TILING_DATA_FIELD_DEF(int64_t, dim7);
TILING_DATA_FIELD_DEF(int64_t, inputSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, dtypeSize);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, splitDim);
TILING_DATA_FIELD_DEF(int64_t, splitDimInNum);
TILING_DATA_FIELD_DEF(int64_t, splitDimTailInNum);
TILING_DATA_FIELD_DEF(int64_t, splitDimLoop);
TILING_DATA_FIELD_DEF(int64_t, dimNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, 8, loopStride);
TILING_DATA_FIELD_DEF(int64_t, dim0Reversed);
TILING_DATA_FIELD_DEF(int64_t, splitDimReversed);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TensorMoveTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockTailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReverseV2, ReverseV2TilingData4AscendC)
REGISTER_TILING_DATA_CLASS(ReverseV2_1000, TensorMoveTilingData)
REGISTER_TILING_DATA_CLASS(ReverseV2_2000, TensorMoveTilingData)
REGISTER_TILING_DATA_CLASS(ReverseV2_4000, TensorMoveTilingData)
REGISTER_TILING_DATA_CLASS(ReverseV2_8000, TensorMoveTilingData)

struct ReverseV2TilingParam {
    int64_t totalCoreNum;
    int64_t ubSize;
    int64_t uo;
    int64_t usedCoreNum;
    int64_t bytesForOneData;
    int64_t ubFactor;
    int64_t tailBlockTailUbFactor;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t tilingKey;
};

struct ReverseV2CompileInfo {
  int64_t core_num;
  int64_t max_elements;
  int64_t max_elements_last_large_size;
  int64_t dtype_rate;
  int64_t topk_threshold;
  int64_t is_vconcat;
  int64_t ubSize{0};
  int64_t totalCoreNum{0};
};

ge::graphStatus ReverseV2TilingForAscendC(gert::TilingContext* context);

class ReverseV2Tiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit ReverseV2Tiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {}
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsTensorMove();

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus GetInputShape();
    template <typename T>
    ge::graphStatus GetReversedDims();
    std::vector<int64_t> GetNonReversedDims(const std::vector<int64_t>& shapeVec, const std::vector<int64_t>& dims,
                                            bool isReversedDim);
    void DoSimdTiling();
    void SingleUbTiling();
    void SplitProcess(int64_t inNegOneSize);
    int64_t CalcLoopStride(int64_t index);

private:
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_;
    int64_t param0_;
    int64_t param1_;
    int64_t param2_;
    int64_t param3_;
    int64_t param4_;
    int64_t param5_;
    int64_t param6_;
    int64_t dim0_;
    int64_t dim1_;
    int64_t dim2_;
    int64_t dim3_;
    int64_t dim4_;
    int64_t dim5_;
    int64_t dim6_;
    int64_t dim7_;
    int64_t inputSize_ = 1;
    uint64_t tilingKey_;
    bool isSimd_ = false;
    int64_t dtypeSize_ = 0;
    int64_t availableUb_ = 0;
    int64_t totalLoop_ = 1;
    int64_t splitDim_ = 0;
    int64_t splitDimInNum_ = 0;
    int64_t splitDimTailInNum_ = 0;
    int64_t splitDimLoop_ = 0;
    int64_t inUbSize_ = 0;
    int64_t blockTail_ = 0;
    int64_t dimNum_ = 1;
    int64_t dim0Reversed_ = 0;
    int64_t splitDimReversed_ = 0;
    int64_t loopStride_[MAX_DIM_SIZE] = {1};
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> reversedDims_;
    bool flag_ = false;
    ReverseV2TilingData4AscendC tilingData_;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_V2_H_
