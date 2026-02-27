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
 * \file foreach_tiling_def.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "foreach_tiling_common.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;
constexpr uint16_t MAX_TENSOR_CONT_950 = 50;
constexpr uint16_t MAX_CORE_CONT_950 = 80;

BEGIN_TILING_DATA_DEF(ForeachCommonTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ForeachSoloTilingDataRegbase)
TILING_DATA_FIELD_DEF(uint32_t, inputsTensorUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT_950, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT_950, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT_950, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT_950, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT_950, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachPowScalarAndTensor, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachReciprocal, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachExpm1, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog1p, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSin, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLerpScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLerpScalar_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachLerpScalar_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachLerpScalar_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachDivScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivScalarList_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachDivScalarList_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachDivScalarList_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachZeroInplace, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachTanh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLerpList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachErf, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachErfc, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCosh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSinh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAsin, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAcos, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachTan, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAtan, ForeachCommonTilingData)

REGISTER_TILING_DATA_CLASS(ForeachAbs, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCopy, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSign, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar_10003, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList_10003, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachExp, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSigmoid, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar_10003, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar_10003, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCos, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog2, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog10, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachNeg, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachRoundOffNumber, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSqrt, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSqrt_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSqrt_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSqrt_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSubScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivScalar, ForeachCommonTilingData)

#define MEMBASE_TILING(OP_TYPE, OP_CODE, INPUT_TYPE)                                              \
    class OP_TYPE##MembaseTiling : public ForeachBaseClass                                        \
    {                                                                                             \
    public:                                                                                       \
        explicit OP_TYPE##MembaseTiling(gert::TilingContext* context) : ForeachBaseClass(context) \
        {}                                                                                        \
        ~OP_TYPE##MembaseTiling() override = default;                                             \
        void Reset(gert::TilingContext* context) override                                         \
        {                                                                                         \
            ForeachBaseClass::Reset(context);                                                     \
        }                                                                                         \
                                                                                                  \
    protected:                                                                                    \
        bool IsCapable() override                                                                 \
        {                                                                                         \
            return !Ops::NN::OpTiling::IsRegbaseSocVersion(context_);                             \
        }                                                                                         \
        ge::graphStatus DoOpTiling() override                                                     \
        {                                                                                         \
            ForeachCommonTiling tilingObject(context_);                                           \
            if (tilingObject.Init(OP_CODE, INPUT_TYPE) != ge::GRAPH_SUCCESS) {                    \
                return ge::GRAPH_FAILED;                                                          \
            }                                                                                     \
            return tilingObject.RunBigScalarKernelTiling();                                       \
        }                                                                                         \
        ge::graphStatus PostTiling() override                                                     \
        {                                                                                         \
            return ge::GRAPH_SUCCESS;                                                             \
        }                                                                                         \
        ge::graphStatus GetShapeAttrsInfo() override                                              \
        {                                                                                         \
            return ge::GRAPH_SUCCESS;                                                             \
        }                                                                                         \
        uint64_t GetTilingKey() const override                                                    \
        {                                                                                         \
            return context_->GetTilingKey();                                                      \
        }                                                                                         \
    }

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_
