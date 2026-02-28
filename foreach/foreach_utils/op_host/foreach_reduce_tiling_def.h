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
 * \file foreach_reduce_tiling_def.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_
#include "op_host/tiling_util.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "foreach_tiling_common.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;
constexpr uint16_t MAX_TENSOR_CONT_950 = 256;
constexpr uint16_t MAX_CORE_CONT_950 = 80;
struct ForeachNormCompileInfo {
    uint32_t coreNum;
};
BEGIN_TILING_DATA_DEF(ForeachReduceTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTensorCount);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 256, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorEndOffsetList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, coreMiddleOffsetList);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ForeachReduceTilingDataRegbase)
TILING_DATA_FIELD_DEF(uint32_t, inputsTensorUbSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTensorCount);
TILING_DATA_FIELD_DEF(uint16_t, maxTensorNumPerCore);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_TENSOR_CONT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT_950, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT_950, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_CONT_950, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_CONT_950, tensorEndOffsetList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_TENSOR_CONT, tensorMiddleCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_TENSOR_CONT, tensorMiddleStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT_950, coreMiddleOffsetList);
END_TILING_DATA_DEF;

#define REDUCE_MEMBASE_TILING(OP_NAME)                                                            \
    class OP_NAME##MembaseTiling : public ForeachBaseClass                                        \
    {                                                                                             \
    public:                                                                                       \
        explicit OP_NAME##MembaseTiling(gert::TilingContext* context) : ForeachBaseClass(context) \
        {}                                                                                        \
        ~OP_NAME##MembaseTiling() override = default;                                             \
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
            ForeachReduceTiling tilingObject(context_);                                           \
            if (tilingObject.Init() != ge::GRAPH_SUCCESS) {                                       \
                return ge::GRAPH_FAILED;                                                          \
            }                                                                                     \
            return tilingObject.RunBigKernelTiling();                                             \
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

REGISTER_TILING_DATA_CLASS(ForeachNorm, ForeachReduceTilingData)
REGISTER_TILING_DATA_CLASS(ForeachNorm_10001, ForeachReduceTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachNorm_10002, ForeachReduceTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachNorm_10004, ForeachReduceTilingDataRegbase)
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_REDUCE_DEF_H_
