/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <limits>

#include "securec.h"

#include "matmul_add_tiling.h"
#include "op_common/log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"

namespace optiling {

constexpr const char* OP_TYPE = "MatmulAdd";
constexpr uint64_t TILING_KEY_FP16 = 10;
constexpr uint64_t TILING_KEY_BF16 = 30;
constexpr uint32_t REQUIRED_DIM_NUM = 2;
constexpr uint32_t INPUT_A_IDX = 0;
constexpr uint32_t INPUT_B_IDX = 1;
constexpr uint32_t INPUT_BIAS_IDX = 2;

struct DtypeInfo {
    uint32_t dtype_size;
    uint64_t tiling_key;
    matmul_tiling::DataType mm_dtype;
};

static bool GetDtypeInfo(ge::DataType dtype, DtypeInfo& info)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            info = {sizeof(uint16_t), TILING_KEY_FP16,
                    matmul_tiling::DataType::DT_FLOAT16};
            return true;
        case ge::DT_BF16:
            info = {sizeof(uint16_t), TILING_KEY_BF16,
                    matmul_tiling::DataType::DT_BFLOAT16};
            return true;
        default:
            return false;
    }
}

struct MatmulShape {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

static bool GetDimAsUint32(int64_t dim, const char* dimName,
    const char* nodeName, uint32_t& value)
{
    if (dim < 0 ||
        dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s dimension %s is out of uint32 range: %lld.",
            nodeName, dimName, static_cast<long long>(dim));
        return false;
    }
    value = static_cast<uint32_t>(dim);
    return true;
}

static ge::graphStatus ParseShapeAndAttrs(
    gert::TilingContext* context, MatmulShape& shape)
{
    auto a_shape = context->GetInputShape(INPUT_A_IDX);
    auto b_shape = context->GetInputShape(INPUT_B_IDX);
    if (a_shape == nullptr || b_shape == nullptr) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s input shape is null.",
            context->GetNodeName());
        return ge::GRAPH_FAILED;
    }

    auto a_dims = a_shape->GetOriginShape().GetDimNum();
    auto b_dims = b_shape->GetOriginShape().GetDimNum();
    if (a_dims != REQUIRED_DIM_NUM || b_dims != REQUIRED_DIM_NUM) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s inputs must be 2-D, got a=%zu b=%zu.",
            context->GetNodeName(), a_dims, b_dims);
        return ge::GRAPH_FAILED;
    }

    auto a_r_dim = a_shape->GetOriginShape().GetDim(0);
    auto a_c_dim = a_shape->GetOriginShape().GetDim(1);
    auto b_r_dim = b_shape->GetOriginShape().GetDim(0);
    auto b_c_dim = b_shape->GetOriginShape().GetDim(1);

    uint32_t a_r = 0;
    uint32_t a_c = 0;
    uint32_t b_r = 0;
    uint32_t b_c = 0;
    if (!GetDimAsUint32(a_r_dim, "a[0]", context->GetNodeName(), a_r) ||
        !GetDimAsUint32(a_c_dim, "a[1]", context->GetNodeName(), a_c) ||
        !GetDimAsUint32(b_r_dim, "b[0]", context->GetNodeName(), b_r) ||
        !GetDimAsUint32(b_c_dim, "b[1]", context->GetNodeName(), b_c)) {
        return ge::GRAPH_FAILED;
    }

    shape.M = a_r;
    shape.N = b_c;

    if (a_c != b_r) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s K mismatch: %u vs %u.",
            context->GetNodeName(), a_c, b_r);
        return ge::GRAPH_FAILED;
    }
    shape.K = a_c;
    return ge::GRAPH_SUCCESS;
}

static bool HasBias(gert::TilingContext* context)
{
    auto bias_shape = context->GetInputShape(INPUT_BIAS_IDX);
    return bias_shape != nullptr &&
           bias_shape->GetOriginShape().GetDimNum() > 0;
}

static ge::graphStatus CheckBiasShape(
    gert::TilingContext* context, const MatmulShape& shape)
{
    if (!HasBias(context)) {
        return ge::GRAPH_SUCCESS;
    }

    auto bias_shape = context->GetInputShape(INPUT_BIAS_IDX);
    auto bias_dims = bias_shape->GetOriginShape().GetDimNum();
    if (bias_dims != 1) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s bias must be 1-D, got %zu.",
            context->GetNodeName(), bias_dims);
        return ge::GRAPH_FAILED;
    }

    auto bias_n_dim = bias_shape->GetOriginShape().GetDim(0);
    uint32_t bias_n = 0;
    if (!GetDimAsUint32(
        bias_n_dim, "bias[0]", context->GetNodeName(), bias_n)) {
        return ge::GRAPH_FAILED;
    }
    if (bias_n != shape.N) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s bias length must equal N, got %u vs %u.",
            context->GetNodeName(), bias_n, shape.N);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupCubeTiling(
    const platform_ascendc::PlatformAscendC& platform,
    const MatmulShape& shape, const DtypeInfo& dtype_info,
    bool has_bias, MatmulAddTilingData& tiling, const char* nodeName)
{
    matmul_tiling::MatmulApiTiling mmTiling(platform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, dtype_info.mm_dtype, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, dtype_info.mm_dtype, false);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, dtype_info.mm_dtype);
    mmTiling.SetBiasType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, dtype_info.mm_dtype);
    mmTiling.SetShape(shape.M, shape.N, shape.K);
    mmTiling.SetOrgShape(shape.M, shape.N, shape.K);
    mmTiling.SetBias(has_bias);

    tiling.M = shape.M;
    tiling.N = shape.N;
    tiling.K = shape.K;
    tiling.has_bias = has_bias ? 1U : 0U;
    if (mmTiling.GetTiling(tiling.cubeTiling) != 0) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s GetTiling failed.", nodeName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputDtypeInfo(
    gert::TilingContext* context, DtypeInfo& dtype_info)
{
    auto input_desc = context->GetInputDesc(INPUT_A_IDX);
    if (input_desc == nullptr) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s input desc null.", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }
    if (!GetDtypeInfo(input_desc->GetDataType(), dtype_info)) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s unsupported dtype %d.", context->GetNodeName(),
            static_cast<int32_t>(input_desc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus WriteRawTilingData(
    gert::TilingContext* context, const MatmulAddTilingData& tiling)
{
    auto raw = context->GetRawTilingData();
    if (raw == nullptr) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s raw tiling null.", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }
    if (raw->GetCapacity() < sizeof(tiling)) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s raw tiling capacity %zu is less than %zu.",
            context->GetNodeName(), raw->GetCapacity(), sizeof(tiling));
        return ge::GRAPH_FAILED;
    }
    auto copy_ret = memcpy_s(raw->GetData(), raw->GetCapacity(),
        &tiling, sizeof(tiling));
    if (copy_ret != EOK) {
        OP_LOGE(OP_TYPE,
            "MatmulAdd node %s copy tiling data failed, ret=%d.",
            context->GetNodeName(), static_cast<int32_t>(copy_ret));
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(tiling));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAddTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    MatmulShape shape{};
    ge::graphStatus ret = ParseShapeAndAttrs(context, shape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckBiasShape(context, shape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DtypeInfo dtype_info{};
    ret = GetInputDtypeInfo(context, dtype_info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MatmulAddTilingData tiling;
    ret = SetupCubeTiling(platform, shape, dtype_info,
        HasBias(context), tiling, context->GetNodeName());
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = WriteRawTilingData(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetTilingKey(dtype_info.tiling_key);
    context->SetBlockDim(tiling.cubeTiling.get_usedCoreNum());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAdd).Tiling(MatmulAddTiling);

} // namespace optiling
