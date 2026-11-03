/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <utility>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_v2.h"
#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_v3.h"
#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_nz.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
enum TensorLayout { LAYOUT_CONTIGUOUS, LAYOUT_TRANSPOSED, LAYOUT_NOT_CONTIGUOUS };

// 标准row-major strides: shape(m, n) -> (n, 1)
static vector<int64_t> RowMajorStrides(const vector<int64_t>& shape)
{
    vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static op::Shape ToOpShape(const vector<int64_t>& dims)
{
    op::Shape shape;
    for (auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

// 创建测试tensor: layout控制连续性(仅2维支持转置), NZ场景需显式传入storageShape(控制C0)
static aclTensor* MakeTensor(const vector<int64_t>& viewShape, aclDataType dtype, aclFormat format, TensorLayout layout,
                             const vector<int64_t>& storageShape = {})
{
    vector<int64_t> strides = RowMajorStrides(viewShape);
    vector<int64_t> originalShape = viewShape;
    if (layout == LAYOUT_TRANSPOSED) {
        // 2维转置view: strides为(1, dim0), originalShape为(dim1, dim0)
        strides = {1, viewShape[0]};
        originalShape = {viewShape[1], viewShape[0]};
    } else if (layout == LAYOUT_NOT_CONTIGUOUS) {
        strides[strides.size() - 1] = 2; // 2: 最后1维步长不为1构造非连续
    }
    vector<int64_t> storage = storageShape;
    if (storage.empty()) {
        storage = originalShape;
        if (layout == LAYOUT_NOT_CONTIGUOUS) {
            storage[storage.size() - 1] *= 2;
        }
    }
    auto tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, strides.data(), 0, format, storage.data(),
                                  storage.size(), nullptr);
    tensor->SetOriginalShape(ToOpShape(originalShape));
    return tensor;
}

// 使用显式strides创建tensor(用于构造特殊连续性语义, 如(1,1)广播步长)
static aclTensor* MakeTensorWithStrides(const vector<int64_t>& viewShape, aclDataType dtype, aclFormat format,
                                        const vector<int64_t>& strides)
{
    auto tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, strides.data(), 0, format,
                                  viewShape.data(), viewShape.size(), nullptr);
    tensor->SetOriginalShape(ToOpShape(viewShape));
    return tensor;
}

static aclnnStatus RunV2(const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                         const aclTensor* antiquantOffset, const aclTensor* quantScale, const aclTensor* quantOffset,
                         const aclTensor* bias, int antiquantGroupSize, const aclTensor* y)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, quantScale,
                                                         quantOffset, bias, antiquantGroupSize, y, &workspaceSize,
                                                         &executor);
}

static aclnnStatus RunV3(const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                         const aclTensor* antiquantOffset, const aclTensor* quantScale, const aclTensor* quantOffset,
                         const aclTensor* bias, int antiquantGroupSize, int innerPrecise, const aclTensor* y)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, quantScale,
                                                         quantOffset, bias, antiquantGroupSize, innerPrecise, y,
                                                         &workspaceSize, &executor);
}

static aclnnStatus RunNz(const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                         const aclTensor* antiquantOffset, const aclTensor* quantScale, const aclTensor* quantOffset,
                         const aclTensor* bias, int antiquantGroupSize, const aclTensor* y)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, quantScale,
                                                         quantOffset, bias, antiquantGroupSize, y, &workspaceSize,
                                                         &executor);
}
} // namespace

// ============================ DAV_9201(SOC96)场景 ============================
class l2_weight_quant_batch_matmul_v2_cov_9201 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v2_cov_9201 SetUp" << endl;
        versionManager = std::make_unique<op::SocVersionManager>(op::SocVersion::RESERVED_VERSION);
        archManager = std::make_unique<op::NpuArchManager>(NpuArch::DAV_9201);
    }
    static void TearDownTestCase()
    {
        archManager.reset();
        versionManager.reset();
        cout << "l2_weight_quant_batch_matmul_v2_cov_9201 TearDown" << endl;
    }
    static std::unique_ptr<op::SocVersionManager> versionManager;
    static std::unique_ptr<op::NpuArchManager> archManager;
};
std::unique_ptr<op::SocVersionManager> l2_weight_quant_batch_matmul_v2_cov_9201::versionManager;
std::unique_ptr<op::NpuArchManager> l2_weight_quant_batch_matmul_v2_cov_9201::archManager;

// INT8 ND per-channel: scale dtype与x一致
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_perchannel_bf16_success)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT8 ND非转置: E8M0 scale与x dtype不一致
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_e8m0_scale_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 ND转置: E8M0 scale既不等于x dtype也不是UINT64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_transposed_e8m0_scale_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({32, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 ND转置 + UINT64 scale(fixpipe)
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_transposed_uint64_scale_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({32, 1}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT8 ND转置 + INT64 scale: TensorPreProcess将其修正为UINT64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_transposed_int64_scale_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({32, 1}, ACL_INT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// UINT64 scale + HIFLOAT8 weight: fixpipe9201不支持该weight dtype
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, uint64_scale_hifloat8_weight_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_HIFLOAT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({32, 1}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// FP8 ND非转置: scale必须为UINT64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp8_nd_scale_not_uint64_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// FP8 ND: K不满足32对齐
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp8_nd_k_not_div32_invalid)
{
    auto* x = MakeTensor({16, 66}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({66, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// FP8 ND: N不满足32对齐
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp8_nd_n_not_div32_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 34}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 34}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 34}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// FP8 ND + UINT64 scale per-channel
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp8_nd_uint64_scale_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT8 NZ: K不满足32对齐
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_k_not_div32_invalid)
{
    auto* x = MakeTensor({16, 66}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({66, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 5, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C032模板: 不支持antiquantOffset
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_offset_not_null_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 4, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* offset = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, offset, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C032模板: transposeWeight必须为true
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_not_transposed_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 4, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C032模板per-channel: scale必须为UINT64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_perchannel_scale_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 4, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C032模板per-channel + UINT64 scale
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_perchannel_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 4, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT8 NZ C032模板per-group: groupSize只支持64/128
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_pergroup_gs_invalid)
{
    auto* x = MakeTensor({16, 512}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({512, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 32, 16, 32});
    auto* scale = MakeTensor({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 256, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C032模板per-group: K必须被groupSize整除
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c032_pergroup_k_not_div_gs_invalid)
{
    auto* x = MakeTensor({16, 96}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({96, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 6, 16, 32});
    auto* scale = MakeTensor({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C016模板: transposeWeight必须为false
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c016_transposed_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {2, 4, 16, 16});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C016模板per-group: groupSize只支持32/64/128/256
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c016_gs_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {2, 8, 16, 16});
    auto* scale = MakeTensor({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 96, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ C016模板per-group
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c016_pergroup_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {2, 8, 16, 16});
    auto* scale = MakeTensor({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// INT8 NZ: C0既不是16也不是32
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_c0_value_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {2, 4, 16, 8});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C064模板per-group: 不支持antiquantOffset
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c064_offset_not_null_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* offset = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, offset, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C064模板(C+V)per-group, 走Nz接口
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c064_pergroup_nzapi_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_SUCCESS);
}

// INT4 NZ C064模板: groupSize只支持64/128
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c064_gs_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C064模板: transposeWeight必须为true
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c064_not_transposed_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C016模板(V+C)per-group: groupSize只支持32/64/128/256
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c016_gs_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {4, 8, 16, 16});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 96, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C016模板: transposeWeight必须为false
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c016_transposed_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {4, 8, 16, 16});
    auto* scale = MakeTensor({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ C016模板(V+C)per-group
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_c016_pergroup_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {4, 8, 16, 16});
    auto* scale = MakeTensor({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// INT4 NZ per-channel非转置weight
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_perchannel_not_transposed_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT4 NZ per-channel转置weight: scale必须为UINT64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_perchannel_scale_dtype_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({64, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ per-channel转置weight + UINT64 scale
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_perchannel_transposed_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({64, 1}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT4 NZ per-channel转置weight: K不满足64对齐
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_perchannel_k_not_div64_invalid)
{
    auto* x = MakeTensor({16, 96}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({96, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 6, 16, 64});
    auto* scale = MakeTensor({64, 1}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 NZ per-tensor scale: 9201上仅INT4非转置weight支持per-tensor
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nz_pertensor_scale_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 4, 16, 32});
    auto* scale = MakeTensor({1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ非转置weight + per-tensor scale
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_pertensor_scale_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// INT4 NZ per-channel: 2维scale形状错误(9201专属报错分支)
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int4_nz_perchannel_2dim_scale_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4+E8M0 micro-scaling NZ weight
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp4_mx_nz_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// FP4 micro-scaling: groupSize必须为32
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp4_mx_groupsize_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 ND micro-scaling: MX scale形状错误(应为(ceil(k/64), n, 2))
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_nd_scale_shape_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({32, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 ND micro-scaling: MX scale格式仅支持ND/NCL
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_scale_format_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_NCHW, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 ND micro-scaling: MX scale支持NCL格式
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_format_ncl_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_NCL, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// FP4 ND micro-scaling: E8M0 scale必须为3维
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_scale_dim_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 ND micro-scaling
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_fp4_nd_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// FP4 ND micro-scaling + BF16 x: bias必须为BF16
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_fp4_nd_bias_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, bias, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 ND micro-scaling + BF16 x + BF16 bias
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, mx_fp4_nd_bias_bf16_success)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, bias, 32, y), ACLNN_SUCCESS);
}

// INT8 ND + BF16 x: bias仅支持BF16/FLOAT
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, int8_nd_bias_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, bias, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// 9201不支持transposeX
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, transpose_x_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// HIFLOAT4 NZ: groupSize必须为64
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, hif4_nz_groupsize_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_HIFLOAT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// HIFLOAT4 NZ: transposeWeight必须为true
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, hif4_nz_not_transposed_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_HIFLOAT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// HIFLOAT4 NZ: scale必须为HIFLOAT4_SCALE
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, hif4_nz_scale_dtype_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_HIFLOAT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// HIFLOAT4 NZ: K不满足64对齐
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, hif4_nz_k_not_div64_invalid)
{
    auto* x = MakeTensor({16, 96}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({96, 64}, ACL_HIFLOAT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 6, 16, 64});
    auto* scale = MakeTensor({2, 64, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// HIFLOAT4 NZ + HIFLOAT4_SCALE scale
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, hif4_nz_success)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_HIFLOAT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 64, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_SUCCESS);
}

// FP4 NZ C016模板per-group: K必须为偶数
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp4_nz_c016_k_odd_invalid)
{
    auto* x = MakeTensor({16, 129}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({129, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {4, 9, 16, 16});
    auto* scale = MakeTensor({5, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 NZ C064模板per-group: N必须被64整除
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp4_nz_c064_n_not_div64_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 33}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {1, 8, 16, 64});
    auto* scale = MakeTensor({2, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// FP4 NZ C016模板per-group: N必须为偶数
TEST_F(l2_weight_quant_batch_matmul_v2_cov_9201, fp4_nz_c016_n_odd_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 33}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {3, 8, 16, 16});
    auto* scale = MakeTensor({4, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// ============================ Ascend910B场景 ============================
class l2_weight_quant_batch_matmul_v2_cov_910b : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v2_cov_910b SetUp" << endl;
        versionManager = std::make_unique<op::SocVersionManager>(op::SocVersion::ASCEND910B);
    }
    static void TearDownTestCase()
    {
        versionManager.reset();
        cout << "l2_weight_quant_batch_matmul_v2_cov_910b TearDown" << endl;
    }
    static std::unique_ptr<op::SocVersionManager> versionManager;
};
std::unique_ptr<op::SocVersionManager> l2_weight_quant_batch_matmul_v2_cov_910b::versionManager;

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, null_x_invalid)
{
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(nullptr, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, null_weight_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, nullptr, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, null_antiquant_scale_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, nullptr, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, null_y_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

// antiquantGroupSize为负数
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, group_size_negative_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, -1, y), ACLNN_ERR_PARAM_INVALID);
}

// antiquantGroupSize必须小于K
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, group_size_gt_k_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// 转置x时M不能超过65535
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, transposed_x_m_exceed_invalid)
{
    auto* x = MakeTensor({65536, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({65536, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// antiquantScale格式必须为ND(不经过ReFormat修正, 判定确定)
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, antiquant_scale_format_nchw_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_NCHW, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// bias格式必须为ND(FRACTAL_NZ不经过ReFormat修正, 判定确定)
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, bias_format_fractal_nz_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 2, 16, 32});
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, bias, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// antiquantOffset格式必须为ND
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, antiquant_offset_format_nchw_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* offset = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_NCHW, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, offset, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// quantScale格式必须为ND
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, quant_scale_format_nchw_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* quantScale = MakeTensor({1, 32}, ACL_UINT64, ACL_FORMAT_NCHW, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, quantScale, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// y dtype不在支持列表
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, y_dtype_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_INT32, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// quantScale非空时y必须为INT8
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, quant_scale_y_not_int8_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* quantScale = MakeTensor({1, 32}, ACL_UINT64, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, quantScale, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// antiquantScale仅支持1-2维
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, antiquant_scale_dim_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// bias仅支持1-2维
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, bias_dim_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({1, 1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, bias, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// x必须为2维
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, x_dim_invalid)
{
    auto* x = MakeTensor({64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// V3接口innerPrecise=1合法
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, v3_innerprecise_one_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV3(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, 1, y), ACLNN_SUCCESS);
}

// INT4 ND非转置: N必须为偶数
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int4_nd_n_odd_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 33}, ACL_INT4, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 33}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 ND转置: K必须为偶数
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int4_nd_transposed_k_odd_invalid)
{
    auto* x = MakeTensor({16, 63}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({63, 32}, ACL_INT4, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ per-group: x不允许转置
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int4_nz_pergroup_transposed_x_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {2, 8, 16, 32});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ per-group: weight不允许转置
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int4_nz_pergroup_transposed_weight_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_TRANSPOSED, {2, 8, 16, 32});
    auto* scale = MakeTensor({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 64, y), ACLNN_ERR_PARAM_INVALID);
}

// INT4 NZ per-group: groupSize只支持64/128
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int4_nz_pergroup_gs_invalid)
{
    auto* x = MakeTensor({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({128, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {2, 8, 16, 32});
    auto* scale = MakeTensor({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_ERR_PARAM_INVALID);
}

// INT8 ND转置weight + 1维scale
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, int8_nd_transposed_1dim_scale_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensor({32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// 二段接口空executor
TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, v2_execute_null_executor_invalid)
{
    EXPECT_EQ(aclnnWeightQuantBatchMatmulV2(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, v3_execute_null_executor_invalid)
{
    EXPECT_EQ(aclnnWeightQuantBatchMatmulV3(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_cov_910b, nz_execute_null_executor_invalid)
{
    EXPECT_EQ(aclnnWeightQuantBatchMatmulNz(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

// ============================ Ascend310P场景 ============================
class l2_weight_quant_batch_matmul_v2_cov_310p : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v2_cov_310p SetUp" << endl;
        versionManager = std::make_unique<op::SocVersionManager>(op::SocVersion::ASCEND310P);
    }
    static void TearDownTestCase()
    {
        versionManager.reset();
        cout << "l2_weight_quant_batch_matmul_v2_cov_310p TearDown" << endl;
    }
    static std::unique_ptr<op::SocVersionManager> versionManager;
};
std::unique_ptr<op::SocVersionManager> l2_weight_quant_batch_matmul_v2_cov_310p::versionManager;

// 310P batch维不可广播
TEST_F(l2_weight_quant_batch_matmul_v2_cov_310p, broadcast_batch_mismatch_invalid)
{
    auto* x = MakeTensor({2, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({3, 64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// 310P y的batch维与推导结果不一致
TEST_F(l2_weight_quant_batch_matmul_v2_cov_310p, broadcast_out_dim_mismatch_invalid)
{
    auto* x = MakeTensor({2, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({2, 64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({3, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// 310P 2维weight与3维x: y的batch维与推导结果不一致
TEST_F(l2_weight_quant_batch_matmul_v2_cov_310p, broadcast_weight_2dim_out_mismatch_invalid)
{
    auto* x = MakeTensor({2, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// Nz接口不支持310P
TEST_F(l2_weight_quant_batch_matmul_v2_cov_310p, nz_api_soc_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 4, 16, 32});
    auto* scale = MakeTensor({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_RUNTIME_ERROR);
}

// 310P ND非group场景1维scale跳过转置处理
TEST_F(l2_weight_quant_batch_matmul_v2_cov_310p, nd_1dim_scale_case)
{
    auto* x = MakeTensor({96, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({96, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto ret = RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_PARAM_INVALID);
}

// ============================ Ascend950(DAV_3510)场景 ============================
class l2_weight_quant_batch_matmul_v2_cov_950 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v2_cov_950 SetUp" << endl;
        versionManager = std::make_unique<op::SocVersionManager>(op::SocVersion::ASCEND950);
        archManager = std::make_unique<op::NpuArchManager>(NpuArch::DAV_3510);
    }
    static void TearDownTestCase()
    {
        archManager.reset();
        versionManager.reset();
        cout << "l2_weight_quant_batch_matmul_v2_cov_950 TearDown" << endl;
    }
    static std::unique_ptr<op::SocVersionManager> versionManager;
    static std::unique_ptr<op::NpuArchManager> archManager;
};
std::unique_ptr<op::SocVersionManager> l2_weight_quant_batch_matmul_v2_cov_950::versionManager;
std::unique_ptr<op::NpuArchManager> l2_weight_quant_batch_matmul_v2_cov_950::archManager;

// FP8 weight n=1时per-tensor等价于per-channel, 允许
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, fp8_n1_pertensor_scale_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 1}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// DAV3510不支持转置x
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, x_transposed_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// DAV3510非转置weight: antiquantScale必须连续
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, antiquant_scale_not_contiguous_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_NOT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// DAV3510非转置weight: antiquantOffset必须连续
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, antiquant_offset_not_contiguous_invalid)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* offset = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_NOT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, offset, nullptr, nullptr, nullptr, 0, y), ACLNN_ERR_PARAM_INVALID);
}

// DAV3510转置weight + 转置语义scale(1,1)步长
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, transposed_weight_scale_consistent_success)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_ND, LAYOUT_TRANSPOSED);
    auto* scale = MakeTensorWithStrides({1, 32}, ACL_BF16, ACL_FORMAT_ND, {1, 1});
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// DAV3510 INT8 NZ per-channel + FLOAT bias, 走Nz接口
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, int8_nz_bias_float_nzapi_success)
{
    auto* x = MakeTensor({16, 64}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, LAYOUT_CONTIGUOUS, {1, 4, 16, 32});
    auto* scale = MakeTensor({1, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* bias = MakeTensor({32}, ACL_FLOAT, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_BF16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, bias, 0, y), ACLNN_SUCCESS);
}

// DAV3510 HIFLOAT8 ND per-channel
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, hifloat8_nd_perchannel_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_HIFLOAT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunV2(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// DAV3510 FP4 4bit紧凑NZ_C0_16格式preprocess, 走Nz接口
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, fp4_nz_c0_16_mx_nzapi_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_16, LAYOUT_CONTIGUOUS,
                              {2, 4, 16, 16});
    auto* scale = MakeTensor({2, 32}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}

// DAV3510 INT32承载INT4的legacy NZ_C0_2打包路径, 走Nz接口
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, int32_nz_c0_2_perchannel_nzapi_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 4}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ_C0_2, LAYOUT_CONTIGUOUS, {1, 4, 16, 4});
    auto* scale = MakeTensor({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 0, y), ACLNN_SUCCESS);
}

// DAV3510 FLOAT32承载FP4 + UINT8承载E8M0的MX路径, 走Nz接口
TEST_F(l2_weight_quant_batch_matmul_v2_cov_950, float_nz_c0_2_mx_nzapi_success)
{
    auto* x = MakeTensor({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* weight = MakeTensor({64, 4}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ_C0_2, LAYOUT_CONTIGUOUS, {1, 4, 16, 4});
    auto* scale = MakeTensor({2, 32}, ACL_UINT8, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    auto* y = MakeTensor({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, LAYOUT_CONTIGUOUS);
    EXPECT_EQ(RunNz(x, weight, scale, nullptr, nullptr, nullptr, nullptr, 32, y), ACLNN_SUCCESS);
}
