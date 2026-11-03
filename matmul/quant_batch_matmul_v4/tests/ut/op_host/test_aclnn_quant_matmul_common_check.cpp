/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_matmul_common_check.cpp
 * \brief quant_matmul_common_check.cpp 白盒UT：直接 include 被测源文件到白盒命名空间，
 *        使内部函数在本TU内可见并可直接调用，覆盖源文件内全部函数及可达分支。
 * 说明：
 *   - quant_matmul_common_check.cpp 编译进 libophost_nn_opapi.so 时符号不可见，
 *     沿用 test_aclnn_matmul.cpp / test_aclnn_batch_matmul.cpp 的直调手法，
 *     将被测源文件 include 进本TU的白盒命名空间执行；
 *   - 被测源文件依赖的 l0op 外部函数在白盒命名空间内打桩（默认直通，可通过全局开关注错），
 *     打桩位于白盒命名空间内，不与可执行文件中其它用例文件的全局 ::l0op 定义冲突；
 *   - Ops::NN::IsTransposeLastTwoDims / SwapLastTwoDimValue 为生产真实实现：真实
 *     matmul/common/op_host/op_api/matmul_util.cpp 及其依赖 cube_util.cpp/matmul.cpp 在
 *     隔离命名空间 matmul_util_real 内直编本TU（这些符号在 libophost_nn_opapi.so 中为
 *     本地符号不可跨 DSO 链接，亦不可全局直编——同可执行文件内 test_aclnn_batch_matmul.cpp
 *     已全局直编同名符号，会重复定义），白盒命名空间经 using 声明绑定该实现而非语义等价
 *     的测试副本，生产实现变更即直接影响本文件用例，杜绝副本漂移导致的虚假覆盖；
 *   - 平台信息由 UT 桩 tests/ut/op_api/stub/opdev/platform.{h,cpp} 提供，
 *     NpuArchManager 可注入 DAV_9201/DAV_9202 等 arch。
 */

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>
#include "gtest/gtest.h"

// 被测头文件 quant_matmul_common_check.h 的全部依赖头文件需先在全局作用域完成包含（但不含被测头
// 文件本身；matmul_util.h 例外：其须随下方隔离命名空间内真实 matmul_util.cpp 首次包含，使声明与
// 定义同落 matmul_util_real::Ops::NN，白盒命名空间的 using 声明才能绑定到带默认实参的真实声明）：
// 之后被测源文件在白盒命名空间内 include 被测头文件时，各依赖头因 include guard 成为空操作，
// 被测头文件内的类型别名、常量与函数声明全部落入白盒命名空间，
// 其内部相互调用不再与全局作用域的同名声明产生二义性
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "util/math_util.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_executor.h"

#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

// 隔离命名空间 matmul_util_real 内直编的 matmul_util.cpp 及其依赖引用的其余头文件也须先在全局
// 作用域完成包含，防止其在隔离命名空间内二次落地（如 opdev/shape_utils.h 展开 namespace op 成员
// 会造成 namespace op/ge 跨作用域分裂、限定引用解析失败）（沿用 test_aclnn_matmul.cpp 手法）
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/cast.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/padv3.h"
#include "op_api/op_api_def_nn.h"
#include "base/err_msg.h"

// l0op 打桩注错开关（fixture 中复位，避免用例间污染）
bool gQccContiguousReturnNull = false;
bool gQccReFormatReturnNull = false;
bool gQccReshapeReturnNull = false;
bool gQccViewCopyReturnNull = false;

// ===================== 真实 matmul_util 实现的隔离直编（沿用 test_aclnn_matmul.cpp 手法） =====================
// matmul_util.cpp 及其依赖 matmul.cpp/cube_util.cpp 编译进 libophost_nn_opapi.so 时为本地符号
// （不可跨 DSO 链接），测试 TU 无法直接链接生产实现；而全局作用域直编又会与同可执行文件内
// test_aclnn_batch_matmul.cpp 已全局直编的同名 ::Ops::NN / ::l0op 符号重复定义，
// 故将真实源文件包进隔离命名空间 matmul_util_real 在本 TU 内编译执行，
// 使下方白盒命名空间经 using 声明绑定真实的 IsTransposeLastTwoDims / SwapLastTwoDimValue。
// matmul_util.h 随源文件首次包含落入 matmul_util_real（勿加入上方全局预包含，否则声明留全局、
// 定义落隔离命名空间，两者错位导致绑定失败）。
// 该链路引用的其余 l0op 外部符号（libopapi_math_stub.so 为 ophost_nn_opapi 的私有依赖，
// 无法满足本 TU 的未定义符号）在 matmul_util_real::l0op 提供 hidden 桩，仅满足链接：
// 本文件用例的实际执行路径只触达 IsTransposeLastTwoDims / SwapLastTwoDimValue 两个纯函数，
// 不经过这些桩。
namespace matmul_util_real {

// 预开 matmul_util_real::Ops（经 using namespace ::Ops 透传全局 ::Ops::Base，全局已包含
// util/math_util.h，其无法再落入本命名空间声明 Ops::Base）：使真实源文件内
// Ops::Base::Ceil* 限定引用可解析；Ops::NN:: 限定调用仍绑定随后由 cube_util.h
// 重开的本命名空间 Ops::NN 成员（全局 ::Ops::NN 未声明，无二义）
namespace Ops {
using namespace ::Ops;
} // namespace Ops

namespace l0op {
__attribute__((visibility("hidden"))) const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype,
                                                            aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Fill(const aclTensor* dims, const aclTensor* value,
                                                            const aclIntArray* outShape, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Mul(const aclTensor* self, const aclTensor* other,
                                                           aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ReFormat(const aclTensor* x, const op::Format& format,
                                                                aclOpExecutor* executor = nullptr);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat,
                                                                 int64_t groups, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Transpose(const aclTensor* x, const aclIntArray* perm,
                                                                 aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y,
                                                                aclOpExecutor* executor);
__attribute__((visibility("hidden"))) bool MmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2,
                                                             const aclTensor* bias, const bool transposeX1,
                                                             const bool transposeX2, op::Format mat2_format,
                                                             bool supportSplitK);
} // namespace l0op

#include "../../../../common/op_host/op_api/cube_util.cpp"
#include "../../../../common/op_host/op_api/matmul.cpp"
#include "../../../../common/op_host/op_api/matmul_util.cpp"

// matmul_util_real::l0op 桩定义：仅满足链接，用例执行路径不触达，一律返回空值
namespace l0op {
const aclTensor* Cast(const aclTensor*, op::DataType, aclOpExecutor*) { return nullptr; }
const aclTensor* Contiguous(const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* Mul(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* ReFormat(const aclTensor*, const op::Format&, aclOpExecutor*) { return nullptr; }
const aclTensor* Reshape(const aclTensor*, const op::Shape&, aclOpExecutor*) { return nullptr; }
const aclTensor* Reshape(const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*) { return nullptr; }
const aclTensor* Transpose(const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
bool MmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, bool)
{
    return false;
}
} // namespace l0op

} // namespace matmul_util_real

namespace quant_matmul_common_check_whitebox {

// 被测源文件依赖的 l0op 外部符号：白盒命名空间内前置声明，随后打桩实现
namespace l0op {
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor* executor = nullptr);
const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor);
const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor);
} // namespace l0op

// 被测源文件 using Ops::Base::CeilDiv 依赖全局模板（经 using namespace ::Ops 透传解析）；
// using Ops::NN::IsTransposeLastTwoDims / SwapLastTwoDimValue 经下方 using 声明绑定到上方
// 隔离命名空间 matmul_util_real 内直编的真实 matmul_util.cpp 实现（生产代码而非本TU副本），
// SwapLastTwoDimValue 的默认实参随 matmul_util_real 内的 matmul_util.h 声明经 using 声明透传保持可用
namespace Ops {
using namespace ::Ops;
namespace NN {
using matmul_util_real::Ops::NN::IsTransposeLastTwoDims;
using matmul_util_real::Ops::NN::SwapLastTwoDimValue;
} // namespace NN
} // namespace Ops

#include "../../../op_host/op_api/quant_matmul_common_check.cpp"

// l0op 桩实现：默认直通，可通过全局开关注入 nullptr 触发被测代码的失败分支
namespace l0op {
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor*) { return gQccContiguousReturnNull ? nullptr : x; }

const aclTensor* ReFormat(const aclTensor* x, const op::Format&, aclOpExecutor*)
{
    return gQccReFormatReturnNull ? nullptr : x;
}

const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    if (gQccReshapeReturnNull || x == nullptr) {
        return nullptr;
    }
    return executor->CreateView(x, shape, x->GetViewOffset());
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor*, aclOpExecutor*)
{
    return gQccViewCopyReturnNull ? nullptr : x;
}
} // namespace l0op

} // namespace quant_matmul_common_check_whitebox

namespace qcc = quant_matmul_common_check_whitebox;

using namespace op;

namespace {

constexpr int64_t kInt4NumsInInt32 = 8; // 与被测头文件 INT4_NUMS_IN_INT32 保持一致

bool ShapeEquals(const op::Shape& shape, const std::vector<int64_t>& expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (shape.GetDim(i) != expected[i]) {
            return false;
        }
    }
    return true;
}

bool StridesEquals(const op::Strides& strides, const std::vector<int64_t>& expected)
{
    if (strides.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (strides[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

op::Shape MakeShape(const std::vector<int64_t>& dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

// viewShape 为视图shape；strides 为空表示按连续内存推导；storageShape 为空表示与视图shape一致
aclTensor* MakeAclTensor(const std::vector<int64_t>& viewShape, aclDataType dtype,
                         const std::vector<int64_t>& strides = {}, aclFormat format = ACL_FORMAT_ND,
                         const std::vector<int64_t>& storageShape = {})
{
    std::vector<int64_t> storage = storageShape.empty() ? viewShape : storageShape;
    return aclCreateTensor(viewShape.data(), viewShape.size(), dtype, strides.empty() ? nullptr : strides.data(), 0,
                           format, storage.data(), storage.size(), nullptr);
}

} // namespace

class QuantMatmulCommonCheckTest : public testing::Test {
protected:
    void SetUp() override
    {
        gQccContiguousReturnNull = false;
        gQccReFormatReturnNull = false;
        gQccReshapeReturnNull = false;
        gQccViewCopyReturnNull = false;
    }

    void TearDown() override { SetUp(); }
};

// ============================ IsNpuArch92 ============================

TEST_F(QuantMatmulCommonCheckTest, IsNpuArch92_Dav9201ReturnsTrue)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    EXPECT_TRUE(qcc::IsNpuArch92());
}

TEST_F(QuantMatmulCommonCheckTest, IsNpuArch92_Dav9202ReturnsTrue)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    EXPECT_TRUE(qcc::IsNpuArch92());
}

TEST_F(QuantMatmulCommonCheckTest, IsNpuArch92_Dav2201ReturnsFalse)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(qcc::IsNpuArch92());
}

TEST_F(QuantMatmulCommonCheckTest, IsNpuArch92_Dav3510ReturnsFalse)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_FALSE(qcc::IsNpuArch92());
}

// ============================ CheckSpecialCase ============================

TEST_F(QuantMatmulCommonCheckTest, CheckSpecialCase_DimsEqualReturnsTrue)
{
    const aclTensor* tensor = MakeAclTensor({4, 16, 16}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // firstLastDim=2, secondLastDim=1: 16 == 16
    EXPECT_TRUE(qcc::CheckSpecialCase(tensor, 2, 1));
}

TEST_F(QuantMatmulCommonCheckTest, CheckSpecialCase_DimsNotEqualReturnsFalse)
{
    const aclTensor* tensor = MakeAclTensor({4, 16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // firstLastDim=2, secondLastDim=1: 32 != 16
    EXPECT_FALSE(qcc::CheckSpecialCase(tensor, 2, 1));
}

TEST_F(QuantMatmulCommonCheckTest, CheckSpecialCase_2DDimsEqualReturnsTrue)
{
    const aclTensor* tensor = MakeAclTensor({16, 16}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // firstLastDim=1, secondLastDim=0: 16 == 16
    EXPECT_TRUE(qcc::CheckSpecialCase(tensor, 1, 0));
}

// ============================ GetTransposeAttrValue ============================

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_ContiguousKeepsTransposeFalse)
{
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // 连续内存排布（strides {32,1}）不触发转置交换，transpose 原样返回
    EXPECT_FALSE(qcc::GetTransposeAttrValue(tensor, false, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {16, 32}));
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {32, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_ContiguousKeepsTransposeTrue)
{
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(qcc::GetTransposeAttrValue(tensor, true, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {16, 32}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_TransposedFlipsAndSwapsViewShape)
{
    // strides {1,16}: 倒数第二维 stride 为 1、最后一维 stride 等于倒数第二维大小，判定为转置存储
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8, {1, 16});
    ASSERT_NE(tensor, nullptr);
    // 交换后 {32,16} 两维不等，不满足特殊case，返回 !transpose
    EXPECT_TRUE(qcc::GetTransposeAttrValue(tensor, false, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {32, 16})) << "view shape should be swapped";
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {16, 1})) << "strides should be re-derived";
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_TransposedFlipsTrueToFalse)
{
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8, {1, 16});
    ASSERT_NE(tensor, nullptr);
    EXPECT_FALSE(qcc::GetTransposeAttrValue(tensor, true, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {32, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_TransposedSpecialCaseKeepsTranspose)
{
    // {16,16} 交换后两维相等，命中 CheckSpecialCase，transpose 原样返回
    const aclTensor* tensor = MakeAclTensor({16, 16}, ACL_INT8, {1, 16});
    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(qcc::GetTransposeAttrValue(tensor, true, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {16, 16}));
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {16, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_SkipSpecialCaseFlips)
{
    // checkSpecialCase=false 时不做特殊case校验，直接返回 !transpose
    const aclTensor* tensor = MakeAclTensor({16, 16}, ACL_INT8, {1, 16});
    ASSERT_NE(tensor, nullptr);
    EXPECT_FALSE(qcc::GetTransposeAttrValue(tensor, true, false));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {16, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_3DTransposedFlips)
{
    // 3D 转置存储：dim1 stride 1、dim2 stride 等于 dim1 大小，且 batch 维连续
    const aclTensor* tensor = MakeAclTensor({2, 16, 32}, ACL_INT8, {512, 1, 16});
    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(qcc::GetTransposeAttrValue(tensor, false, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {2, 32, 16}));
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {512, 16, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, GetTransposeAttrValue_3DIgnoresBatchStride)
{
    // GetTransposeAttrValue 只检查最后两维 stride，不校验 batch 维（与 IsTransposeLastTwoDims 的差异）
    const aclTensor* tensor = MakeAclTensor({2, 16, 32}, ACL_INT8, {999, 1, 16});
    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(qcc::GetTransposeAttrValue(tensor, false, true));
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {2, 32, 16}));
}

// ============================ GetWeightNzShape ============================

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int8TransposeFalse)
{
    const aclTensor* tensor = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // k=32, n=16, nz_k0=32: k1=CeilDiv(32,16)=2, n1=CeilDiv(16,32)=1, 追加 {n1,k1,16,32}
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, false);
    EXPECT_TRUE(ShapeEquals(nzShape, {1, 2, 16, 32}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int8TransposeTrue)
{
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // k=32, n=16, nz_k0=32: k1=CeilDiv(32,32)=1, n1=CeilDiv(16,16)=1, 追加 {k1,n1,16,32}
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, true);
    EXPECT_TRUE(ShapeEquals(nzShape, {1, 1, 16, 32}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int32TransposeTrue)
{
    const aclTensor* tensor = MakeAclTensor({16, 64}, ACL_INT32);
    ASSERT_NE(tensor, nullptr);
    // int32 的 nz_k0=64: k1=CeilDiv(64,64)=1, n1=CeilDiv(16,16)=1
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, true);
    EXPECT_TRUE(ShapeEquals(nzShape, {1, 1, 16, 64}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int32TransposeFalse)
{
    const aclTensor* tensor = MakeAclTensor({16, 128}, ACL_INT32);
    ASSERT_NE(tensor, nullptr);
    // k=16, n=128, nz_k0=64: k1=CeilDiv(16,16)=1, n1=CeilDiv(128,64)=2, 追加 {n1,k1,16,64}
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, false);
    EXPECT_TRUE(ShapeEquals(nzShape, {2, 1, 16, 64}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int4TransposeFalse3D)
{
    const aclTensor* tensor = MakeAclTensor({2, 32, 16}, ACL_INT4);
    ASSERT_NE(tensor, nullptr);
    // int4 的 nz_k0=64: k1=CeilDiv(32,16)=2, n1=CeilDiv(16,64)=1, 保留前置维 {2} 后追加 {n1,k1,16,64}
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, false);
    EXPECT_TRUE(ShapeEquals(nzShape, {2, 1, 2, 16, 64}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Int8TransposeTrue3DUnaligned)
{
    const aclTensor* tensor = MakeAclTensor({3, 16, 33}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    // k=33 不按32对齐: k1=CeilDiv(33,32)=2, n1=CeilDiv(16,16)=1, 保留前置维 {3} 后追加 {k1,n1,16,32}
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, true);
    EXPECT_TRUE(ShapeEquals(nzShape, {3, 2, 1, 16, 32}));
}

TEST_F(QuantMatmulCommonCheckTest, GetWeightNzShape_Float16TransposeFalse)
{
    const aclTensor* tensor = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(tensor, nullptr);
    // 非int32/int4 的 nz_k0=32: k1=CeilDiv(16,16)=1, n1=CeilDiv(16,32)=1
    op::Shape nzShape = qcc::GetWeightNzShape(tensor, false);
    EXPECT_TRUE(ShapeEquals(nzShape, {1, 1, 16, 32}));
}

// ============================ CheckWeightNzStorageShape ============================

TEST_F(QuantMatmulCommonCheckTest, CheckWeightNzStorageShape_SameProductReturnsTrue)
{
    // 元素总数相同（1024 == 1024），维数可以不同
    op::Shape nzShape = MakeShape({1, 2, 16, 32});
    op::Shape storageShape = MakeShape({2, 8, 64});
    EXPECT_TRUE(qcc::CheckWeightNzStorageShape(nzShape, storageShape));
}

TEST_F(QuantMatmulCommonCheckTest, CheckWeightNzStorageShape_DifferentProductReturnsFalse)
{
    op::Shape nzShape = MakeShape({1, 2, 16, 32});
    op::Shape storageShape = MakeShape({1, 2, 16, 16});
    EXPECT_FALSE(qcc::CheckWeightNzStorageShape(nzShape, storageShape));
}

TEST_F(QuantMatmulCommonCheckTest, CheckWeightNzStorageShape_SingleDimReturnsTrue)
{
    op::Shape nzShape = MakeShape({1024});
    op::Shape storageShape = MakeShape({1024});
    EXPECT_TRUE(qcc::CheckWeightNzStorageShape(nzShape, storageShape));
}

// ============================ SetTensorToNZFormat ============================

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNZFormat_SetsNzFormatAndKeepsViewShape)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* input = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(input, nullptr);
    op::Shape nzShape = MakeShape({1, 2, 16, 32});

    const aclTensor* nzTensor = qcc::SetTensorToNZFormat(input, nzShape, executor);
    ASSERT_NE(nzTensor, nullptr);
    EXPECT_EQ(nzTensor->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(nzTensor->GetOriginalFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(nzTensor->GetDataType(), op::DataType::DT_INT8);
    EXPECT_TRUE(ShapeEquals(nzTensor->GetViewShape(), {32, 16})) << "view shape should keep input's";
    EXPECT_TRUE(ShapeEquals(nzTensor->GetStorageShape(), {1, 2, 16, 32})) << "storage shape should be nz shape";
    EXPECT_TRUE(StridesEquals(nzTensor->GetViewStrides(), {16, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNZFormat_3DInputKeepsViewShape)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* input = MakeAclTensor({2, 32, 16}, ACL_INT8);
    ASSERT_NE(input, nullptr);
    op::Shape nzShape = MakeShape({2, 1, 2, 16, 32});

    const aclTensor* nzTensor = qcc::SetTensorToNZFormat(input, nzShape, executor);
    ASSERT_NE(nzTensor, nullptr);
    EXPECT_EQ(nzTensor->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
    EXPECT_TRUE(ShapeEquals(nzTensor->GetViewShape(), {2, 32, 16}));
    EXPECT_TRUE(ShapeEquals(nzTensor->GetStorageShape(), {2, 1, 2, 16, 32}));
}

// ============================ TensorContiguousProcess ============================

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_NullTensorReturnsTrue)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    const aclTensor* tensor = nullptr;
    bool transpose = false;
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, uniqueExecutor.get()));
    EXPECT_EQ(tensor, nullptr);
    EXPECT_FALSE(transpose);
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_ContiguousKeepsFlagFalse)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    const aclTensor* input = tensor;
    bool transpose = false;
    // 连续排布走 l0op::Contiguous（桩直通），transpose 不翻转
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, uniqueExecutor.get()));
    EXPECT_NE(tensor, nullptr);
    EXPECT_FALSE(transpose);
    EXPECT_EQ(tensor, input);
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_ContiguousKeepsFlagTrue)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    bool transpose = true;
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, uniqueExecutor.get()));
    EXPECT_NE(tensor, nullptr);
    EXPECT_TRUE(transpose);
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_Transposed2DSwapsAndFlips)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8, {1, 16});
    ASSERT_NE(tensor, nullptr);
    const aclTensor* input = tensor;
    bool transpose = false;
    // 转置存储走 CreateView 交换最后两维，transpose 翻转
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, executor));
    EXPECT_NE(tensor, nullptr);
    EXPECT_NE(tensor, input);
    EXPECT_TRUE(transpose);
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {32, 16}));
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {16, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_Transposed3DSwapsAndFlips)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* tensor = MakeAclTensor({2, 16, 32}, ACL_INT8, {512, 1, 16});
    ASSERT_NE(tensor, nullptr);
    bool transpose = true;
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, executor));
    EXPECT_NE(tensor, nullptr);
    EXPECT_FALSE(transpose);
    EXPECT_TRUE(ShapeEquals(tensor->GetViewShape(), {2, 32, 16}));
    EXPECT_TRUE(StridesEquals(tensor->GetViewStrides(), {512, 16, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_BatchNotContiguousUsesContiguous)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    // batch 维 stride 不满足连续性，IsTransposeLastTwoDims 判假，走 l0op::Contiguous（桩直通）
    const aclTensor* tensor = MakeAclTensor({2, 16, 32}, ACL_INT8, {256, 1, 16});
    ASSERT_NE(tensor, nullptr);
    const aclTensor* input = tensor;
    bool transpose = false;
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, executor));
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor, input);
    EXPECT_FALSE(transpose);
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_AllOneDimUsesContiguous)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    // 最后两维均为1时 IsTransposeLastTwoDims 判假，走 l0op::Contiguous
    const aclTensor* tensor = MakeAclTensor({1, 1}, ACL_INT8, {1, 1});
    ASSERT_NE(tensor, nullptr);
    const aclTensor* input = tensor;
    bool transpose = false;
    EXPECT_TRUE(qcc::TensorContiguousProcess(tensor, transpose, executor));
    EXPECT_EQ(tensor, input);
    EXPECT_FALSE(transpose);
}

TEST_F(QuantMatmulCommonCheckTest, TensorContiguousProcess_ContiguousFailReturnsFalse)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    const aclTensor* tensor = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(tensor, nullptr);
    bool transpose = false;
    gQccContiguousReturnNull = true;
    EXPECT_FALSE(qcc::TensorContiguousProcess(tensor, transpose, uniqueExecutor.get()));
    EXPECT_EQ(tensor, nullptr);
    EXPECT_FALSE(transpose);
}

// ============================ WeightNZCaseProcess ============================

TEST_F(QuantMatmulCommonCheckTest, WeightNZCaseProcess_NzFormatSetsOriginalShape)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ, {1, 2, 16, 32});
    ASSERT_NE(x2, nullptr);
    const aclTensor* input = x2;
    // aclCreateTensor 构造时 original shape 等于 storage shape
    ASSERT_TRUE(ShapeEquals(x2->GetOriginalShape(), {1, 2, 16, 32}));

    bool transposeX2 = false;
    EXPECT_EQ(qcc::WeightNZCaseProcess(x2, transposeX2, executor), ACLNN_SUCCESS);
    // 已是 NZ 格式：不做连续化，仅将 original shape 设置为视图shape
    EXPECT_EQ(x2, input);
    EXPECT_FALSE(transposeX2);
    EXPECT_TRUE(ShapeEquals(x2->GetOriginalShape(), {32, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, WeightNZCaseProcess_NzC032ConvertsToNormalNz)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ_C0_32, {2, 1, 16, 32});
    ASSERT_NE(x2, nullptr);
    const aclTensor* input = x2;

    bool transposeX2 = false;
    EXPECT_EQ(qcc::WeightNZCaseProcess(x2, transposeX2, executor), ACLNN_SUCCESS);
    // 特殊 NZ 格式经 SetSpecilNZTensorToNormalNZFormat 转换为普通 NZ
    EXPECT_NE(x2, input);
    EXPECT_FALSE(transposeX2);
    EXPECT_EQ(x2->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(x2->GetViewFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(x2->GetOriginalFormat(), op::Format::FORMAT_ND);
    EXPECT_TRUE(ShapeEquals(x2->GetViewShape(), {32, 16}));
    EXPECT_TRUE(ShapeEquals(x2->GetStorageShape(), {2, 1, 16, 32}));
    // 转换前 WeightNZCaseProcess 先将 original shape 置为视图shape，转换过程按入参透传
    EXPECT_TRUE(ShapeEquals(x2->GetOriginalShape(), {32, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, WeightNZCaseProcess_NdContiguousSuccess)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(x2, nullptr);
    const aclTensor* input = x2;

    bool transposeX2 = false;
    EXPECT_EQ(qcc::WeightNZCaseProcess(x2, transposeX2, executor), ACLNN_SUCCESS);
    EXPECT_EQ(x2, input);
    EXPECT_FALSE(transposeX2);
}

TEST_F(QuantMatmulCommonCheckTest, WeightNZCaseProcess_NdTransposedSwaps)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x2 = MakeAclTensor({16, 32}, ACL_INT8, {1, 16});
    ASSERT_NE(x2, nullptr);
    bool transposeX2 = false;
    EXPECT_EQ(qcc::WeightNZCaseProcess(x2, transposeX2, executor), ACLNN_SUCCESS);
    EXPECT_NE(x2, nullptr);
    EXPECT_TRUE(transposeX2);
    EXPECT_TRUE(ShapeEquals(x2->GetViewShape(), {32, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, WeightNZCaseProcess_ContiguousFailReturnsError)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(x2, nullptr);
    bool transposeX2 = false;
    gQccContiguousReturnNull = true;
    EXPECT_EQ(qcc::WeightNZCaseProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    EXPECT_EQ(x2, nullptr);
}

// ============================ ConvertTensorToInt4 ============================

TEST_F(QuantMatmulCommonCheckTest, ConvertTensorToInt4_Int32To2DInt4)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* input = MakeAclTensor({2, 4}, ACL_INT32);
    ASSERT_NE(input, nullptr);
    const aclTensor* int4Tensor = qcc::ConvertTensorToInt4(input, executor);
    ASSERT_NE(int4Tensor, nullptr);
    // int32 转 int4: 最后一维放大8倍，dtype 修改为 DT_INT4
    EXPECT_EQ(int4Tensor->GetDataType(), op::DataType::DT_INT4);
    EXPECT_TRUE(ShapeEquals(int4Tensor->GetViewShape(), {2, 4 * kInt4NumsInInt32}));
    EXPECT_TRUE(ShapeEquals(int4Tensor->GetStorageShape(), {2, 4 * kInt4NumsInInt32}));
    EXPECT_TRUE(StridesEquals(int4Tensor->GetViewStrides(), {4 * kInt4NumsInInt32, 1}));
}

TEST_F(QuantMatmulCommonCheckTest, ConvertTensorToInt4_Int32To3DInt4)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* input = MakeAclTensor({2, 4, 8}, ACL_INT32);
    ASSERT_NE(input, nullptr);
    const aclTensor* int4Tensor = qcc::ConvertTensorToInt4(input, executor);
    ASSERT_NE(int4Tensor, nullptr);
    EXPECT_EQ(int4Tensor->GetDataType(), op::DataType::DT_INT4);
    EXPECT_TRUE(ShapeEquals(int4Tensor->GetViewShape(), {2, 4, 8 * kInt4NumsInInt32}));
}

// ============================ InputPreProcessA4W4 / A4W4CaseProcess ============================

TEST_F(QuantMatmulCommonCheckTest, InputPreProcessA4W4_X2Int32OnlyConverted)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT32);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    const aclTensor* inputX1 = x1;
    const aclTensor* inputX2 = x2;

    EXPECT_EQ(qcc::InputPreProcessA4W4(x1, x2, executor), ACLNN_SUCCESS);
    // x1 非 int32 不转换
    EXPECT_EQ(x1, inputX1);
    EXPECT_EQ(x1->GetDataType(), op::DataType::DT_INT8);
    // x2 为 int32 转换为 int4
    EXPECT_NE(x2, inputX2);
    EXPECT_EQ(x2->GetDataType(), op::DataType::DT_INT4);
    EXPECT_TRUE(ShapeEquals(x2->GetViewShape(), {32, 16 * kInt4NumsInInt32}));
}

TEST_F(QuantMatmulCommonCheckTest, InputPreProcessA4W4_X1Int32OnlyConverted)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT32);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    const aclTensor* inputX1 = x1;
    const aclTensor* inputX2 = x2;

    EXPECT_EQ(qcc::InputPreProcessA4W4(x1, x2, executor), ACLNN_SUCCESS);
    EXPECT_NE(x1, inputX1);
    EXPECT_EQ(x1->GetDataType(), op::DataType::DT_INT4);
    EXPECT_TRUE(ShapeEquals(x1->GetViewShape(), {16, 32 * kInt4NumsInInt32}));
    EXPECT_EQ(x2, inputX2);
    EXPECT_EQ(x2->GetDataType(), op::DataType::DT_INT8);
}

TEST_F(QuantMatmulCommonCheckTest, InputPreProcessA4W4_BothConverted)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT32);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT32);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    const aclTensor* inputX1 = x1;
    const aclTensor* inputX2 = x2;

    EXPECT_EQ(qcc::InputPreProcessA4W4(x1, x2, executor), ACLNN_SUCCESS);
    EXPECT_NE(x1, inputX1);
    EXPECT_NE(x2, inputX2);
    EXPECT_EQ(x1->GetDataType(), op::DataType::DT_INT4);
    EXPECT_EQ(x2->GetDataType(), op::DataType::DT_INT4);
}

TEST_F(QuantMatmulCommonCheckTest, InputPreProcessA4W4_NeitherConverted)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    const aclTensor* inputX1 = x1;
    const aclTensor* inputX2 = x2;

    EXPECT_EQ(qcc::InputPreProcessA4W4(x1, x2, executor), ACLNN_SUCCESS);
    EXPECT_EQ(x1, inputX1);
    EXPECT_EQ(x2, inputX2);
}

TEST_F(QuantMatmulCommonCheckTest, A4W4CaseProcess_BothInt32Converted)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT32);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT32);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);

    EXPECT_EQ(qcc::A4W4CaseProcess(x1, x2, executor), ACLNN_SUCCESS);
    EXPECT_EQ(x1->GetDataType(), op::DataType::DT_INT4);
    EXPECT_EQ(x2->GetDataType(), op::DataType::DT_INT4);
}

// ============================ SetTensorToNDFormat / GetNDFormat ============================

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNDFormat_NullInputReturnsNull)
{
    EXPECT_EQ(qcc::SetTensorToNDFormat(nullptr), nullptr);
}

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNDFormat_NzStorageReturnsInput)
{
    const aclTensor* input = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ, {1, 2, 16, 32});
    ASSERT_NE(input, nullptr);
    // 存储格式已是 FRACTAL_NZ：直接返回输入本身
    EXPECT_EQ(qcc::SetTensorToNDFormat(input), input);
}

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNDFormat_NdStorageCallsReFormat)
{
    const aclTensor* input = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(input, nullptr);
    // 非 NZ 存储格式走 l0op::ReFormat（桩直通返回输入）
    EXPECT_EQ(qcc::SetTensorToNDFormat(input), input);
}

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNDFormat_NzC032CallsReFormat)
{
    const aclTensor* input = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ_C0_32, {2, 1, 16, 32});
    ASSERT_NE(input, nullptr);
    // 特殊 NZ 格式 != FORMAT_FRACTAL_NZ，同样走 l0op::ReFormat
    EXPECT_EQ(qcc::SetTensorToNDFormat(input), input);
}

TEST_F(QuantMatmulCommonCheckTest, SetTensorToNDFormat_ReFormatFailReturnsNull)
{
    const aclTensor* input = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(input, nullptr);
    gQccReFormatReturnNull = true;
    EXPECT_EQ(qcc::SetTensorToNDFormat(input), nullptr);
}

TEST_F(QuantMatmulCommonCheckTest, GetNDFormat_NullInputReturnsNull) { EXPECT_EQ(qcc::GetNDFormat(nullptr), nullptr); }

TEST_F(QuantMatmulCommonCheckTest, GetNDFormat_NzInputReturnsInput)
{
    const aclTensor* input = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ, {1, 2, 16, 32});
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(qcc::GetNDFormat(input), input);
}

TEST_F(QuantMatmulCommonCheckTest, GetNDFormat_NdInputReturnsNonNull)
{
    const aclTensor* input = MakeAclTensor({16, 32}, ACL_INT8);
    ASSERT_NE(input, nullptr);
    EXPECT_NE(qcc::GetNDFormat(input), nullptr);
}

// ============================ GetDtypeAndTranspose ============================

TEST_F(QuantMatmulCommonCheckTest, GetDtypeAndTranspose_BasicOutDtypeAndTranspose)
{
    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({16, 32}, ACL_INT8, {1, 16});
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);

    int64_t dtype = -1;
    bool transposeX1 = false;
    bool transposeX2 = false;
    qcc::GetDtypeAndTranspose(std::make_tuple(x1, x2, out), dtype, transposeX1, transposeX2);
    // dtype 取出参 out 的 dtype
    EXPECT_EQ(dtype, static_cast<int64_t>(op::DataType::DT_FLOAT16));
    // x1 连续：transposeX1 原样返回；x2 转置存储：transposeX2 翻转
    EXPECT_FALSE(transposeX1);
    EXPECT_TRUE(transposeX2);
    // GetTransposeAttrValue 对 x2 视图shape做了交换
    EXPECT_TRUE(ShapeEquals(x2->GetViewShape(), {32, 16}));
}

TEST_F(QuantMatmulCommonCheckTest, GetDtypeAndTranspose_BothContiguousKeepsFlags)
{
    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_INT32);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);

    int64_t dtype = -1;
    bool transposeX1 = true;
    bool transposeX2 = false;
    qcc::GetDtypeAndTranspose(std::make_tuple(x1, x2, out), dtype, transposeX1, transposeX2);
    EXPECT_EQ(dtype, static_cast<int64_t>(op::DataType::DT_INT32));
    EXPECT_TRUE(transposeX1);
    EXPECT_FALSE(transposeX2);
}

// ============================ SetSpecilNZTensorToNormalNZFormat ============================

TEST_F(QuantMatmulCommonCheckTest, SetSpecilNZTensorToNormalNZFormat_ConvertsFormats)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* input = MakeAclTensor({32, 16}, ACL_INT8, {}, ACL_FORMAT_FRACTAL_NZ_C0_32, {2, 1, 16, 32});
    ASSERT_NE(input, nullptr);
    const aclTensor* inputTensor = input;

    EXPECT_EQ(qcc::SetSpecilNZTensorToNormalNZFormat(input, executor), ACLNN_SUCCESS);
    EXPECT_NE(input, inputTensor);
    EXPECT_EQ(input->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(input->GetViewFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(input->GetOriginalFormat(), op::Format::FORMAT_ND);
    EXPECT_TRUE(ShapeEquals(input->GetViewShape(), {32, 16}));
    EXPECT_TRUE(ShapeEquals(input->GetStorageShape(), {2, 1, 16, 32}));
    EXPECT_TRUE(ShapeEquals(input->GetOriginalShape(), {2, 1, 16, 32}));
}

// ============================ SpecialOutputProcess ============================

TEST_F(QuantMatmulCommonCheckTest, SpecialOutputProcess_NullMatmulRetReturnsError)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = nullptr;

    EXPECT_EQ(qcc::SpecialOutputProcess(x1, x2, out, matmulRet, executor), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(QuantMatmulCommonCheckTest, SpecialOutputProcess_SpecialCaseReshapes)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    // x1、x2 均为2维，out 为3维且倒数第二维为1：触发 reshape
    const aclTensor* x1 = MakeAclTensor({1, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 1}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({1, 1, 1}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({1, 1}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);

    EXPECT_EQ(qcc::SpecialOutputProcess(x1, x2, out, matmulRet, executor), ACLNN_SUCCESS);
    EXPECT_NE(matmulRet, nullptr);
    EXPECT_TRUE(ShapeEquals(matmulRet->GetViewShape(), {1, 1, 1})) << "matmulRet should be reshaped to out shape";
}

TEST_F(QuantMatmulCommonCheckTest, SpecialOutputProcess_ReshapeFailReturnsError)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({1, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 1}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({1, 1, 1}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({1, 1}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);

    gQccReshapeReturnNull = true;
    EXPECT_EQ(qcc::SpecialOutputProcess(x1, x2, out, matmulRet, executor), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(QuantMatmulCommonCheckTest, SpecialOutputProcess_NonSpecialKeepsMatmulRet)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({16, 16}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);
    const aclTensor* inputMatmulRet = matmulRet;

    // out 为2维，不触发 reshape，matmulRet 原样保留
    EXPECT_EQ(qcc::SpecialOutputProcess(x1, x2, out, matmulRet, executor), ACLNN_SUCCESS);
    EXPECT_EQ(matmulRet, inputMatmulRet);
}

TEST_F(QuantMatmulCommonCheckTest, SpecialOutputProcess_Out3DMdimNotOneKeepsMatmulRet)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({2, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 8}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({2, 4, 8}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({2, 8}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);
    const aclTensor* inputMatmulRet = matmulRet;

    // out 为3维但倒数第二维不为1，不触发 reshape
    EXPECT_EQ(qcc::SpecialOutputProcess(x1, x2, out, matmulRet, executor), ACLNN_SUCCESS);
    EXPECT_EQ(matmulRet, inputMatmulRet);
}

// ============================ PostMatmulCalcProcess ============================

TEST_F(QuantMatmulCommonCheckTest, PostMatmulCalcProcess_NullMatmulRetReturnsError)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);

    EXPECT_EQ(qcc::PostMatmulCalcProcess(nullptr, x1, x2, out, executor), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(QuantMatmulCommonCheckTest, PostMatmulCalcProcess_BasicSuccess)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({16, 16}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);

    EXPECT_EQ(qcc::PostMatmulCalcProcess(matmulRet, x1, x2, out, executor), ACLNN_SUCCESS);
}

TEST_F(QuantMatmulCommonCheckTest, PostMatmulCalcProcess_ViewCopyFailReturnsError)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    const aclTensor* x1 = MakeAclTensor({16, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 16}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({16, 16}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({16, 16}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);

    gQccViewCopyReturnNull = true;
    EXPECT_EQ(qcc::PostMatmulCalcProcess(matmulRet, x1, x2, out, executor), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(QuantMatmulCommonCheckTest, PostMatmulCalcProcess_SpecialOutputSuccess)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    aclOpExecutor* executor = uniqueExecutor.get();

    // 特殊场景：x1、x2 为2维且 out 为3维（倒数第二维为1），先 reshape 再 viewcopy
    const aclTensor* x1 = MakeAclTensor({1, 32}, ACL_INT8);
    const aclTensor* x2 = MakeAclTensor({32, 1}, ACL_INT8);
    const aclTensor* out = MakeAclTensor({1, 1, 1}, ACL_FLOAT16);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    const aclTensor* matmulRet = executor->AllocTensor(op::Shape({1, 1}), op::DataType::DT_FLOAT16);
    ASSERT_NE(matmulRet, nullptr);

    EXPECT_EQ(qcc::PostMatmulCalcProcess(matmulRet, x1, x2, out, executor), ACLNN_SUCCESS);
}
