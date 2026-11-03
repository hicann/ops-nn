/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_batch_matmul_v4_polling_tiling.cpp
 * \brief quant_batch_matmul_v4_polling_tiling.cpp 的单元测试
 */

// op_host UT 目标以 -fno-access-control 编译，可直接访问被测类的 private/protected 成员，
// 因此除公共接口（SetShape/SetDtypeBits/SetLimitParams/SetPlatformInfo/ComputeTiling）外，
// 对 CheckBufferSize/GetMte2Cost/GetBasicBlockSolution/CalculateBasicBlockTiling/
// ClosePreAllocCond/GetPollingTilingSolution 等私有分支做白盒覆盖。
// 平台与限制参数取值与 pergroup_arch92 tiling 的生产调用保持一致。
// 被测头文件依赖 "common/op_host/math_util_nn.h"（仅含 -I matmul 的目标可解析），op_host UT
// 目标没有该包含路径（见 test_quant_batch_matmul_v4_pergroup_arch92_tiling.cpp 头部注释），
// 无法直接 include，故下方按被测头文件逐字镜像声明接口，函数实现统一由 tiling 静态库链接提供。
// 注意：若 op_host/op_tiling/arch35/quant_batch_matmul_v4_polling_tiling.h 结构变更，需同步
// 修改下方镜像声明；镜像后的 static_assert 与 SettersUpdateState 的字段回读断言用于暴露不一致。
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "common/inc/error_util.h"                   // ge::graphStatus
#include "../../../../common/op_host/math_util_nn.h" // ops::CeilDiv，operator< 依赖

// ===================== 被测接口镜像声明（与被测头文件逐字一致，勿随意改动） =====================
namespace optiling {
struct PollingBasicBlockParam {
    uint64_t mSize = 0;
    uint64_t nSize = 0;
    uint64_t kSize = 0;
    uint64_t groupSize = 0;
    uint64_t singleM = 0;
    uint64_t singleN = 0;
    uint64_t mDim = 0;
    uint64_t nDim = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t aDtypeBits = 0;
    uint64_t bDtypeBits = 0;
    uint64_t x2ScaleDtypeBits = 0;
    uint64_t x1ScaleDtypeBits = 0;
    uint64_t scaleFactor = 0;
    double mte2Cost = 0;
    bool enablePreAlloc = true; // MTE2是否开启PreAllocation

    bool operator<(const PollingBasicBlockParam& other) const
    {
        if (mte2Cost != other.mte2Cost) {
            return mte2Cost < other.mte2Cost;
        }
        if (singleM % baseM != 0) { // 优先选择核内无尾块的解
            return false;
        }
        if (ops::CeilDiv(singleN, baseN) != ops::CeilDiv(other.singleN, other.baseN)) {
            return ops::CeilDiv(singleN, baseN) < ops::CeilDiv(other.singleN, other.baseN);
        }
        if (baseN != other.baseN) {
            return baseN < other.baseN;
        }
        return baseM > other.baseM;
    }

    bool isValid() const
    {
        return mSize != 0 && nSize != 0 && kSize != 0 && groupSize != 0 && aDtypeBits != 0 && bDtypeBits != 0 &&
               x1ScaleDtypeBits != 0 && x2ScaleDtypeBits != 0;
    }
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "PollingBasicBlockParam { "
           << "mSize: " << mSize << ", "
           << "nSize: " << nSize << ", "
           << "kSize: " << kSize << ", "
           << "groupSize: " << groupSize << ", "
           << "singleM: " << singleM << ", "
           << "singleN: " << singleN << ", "
           << "mDim: " << mDim << ", "
           << "nDim: " << nDim << ", "
           << "baseM: " << baseM << ", "
           << "baseN: " << baseN << ", "
           << "baseK: " << baseK << ", "
           << "aDtypeBits: " << aDtypeBits << ", "
           << "bDtypeBits: " << bDtypeBits << ", "
           << "x2ScaleDtypeBits: " << x2ScaleDtypeBits << ", "
           << "x1ScaleDtypeBits: " << x1ScaleDtypeBits << ", "
           << "scaleFactor: " << scaleFactor << ", "
           << "mte2Cost: " << mte2Cost << " }";
        return ss.str();
    }
};

struct PlatformInfo {
    uint64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    double l2HbmBWRatio = 0;
    bool isValid() const
    {
        return aicNum != 0 && ubSize != 0 && l1Size != 0 && l0ASize != 0 && l0BSize != 0 && l0CSize != 0 &&
               l2HbmBWRatio > 1e-6;
    }
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "PlatformInfos { "
           << "aicNum: " << aicNum << ", "
           << "ubSize: " << ubSize << ", "
           << "l1Size: " << l1Size << ", "
           << "l0ASize: " << l0ASize << ", "
           << "l0BSize: " << l0BSize << ", "
           << "l0CSize: " << l0CSize << ", "
           << "l2HbmBWRatio: " << l2HbmBWRatio << " }";
        return ss.str();
    }
};

struct PollingLimitParam {
    uint64_t alignSizeN = 0;
    uint64_t maxBaseMSize = 0;
    uint64_t maxBaseNSize = 0;
    uint64_t maxMNSize = 0;
};

class QuantBatchMatmulV4PollingTiling {
public:
    QuantBatchMatmulV4PollingTiling() {}
    ~QuantBatchMatmulV4PollingTiling() = default;

    void SetShape(const uint64_t mSize, const uint64_t nSize, const uint64_t kSize, const uint64_t groupSize);
    void SetDtypeBits(const uint64_t aDtypeBits, const uint64_t bDtypeBits, const uint64_t x1ScaleDtypeBits,
                      const uint64_t x2ScaleDtypeBits);
    void SetPlatformInfo(const PlatformInfo& platformInfos);
    void SetLimitParams(const uint64_t alignSizeN, const uint64_t maxBaseMSize, const uint64_t maxBaseNSize,
                        const uint64_t maxMNSize);
    ge::graphStatus ComputeTiling();
    const PollingBasicBlockParam& GetTilingResult() const { return basicBlockParam_; }

protected:
    PollingBasicBlockParam basicBlockParam_;
    std::vector<PollingBasicBlockParam> optionalResults_;
    PlatformInfo platformInfos_;
    PollingLimitParam limitParam_;

private:
    void GetBasicBlockSolution();
    void CalculateBasicBlockTiling();
    bool CheckBufferSize(const uint64_t baseM, const uint64_t baseN);
    void GetMte2Cost();
    double BWUtilization(const uint64_t mDim, const uint64_t nDim) const;
    bool GetPollingTilingSolution();
    bool ClosePreAllocCond();
};
} // namespace optiling
// ===================== 镜像声明结束 =====================

// 镜像布局校验（lp64）：固定与被测头文件一致的内存布局预期
static_assert(sizeof(optiling::PollingBasicBlockParam) == 144, "mirror layout mismatch: PollingBasicBlockParam");
static_assert(sizeof(optiling::PlatformInfo) == 56, "mirror layout mismatch: PlatformInfo");
static_assert(sizeof(optiling::PollingLimitParam) == 32, "mirror layout mismatch: PollingLimitParam");
static_assert(sizeof(optiling::QuantBatchMatmulV4PollingTiling) == 256, "mirror layout mismatch: tiling class");
static_assert(offsetof(optiling::QuantBatchMatmulV4PollingTiling, basicBlockParam_) == 0,
              "mirror layout mismatch: basicBlockParam_");
static_assert(offsetof(optiling::QuantBatchMatmulV4PollingTiling, optionalResults_) == 144,
              "mirror layout mismatch: optionalResults_");
static_assert(offsetof(optiling::QuantBatchMatmulV4PollingTiling, platformInfos_) == 168,
              "mirror layout mismatch: platformInfos_");
static_assert(offsetof(optiling::QuantBatchMatmulV4PollingTiling, limitParam_) == 224,
              "mirror layout mismatch: limitParam_");

namespace {
using optiling::PlatformInfo;
using optiling::PollingBasicBlockParam;
using optiling::QuantBatchMatmulV4PollingTiling;

// 生产调用侧限制参数：BASE_N_ALIGN_SIZE / MAX_BASE_M_BLOCK_SIZE / MAX_BASE_N_BLOCK_SIZE / MAX_BASE_MN_SIZE
constexpr uint64_t kAlignSizeN = 64;
constexpr uint64_t kMaxBaseMSize = 128;
constexpr uint64_t kMaxBaseNSize = 512;
constexpr uint64_t kMaxMNSize = 32768;
constexpr double kL2HbmBwRatio = 1.8;

PlatformInfo MakePlatformInfo(const uint64_t aicNum = 32, const uint64_t ubSize = 196352,
                              const uint64_t l1Size = 524032, const uint64_t l0aSize = 65536,
                              const uint64_t l0bSize = 65536)
{
    PlatformInfo info;
    info.aicNum = aicNum;
    info.ubSize = ubSize;
    info.l1Size = l1Size;
    info.l0ASize = l0aSize;
    info.l0BSize = l0bSize;
    info.l0CSize = 262144;
    info.l2HbmBWRatio = kL2HbmBwRatio;
    return info;
}

// 构造可通过 isValid 校验的完整输入，部分用例再按需覆盖平台/限制参数
void InitValidTiling(QuantBatchMatmulV4PollingTiling& tiling, const uint64_t mSize = 128, const uint64_t nSize = 128,
                     const uint64_t kSize = 512, const uint64_t groupSize = 128)
{
    tiling.SetShape(mSize, nSize, kSize, groupSize);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo());
}
} // namespace

// ---------------- 结构体辅助逻辑（头文件内联，被 cpp 逻辑依赖） ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, BasicBlockParamIsValid)
{
    PollingBasicBlockParam param;
    EXPECT_FALSE(param.isValid()); // 默认全 0

    param.mSize = 16;
    param.nSize = 64;
    param.kSize = 512;
    param.groupSize = 128;
    param.aDtypeBits = 8;
    param.bDtypeBits = 16;
    param.x1ScaleDtypeBits = 32;
    param.x2ScaleDtypeBits = 32;
    EXPECT_TRUE(param.isValid());

    // 逐个字段清零，均应失效
    const std::vector<uint64_t PollingBasicBlockParam::*> fields = {&PollingBasicBlockParam::mSize,
                                                                    &PollingBasicBlockParam::nSize,
                                                                    &PollingBasicBlockParam::kSize,
                                                                    &PollingBasicBlockParam::groupSize,
                                                                    &PollingBasicBlockParam::aDtypeBits,
                                                                    &PollingBasicBlockParam::bDtypeBits,
                                                                    &PollingBasicBlockParam::x1ScaleDtypeBits,
                                                                    &PollingBasicBlockParam::x2ScaleDtypeBits};
    for (const auto field : fields) {
        PollingBasicBlockParam temp = param;
        temp.*field = 0;
        EXPECT_FALSE(temp.isValid());
    }
}

TEST(QuantBatchMatmulV4PollingTilingTest, PlatformInfoIsValidAndToString)
{
    PlatformInfo info;
    EXPECT_FALSE(info.isValid()); // 默认全 0

    info = MakePlatformInfo();
    EXPECT_TRUE(info.isValid());

    // 逐个字段破坏均应失效
    info.aicNum = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.ubSize = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.l1Size = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.l0ASize = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.l0BSize = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.l0CSize = 0;
    EXPECT_FALSE(info.isValid());
    info = MakePlatformInfo();
    info.l2HbmBWRatio = 1e-7; // 不大于 1e-6
    EXPECT_FALSE(info.isValid());
    info.l2HbmBWRatio = 2e-6;
    EXPECT_TRUE(info.isValid());

    const std::string platformStr = info.ToString();
    EXPECT_NE(platformStr.find("aicNum"), std::string::npos);
    EXPECT_NE(platformStr.find("ubSize"), std::string::npos);
}

TEST(QuantBatchMatmulV4PollingTilingTest, BasicBlockParamToString)
{
    PollingBasicBlockParam param;
    param.mSize = 16;
    param.nSize = 64;
    param.kSize = 512;
    param.mDim = 1;
    param.nDim = 2;
    param.baseM = 16;
    param.baseN = 64;
    const std::string paramStr = param.ToString();
    EXPECT_NE(paramStr.find("mSize"), std::string::npos);
    EXPECT_NE(paramStr.find("baseM"), std::string::npos);
    EXPECT_NE(paramStr.find("mte2Cost"), std::string::npos);
}

TEST(QuantBatchMatmulV4PollingTilingTest, BasicBlockParamCompare)
{
    PollingBasicBlockParam lhs;
    PollingBasicBlockParam rhs;
    // mte2Cost 不同
    lhs.mte2Cost = 1.0;
    rhs.mte2Cost = 2.0;
    EXPECT_TRUE(lhs < rhs);
    EXPECT_FALSE(rhs < lhs);

    // mte2Cost 相同，lhs 存在核内尾块（singleM % baseM != 0）
    lhs = PollingBasicBlockParam();
    rhs = PollingBasicBlockParam();
    lhs.mte2Cost = 10.0;
    rhs.mte2Cost = 10.0;
    lhs.singleM = 100;
    lhs.baseM = 16; // 100 % 16 != 0
    rhs.singleM = 128;
    rhs.baseM = 16;
    EXPECT_FALSE(lhs < rhs);
    // 两侧均有尾块
    rhs.singleM = 180; // 180 % 16 != 0
    EXPECT_FALSE(lhs < rhs);
    EXPECT_FALSE(rhs < lhs);

    // mte2Cost 相同且无尾块，CeilDiv(singleN, baseN) 不同
    lhs.singleM = 128;
    rhs.singleM = 128;
    lhs.singleN = 128;
    lhs.baseN = 64; // CeilDiv = 2
    rhs.singleN = 256;
    rhs.baseN = 64; // CeilDiv = 4
    EXPECT_TRUE(lhs < rhs);
    EXPECT_FALSE(rhs < lhs);

    // CeilDiv 相同，baseN 不同
    rhs.singleN = 256;
    rhs.baseN = 128; // CeilDiv(256, 128) = 2，与 lhs 相同
    EXPECT_TRUE(lhs < rhs);
    EXPECT_FALSE(rhs < lhs);

    // baseN 相同，baseM 大者优先
    rhs.singleN = 128;
    rhs.baseN = 64; // CeilDiv = 2，与 lhs 相同
    lhs.baseM = 32;
    rhs.baseM = 16;
    EXPECT_TRUE(lhs < rhs);
    EXPECT_FALSE(rhs < lhs);
}

// ---------------- Setter ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, SettersUpdateState)
{
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.basicBlockParam_.singleM = 1; // 先污染派生字段，验证 SetShape 会清零
    tiling.basicBlockParam_.singleN = 1;
    tiling.basicBlockParam_.mDim = 1;
    tiling.basicBlockParam_.nDim = 1;
    tiling.basicBlockParam_.baseM = 1;
    tiling.basicBlockParam_.baseN = 1;
    tiling.basicBlockParam_.baseK = 1;

    tiling.SetShape(16, 64, 512, 128);
    const PollingBasicBlockParam& param = tiling.GetTilingResult();
    EXPECT_EQ(param.mSize, 16);
    EXPECT_EQ(param.nSize, 64);
    EXPECT_EQ(param.kSize, 512);
    EXPECT_EQ(param.groupSize, 128);
    EXPECT_EQ(param.singleM, 0);
    EXPECT_EQ(param.singleN, 0);
    EXPECT_EQ(param.mDim, 0);
    EXPECT_EQ(param.nDim, 0);
    EXPECT_EQ(param.baseM, 0);
    EXPECT_EQ(param.baseN, 0);
    EXPECT_EQ(param.baseK, 0);

    tiling.SetDtypeBits(8, 16, 32, 64);
    EXPECT_EQ(param.aDtypeBits, 8);
    EXPECT_EQ(param.bDtypeBits, 16);
    EXPECT_EQ(param.x1ScaleDtypeBits, 32);
    EXPECT_EQ(param.x2ScaleDtypeBits, 64);

    tiling.SetLimitParams(64, 128, 512, 32768);
    EXPECT_EQ(tiling.limitParam_.alignSizeN, 64);
    EXPECT_EQ(tiling.limitParam_.maxBaseMSize, 128);
    EXPECT_EQ(tiling.limitParam_.maxBaseNSize, 512);
    EXPECT_EQ(tiling.limitParam_.maxMNSize, 32768);

    const PlatformInfo platform = MakePlatformInfo(20, 111, 222, 333, 444);
    tiling.SetPlatformInfo(platform);
    EXPECT_EQ(tiling.platformInfos_.aicNum, 20);
    EXPECT_EQ(tiling.platformInfos_.ubSize, 111);
    EXPECT_EQ(tiling.platformInfos_.l1Size, 222);
    EXPECT_EQ(tiling.platformInfos_.l0ASize, 333);
    EXPECT_EQ(tiling.platformInfos_.l0BSize, 444);
}

// ---------------- ComputeTiling 入参校验 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingFailedWithoutShape)
{
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetPlatformInfo(MakePlatformInfo()); // 仅设置平台，形状未初始化
    EXPECT_EQ(tiling.ComputeTiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingFailedWithoutPlatform)
{
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(16, 64, 512, 128);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize); // 平台未初始化
    EXPECT_EQ(tiling.ComputeTiling(), ge::GRAPH_FAILED);
}

// ---------------- ComputeTiling 默认解（无可用分块解） ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingDefaultSolution)
{
    // L0A 过小使所有候选分块均通过不了 CheckBufferSize，走默认解分支
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(200, 300, 512, 64);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo(32, 196352, 524032, 1024, 65536));
    ASSERT_EQ(tiling.ComputeTiling(), ge::GRAPH_SUCCESS);

    const PollingBasicBlockParam& param = tiling.GetTilingResult();
    EXPECT_EQ(param.baseM, 128); // min(128, CeilAlign(200, 16))
    EXPECT_EQ(param.baseN, 128); // min(128, CeilAlign(300, 16))
    EXPECT_EQ(param.baseK, 64);  // min(128, groupSize)
    EXPECT_EQ(param.mDim, 1);
    EXPECT_EQ(param.nDim, 6); // min(CeilDiv(200,128) * CeilDiv(300,128), aicNum) = min(2*3, 32)
    // aL1 + bL1 = 128*64*4 + 128*64*4 = 65536 < 128KB，不满足关闭条件
    EXPECT_TRUE(param.enablePreAlloc);
}

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingDefaultSolutionPreAllocClosed)
{
    // 同样走默认解，但 L1 载入量大且 k 较大，触发关闭 preAlloc
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(128, 128, 2048, 128);
    tiling.SetDtypeBits(16, 16, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo(32, 196352, 524032, 1024, 65536));
    ASSERT_EQ(tiling.ComputeTiling(), ge::GRAPH_SUCCESS);

    const PollingBasicBlockParam& param = tiling.GetTilingResult();
    EXPECT_EQ(param.baseM, 128);
    EXPECT_EQ(param.baseN, 128);
    EXPECT_EQ(param.baseK, 128);
    EXPECT_EQ(param.mDim, 1);
    EXPECT_EQ(param.nDim, 1); // min(1 * 1, 32)
    // aL1 + bL1 = 128*128*4*2 + 128*128*4*2 = 262144 >= 128KB 且 kSize 2048 > 1024
    EXPECT_FALSE(param.enablePreAlloc);
}

// ---------------- ComputeTiling 正常求解 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingSingleSolution)
{
    // aicNum=1 且形状极小，仅存在唯一候选解，可精确校验全部分块结果
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(16, 64, 512, 128);
    tiling.SetDtypeBits(8, 16, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo(1));
    ASSERT_EQ(tiling.ComputeTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.optionalResults_.size(), 1UL);

    const PollingBasicBlockParam& param = tiling.GetTilingResult();
    EXPECT_EQ(param.mDim, 1);
    EXPECT_EQ(param.nDim, 1);
    EXPECT_EQ(param.singleM, 16);
    EXPECT_EQ(param.singleN, 64);
    EXPECT_EQ(param.baseM, 16);
    EXPECT_EQ(param.baseN, 64);
    EXPECT_EQ(param.baseK, 128);
    // freeUb/最小DB占比被 16KB*2 截断：32768/2048=16，FloorDiv(16,4)=4，
    // CeilDiv(kSize/groupSize=4, 4)=1，最终 min(4,1)=1
    EXPECT_EQ(param.scaleFactor, 1);
    // mte2Cost = (16*8*1.8 + 64*16*1.8) / 8 / (1*1/1) = 259.2
    EXPECT_NEAR(param.mte2Cost, 259.2, 1e-6);
    // aL1 + bL1 = 16*128*4 + 64*128*8 = 73728 < 128KB
    EXPECT_TRUE(param.enablePreAlloc);
}

TEST(QuantBatchMatmulV4PollingTilingTest, ComputeTilingGeneralShape)
{
    // 常规大形状：校验求解不变式而非精确值
    QuantBatchMatmulV4PollingTiling tiling;
    InitValidTiling(tiling, 1024, 1024, 4096, 128);
    tiling.SetDtypeBits(8, 16, 32, 32);
    ASSERT_EQ(tiling.ComputeTiling(), ge::GRAPH_SUCCESS);

    const PollingBasicBlockParam& param = tiling.GetTilingResult();
    EXPECT_GE(param.mDim, 1);
    EXPECT_GE(param.nDim, 1);
    EXPECT_LE(param.mDim * param.nDim, 32);
    EXPECT_GE(param.singleM * param.mDim, 1024);
    EXPECT_GE(param.singleN * param.nDim, 1024);
    EXPECT_GE(param.baseM, 16);
    EXPECT_LE(param.baseM, kMaxBaseMSize);
    EXPECT_EQ(param.baseM % 16, 0);
    EXPECT_GE(param.baseN, kAlignSizeN);
    EXPECT_LE(param.baseN, kMaxBaseNSize);
    EXPECT_EQ(param.baseN % kAlignSizeN, 0);
    EXPECT_EQ(param.baseK, 128); // min(128, groupSize)
    EXPECT_GE(param.scaleFactor, 1);
    EXPECT_GT(param.mte2Cost, 0.0);
    // enablePreAlloc 与关闭条件保持一致
    EXPECT_EQ(param.enablePreAlloc, !tiling.ClosePreAllocCond());
}

// ---------------- CalculateBasicBlockTiling 分核剪枝与对齐校验 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, CalculateBasicBlockTilingPruneAndDimFixed)
{
    // m=n=512：首个 (mDim=32, nDim=1) 组合即产生解；mDim=31..26 因 singleM 对齐到 32 后
    // mDimFixed != mDim 被跳过；已有解后乘积小于 0.8*aicNum(25.6) 的组合被剪枝，
    // 因此所有保留解的分核乘积均不小于 26 且不超过 aicNum
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(512, 512, 4096, 128);
    tiling.SetDtypeBits(8, 16, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo());
    tiling.CalculateBasicBlockTiling();

    ASSERT_GE(tiling.optionalResults_.size(), 3UL);
    for (const PollingBasicBlockParam& res : tiling.optionalResults_) {
        // 若 (6,4) 等乘积 24 的可行组合未被剪枝，会推入乘积小于 26 的解
        EXPECT_GE(res.mDim * res.nDim, 26UL);
        EXPECT_LE(res.mDim * res.nDim, 32UL);
        EXPECT_GE(res.singleM * res.mDim, 512);
        EXPECT_GE(res.singleN * res.nDim, 512);
        EXPECT_GT(res.mte2Cost, 0.0);
        EXPECT_EQ(res.baseK, 128);
    }
}

TEST(QuantBatchMatmulV4PollingTilingTest, CalculateBasicBlockTilingDimFixedOnN)
{
    // m=16, n=100：nDim=7..3 因 singleN 对齐到 64 后 nDimFixed 变小被跳过，
    // 仅 nDim=2 产生解；随后 nDim=1 因乘积 1 < 0.8*aicNum 被剪枝
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(16, 100, 512, 128);
    tiling.SetDtypeBits(8, 16, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo());
    tiling.CalculateBasicBlockTiling();

    ASSERT_EQ(tiling.optionalResults_.size(), 1UL);
    const PollingBasicBlockParam& res = tiling.optionalResults_.front();
    EXPECT_EQ(res.mDim, 1);
    EXPECT_EQ(res.nDim, 2);
    EXPECT_EQ(res.singleM, 16);
    EXPECT_EQ(res.singleN, 64);
    EXPECT_EQ(res.baseM, 16);
    EXPECT_EQ(res.baseN, 64);
}

// ---------------- GetBasicBlockSolution 分支 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, GetBasicBlockSolutionMaxMnExceeded)
{
    // maxMNSize=100：CeilAlign(100/baseM, 64) 得到的 baseN 与 baseM 乘积恒大于 100，全部 continue
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(128, 128, 512, 128);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, 100);
    tiling.SetPlatformInfo(MakePlatformInfo());
    tiling.basicBlockParam_.singleM = 128;
    tiling.basicBlockParam_.singleN = 64;
    tiling.basicBlockParam_.mDim = 1;
    tiling.basicBlockParam_.nDim = 1;
    tiling.GetBasicBlockSolution();
    EXPECT_TRUE(tiling.optionalResults_.empty());

    // maxMNSize < baseM：baseNMax 为 0，baseN 循环不进入
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, 8);
    tiling.GetBasicBlockSolution();
    EXPECT_TRUE(tiling.optionalResults_.empty());
}

TEST(QuantBatchMatmulV4PollingTilingTest, GetBasicBlockSolutionBufferCheckContinue)
{
    // l1Size 收紧到 262144：所有 baseN=256 候选因 L1 超限 continue（baseN>=320 则先因 L0B 超限
    // continue），baseM>=96 且 baseN=128 的候选因 UB 超限 continue，其余候选正常产生解
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(128, 512, 512, 128);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo(32, 196352, 262144));
    tiling.basicBlockParam_.singleM = 128;
    tiling.basicBlockParam_.singleN = 512;
    tiling.basicBlockParam_.mDim = 1;
    tiling.basicBlockParam_.nDim = 1;
    tiling.GetBasicBlockSolution();

    ASSERT_GE(tiling.optionalResults_.size(), 3UL);
    uint64_t maxBaseN = 0;
    for (const PollingBasicBlockParam& res : tiling.optionalResults_) {
        EXPECT_LE(res.baseN, 192UL); // baseN=256 全部被 L1 检查拒绝
        EXPECT_EQ(res.baseM % 16, 0UL);
        EXPECT_EQ(res.baseN % kAlignSizeN, 0UL);
        EXPECT_EQ(res.baseK, 128);
        EXPECT_GT(res.mte2Cost, 0.0);
        if (res.baseN == 128UL) {
            EXPECT_LE(res.baseM, 80UL); // baseM>=96 时 UB 检查拒绝 baseN=128
        }
        maxBaseN = std::max(maxBaseN, res.baseN);
    }
    EXPECT_EQ(maxBaseN, 192UL); // (16,192) 等候选通过全部检查
}

// ---------------- GetPollingTilingSolution 排序选优 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, GetPollingTilingSolutionNoSolution)
{
    // 无可用解时返回 false
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(128, 128, 512, 128);
    tiling.SetDtypeBits(8, 8, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, 100); // maxMN 限制掉全部解
    tiling.SetPlatformInfo(MakePlatformInfo());
    EXPECT_FALSE(tiling.GetPollingTilingSolution());
    EXPECT_TRUE(tiling.optionalResults_.empty());
}

TEST(QuantBatchMatmulV4PollingTilingTest, GetPollingTilingSolutionPicksMinCost)
{
    // 多个候选解时排序后取 mte2Cost 最小的解
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetShape(512, 512, 4096, 128);
    tiling.SetDtypeBits(8, 16, 32, 32);
    tiling.SetLimitParams(kAlignSizeN, kMaxBaseMSize, kMaxBaseNSize, kMaxMNSize);
    tiling.SetPlatformInfo(MakePlatformInfo());
    ASSERT_TRUE(tiling.GetPollingTilingSolution());
    ASSERT_GE(tiling.optionalResults_.size(), 2UL);

    double minCost = tiling.optionalResults_.front().mte2Cost;
    for (const PollingBasicBlockParam& res : tiling.optionalResults_) {
        minCost = std::min(minCost, res.mte2Cost);
    }
    EXPECT_DOUBLE_EQ(tiling.GetTilingResult().mte2Cost, minCost);
}

// ---------------- CheckBufferSize 白盒分支 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, CheckBufferSizeBranches)
{
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetPlatformInfo(MakePlatformInfo());
    PollingBasicBlockParam& param = tiling.basicBlockParam_;
    param.kSize = 4096;
    param.groupSize = 128;
    param.baseK = 128;
    param.aDtypeBits = 8;
    param.bDtypeBits = 8;
    param.x1ScaleDtypeBits = 32;
    param.x2ScaleDtypeBits = 32;

    // L0A 双Buffer超限：aL0Db = 128*128*1*2 = 32768 > 1024
    tiling.platformInfos_.l0ASize = 1024;
    EXPECT_FALSE(tiling.CheckBufferSize(128, 64));
    tiling.platformInfos_.l0ASize = 65536;

    // L0B 双Buffer超限：bL0Db = 512*128*1*2 = 131072 > 1024
    tiling.platformInfos_.l0BSize = 1024;
    EXPECT_FALSE(tiling.CheckBufferSize(16, 512));
    tiling.platformInfos_.l0BSize = 65536;

    // L1 超限：4*32768 + 4*131072 = 655360 > 100000
    tiling.platformInfos_.l0BSize = 262144;
    tiling.platformInfos_.l1Size = 100000;
    EXPECT_FALSE(tiling.CheckBufferSize(128, 512));
    tiling.platformInfos_.l0BSize = 65536;
    tiling.platformInfos_.l1Size = 524032;

    // UB 超限：confirmed(295424) + minDbScale(4096) > 196352
    EXPECT_FALSE(tiling.CheckBufferSize(128, 128));

    // 成功：scaleFactor = min(FloorDiv(32768/2048, 4), CeilDiv(4096/128, 4)) = min(4, 8) = 4
    EXPECT_TRUE(tiling.CheckBufferSize(16, 64));
    EXPECT_EQ(param.scaleFactor, 4);

    // 成功：freeUb 缩小使 FloorDiv 结果为 0，被 max 钳位到 1
    tiling.platformInfos_.ubSize = 22592; // freeUb = 22592 - 18496 = 4096，仅 2 倍 minDbScale
    EXPECT_TRUE(tiling.CheckBufferSize(16, 64));
    EXPECT_EQ(param.scaleFactor, 1);
    tiling.platformInfos_.ubSize = 196352;

    // 成功：kSize == groupSize 使 CeilDiv(1, 4) = 1 钳位
    param.kSize = 128;
    EXPECT_TRUE(tiling.CheckBufferSize(16, 64));
    EXPECT_EQ(param.scaleFactor, 1);

    // 成功：kSize = 1024 时 CeilDiv(8, 4) = 2，取 min(4, 2) = 2
    param.kSize = 1024;
    EXPECT_TRUE(tiling.CheckBufferSize(16, 64));
    EXPECT_EQ(param.scaleFactor, 2);
}

// ---------------- GetMte2Cost / BWUtilization 公式 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, Mte2CostAndBwUtilization)
{
    QuantBatchMatmulV4PollingTiling tiling;
    tiling.SetPlatformInfo(MakePlatformInfo()); // aicNum=32，l2HbmBWRatio=1.8
    PollingBasicBlockParam& param = tiling.basicBlockParam_;
    param.mSize = 64;
    param.nSize = 128;
    param.singleM = 48;
    param.singleN = 192;
    param.baseM = 16;
    param.baseN = 64;
    param.mDim = 3;
    param.nDim = 2;
    param.aDtypeBits = 8;
    param.bDtypeBits = 16;

    EXPECT_DOUBLE_EQ(tiling.BWUtilization(4, 8), 1.0);
    EXPECT_DOUBLE_EQ(tiling.BWUtilization(2, 4), 0.25);

    tiling.GetMte2Cost();
    // t1 = 64*(2*CeilDiv(192,64)-1)*8 = 2560, t2 = 64*8*1.8 = 921.6,
    // t3 = 128*(3*CeilDiv(48,16)-1)*16 = 16384, t4 = 128*16*1.8 = 3686.4,
    // mte2Cost = (2560+921.6+16384+3686.4)/8/(6/32) = 2944/0.1875
    EXPECT_NEAR(param.mte2Cost, 47104.0 / 3.0, 1e-6);
}

// ---------------- ClosePreAllocCond 白盒分支 ----------------

TEST(QuantBatchMatmulV4PollingTilingTest, ClosePreAllocCondBranches)
{
    QuantBatchMatmulV4PollingTiling tiling;
    PollingBasicBlockParam& param = tiling.basicBlockParam_;
    param.kSize = 2048;
    param.baseK = 128;
    param.mSize = 256;
    param.nSize = 256;
    param.baseM = 128;
    param.baseN = 128;
    param.aDtypeBits = 16;
    param.bDtypeBits = 16;

    // aL1+bL1 = 131072*2 >= 128KB 且 kSize > 1024
    EXPECT_TRUE(tiling.ClosePreAllocCond());

    // 边界：aL1+bL1 恰好等于 128*1024
    param.baseM = 64;
    param.baseN = 64;
    EXPECT_TRUE(tiling.ClosePreAllocCond());

    // kSize 不大于 1024
    param.kSize = 1024;
    EXPECT_FALSE(tiling.ClosePreAllocCond());

    // L1 载入量低于阈值（kSize 满足）
    param.kSize = 2048;
    param.baseM = 16;
    param.baseN = 16;
    EXPECT_FALSE(tiling.ClosePreAllocCond());
}
