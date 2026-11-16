/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_map_index_infershape.cpp
 * \brief
 */
#include <cstdint>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

// 平台桩: 按 soc_info.arch_type 注入(与 group_norm_silu UT 样板一致)
static void SetPlatform(const int32_t npuArch, const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.soc_info.arch_type = npuArch;
    platformInfo.str_info.short_soc_version = soc;
    optiCompilationInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

TEST(MapIndexInfershapeTest, map_index_infershape_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {3};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_invalid_x_dim_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8, 4};
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_not_multiple_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {25};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_level_index_len_mismatch_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {4}; // 24 / 8 = 3，长度不符

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_level_index_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {-1}; // 动态维：不支持动态场景, 统一拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_data_seq_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {-1}; // 动态维：不支持动态场景，显式拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_x_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-1}; // x 为动态维：不支持动态场景，显式拒绝
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_x_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-2}; // x 为未知秩：不支持动态场景，显式拒绝
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_data_seq_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {-2}; // data_seq 为未知秩：不支持动态场景，显式拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_level_index_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {-2}; // level_index 为未知秩：不支持动态场景, 统一拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_x_length_zero_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {0}; // x 长度为 0：拒绝
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_exceed_max_gear_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {2056}; // 2056 / 8 = 257 档 > 256 上限：拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_boundary_max_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // 边界恰好达标：x = 24000（长度上限），档位 = 6144000 / 24000 = 256（档位上限）
    gert::Shape xShape = {24000};
    gert::Shape dataSeqShape = {6144000};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_exceed_max_len_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {24001};         // x 长度 24001 > 24000 上限：拒绝
    gert::Shape dataSeqShape = {6144256}; // 24001 * 256

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 老芯片(910B)上限看护: maxLen=400/maxGear=100 边界恰好达标
TEST(MapIndexInfershapeTest, map_index_infershape_old_chip_boundary_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_2201), "Ascend910B");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {400};
    gert::Shape dataSeqShape = {40000};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// 老芯片(910B) x 长度 401 > 400 上限: 拒绝
TEST(MapIndexInfershapeTest, map_index_infershape_old_chip_exceed_max_len_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_2201), "Ascend910B");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {401};
    gert::Shape dataSeqShape = {40100};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 老芯片(910B) 档位 101 > 100 上限: 拒绝
TEST(MapIndexInfershapeTest, map_index_infershape_old_chip_exceed_max_gear_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_2201), "Ascend910B");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {808};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 9202 属 arch35 家族: 走 24000/256 放宽上限
TEST(MapIndexInfershapeTest, map_index_infershape_9202_boundary_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_9202), "Ascend960PR");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {24000};
    gert::Shape dataSeqShape = {6144000};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// 老芯片(910B)不豁免 unknown level_index: 对齐 canndev else 分支(unknown 长度必不等商, 拒绝)
TEST(MapIndexInfershapeTest, map_index_infershape_old_chip_unknown_level_index_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_2201), "Ascend910B");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {-1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 老芯片(910B)拒绝负数 x 长度: 对齐 canndev else 分支的 x_length <= 0 校验
TEST(MapIndexInfershapeTest, map_index_infershape_old_chip_negative_x_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_2201), "Ascend910B");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-5};
    gert::Shape dataSeqShape = {25};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// arch35 家族同样拒绝负数 x 长度: 统一 <=0 校验(较 canndev 950 分支的 ==0 洞更严)
TEST(MapIndexInfershapeTest, map_index_infershape_3510_negative_x_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-5};
    gert::Shape dataSeqShape = {25};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// data_seq 负长度(恰为 x 整倍数, 可穿过整除/档位检查): 拒绝
TEST(MapIndexInfershapeTest, map_index_infershape_negative_data_seq_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {-16}; // -16 为 8 的整倍数, 负商 -2 不超档位上限

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// data_seq 空张量(长度 0): 拒绝
TEST(MapIndexInfershapeTest, map_index_infershape_zero_data_seq_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {0};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// data_seq 维度非 1: 拒绝
TEST(MapIndexInfershapeTest, map_index_infershape_invalid_data_seq_dim_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_3510), "Ascend950");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24, 12};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 9201 属 arch35 家族: 走 24000/256 放宽上限
TEST(MapIndexInfershapeTest, map_index_infershape_9201_boundary_test)
{
    SetPlatform(static_cast<int32_t>(NpuArch::DAV_9201), "Ascend960DT");
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {24000};
    gert::Shape dataSeqShape = {6144000};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
