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
 * \file test_max_pool_v3_tiling.cpp
 * \brief
 */

 #include <iostream>
 #include <fstream>
 #include <vector>
 #include <gtest/gtest.h>
 
 #include "log/log.h"
 #include "register/op_impl_registry.h"
 #include "array_ops.h"
 #include "ut_op_util.h"
 #include "ut_op_common.h"
 #include "kernel_run_context_facker.h"
 #include "test_cube_util.h"
 #include "exe_graph/runtime/storage_format.h"
 #include "exe_graph/runtime/storage_shape.h"
 #include "platform/platform_infos_def.h"
 #include "tiling/platform/platform_ascendc.h"
 #include "../../../../op_host/arch35/bucketize_v2_tiling.h"

 using namespace ut_util;
 using namespace std;
 using namespace ge;
 
 static string to_string(const std::stringstream& tiling_data)
 {
     auto data = tiling_data.str();
     string result;
     int32_t tmp = 0;
     for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
         memcpy_s(&tmp, sizeof(tmp), data.c_str() + i, sizeof(tmp));
         result += std::to_string(tmp);
         result += " ";
     }
 
     return result;
 }
 
 template <typename T>
 static string to_string(void *buf, size_t size)
 {
     std::string result;
     const T* data = reinterpret_cast<const T*>(buf);
     size_t len = size / sizeof(T);
     for (size_t i = 0; i < len; i++) {
         result += std::to_string(data[i]);
         result += " ";
     }
     return result;
 }
 
 class BucketizeV2Tiling : public testing::Test {
  protected:
   static void SetUpTestCase() {
     std::cout << "BucketizeV2Tiling SetUp" << std::endl;
   }
 
   static void TearDownTestCase() {
     std::cout << "BucketizeV2Tiling TearDown" << std::endl;
   }
 };
 
 static void ExecuteTestCase(gert::StorageShape xShape,
                             gert::StorageShape boundShape, bool out_int32, bool right, ge::DataType xDtype,
                             ge::DataType boundDtype, ge::DataType yDtype, uint64_t except_tilingkey, std::string expect, bool is_failed=false) {
   dlog_setlevel(0, 0, 0);
 
   string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64}
                           })";
   map<string, string> soc_infos;
   map<string, string> aicore_spec;
   map<string, string> intrinsics;
   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
   std::map<std::string, std::string> soc_version_infos = { { "Short_SoC_version", "Ascend950" } };
 
   // platform info
   fe::PlatFormInfos platform_info;
   platform_info.Init();
   // compile info
   optiling::BucketizeV2CompileInfo compile_info;
 
   std::string op_type("BucketizeV2");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
 
   // tilingParseFunc simulate
   auto kernel_holder =
       gert::KernelRunContextFaker()
           .KernelIONum(2, 1)
           .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
           .Outputs({&compile_info})
           .Build();
 
   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
 
   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
 
   // tilingFunc simulate
   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
   ASSERT_NE(param, nullptr);
   auto holder = gert::TilingContextFaker()
                     .SetOpType(op_type)
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&xShape, &boundShape})
                     .OutputShapes({&xShape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, boundDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeAttrs({
                                 {"out_int32", Ops::NN::AnyValue::CreateFrom<bool>(out_int32)},
                                 {"right", Ops::NN::AnyValue::CreateFrom<bool>(right)}
                                 })
                     .TilingData(param.get())
                     .Workspace(ws_size)
                     .Build();
 
   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
   if (!is_failed) {
        // workspaces nullptr return failed
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, except_tilingkey);
        auto tilingData = tiling_context->GetRawTilingData();
        ASSERT_NE(tilingData, nullptr);
        dlog_setlevel(0, 3, 0);
        auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
        EXPECT_EQ(tiling_data_result, expect);
   } else {
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
   }
 }
 
 TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test0) {
   gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
   gert::StorageShape boundShape = {{163}, {163}};

   ge::DataType xDtype = ge::DT_FLOAT;
   ge::DataType boundDtype = ge::DT_FLOAT;
   ge::DataType yDtype = ge::DT_INT64;
   bool right = true;

   bool out_int32 = false;
   uint64_t except_tilingkey = 5;
   std::string expect = "64 6259200 6259200 10148 672 40704 81408 163 9 ";
   ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
      except_tilingkey, expect);
 }

 TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test1) {
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape boundShape = {{163}, {163}};
 
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType boundDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT64;
    bool right = false;
 
    bool out_int32 = true;
    uint64_t except_tilingkey = 1;
    std::string expect = "64 6259200 6259200 10148 672 40704 81408 163 9 ";
    ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
       except_tilingkey, expect);
  }

  TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test2) {
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape boundShape = {{216300}, {216300}};
 
    ge::DataType xDtype = ge::DT_INT32;
    ge::DataType boundDtype = ge::DT_INT32;
    ge::DataType yDtype = ge::DT_INT64;
    bool right = false;
 
    bool out_int32 = false;
    uint64_t except_tilingkey = 0;
    std::string expect = "64 6259200 6259200 5885 23552 23552 47104 47104 216300 5847 37 14 7 ";
    ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
       except_tilingkey, expect);
  }

  TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test3) {
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape boundShape = {{216300}, {216300}};
 
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType boundDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_INT64;
    bool right = true;
 
    bool out_int32 = true;
    uint64_t except_tilingkey = 4;
    std::string expect = "64 6259200 6259200 7045 14112 14336 56384 57344 216300 6979 31 14 6 ";
    ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
       except_tilingkey, expect);
  }
 
  TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test4_FAILED) {
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape boundShape = {{216300, 2}, {216300, 2}};
 
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType boundDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_INT64;
    bool right = true;
 
    bool out_int32 = true;
    uint64_t except_tilingkey = 4;
    bool is_failed = true;
    std::string expect = " ";
    ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
       except_tilingkey, expect, is_failed);
  }
 
  TEST_F(BucketizeV2Tiling, BucketizeV2Tiling_Test5_FAILED) {
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape boundShape = {{216300, }, {216300, }};
 
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType boundDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT64;
    bool right = true;
 
    bool out_int32 = true;
    uint64_t except_tilingkey = 4;
    bool is_failed = true;
    std::string expect = " ";
    ExecuteTestCase(xShape, boundShape, out_int32, right, xDtype, boundDtype, yDtype,
       except_tilingkey, expect, is_failed);
  }
 