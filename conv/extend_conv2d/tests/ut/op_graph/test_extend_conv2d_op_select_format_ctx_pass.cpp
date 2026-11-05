/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv2d_op_select_format_ctx_pass.cpp
 * \brief ExtendConv2D OpSelectFormat UT：用手工伪造的 gert::OpCheckContext 覆盖"从context判定c04条件"
 *        路径，即 CheckC04HwSplitModeL1Valid 与入口的判定分支（本文件与
 *        test_extend_conv2d_op_select_format_pass.cpp 互补，后者只测拿到判定结果之后的JSON拼装）。
 *        OpCheckContext 构造私有、仅GE可建，UT按其真实内存布局伪造：
 *        values[0]=Node*(占位)、values[1]=输入计数、values[2]=输出计数、values[3+i]=第i个输入的
 *        StorageShape，input_size=3+输入数（布局依据 libregister 的 OpCheckContext::GetInputShape
 *        实现，读 values[3+index]）；compute_node_info 用公开 OpKernelContextBuilder 构建，
 *        属性按 ExtendConv2D IR 顺序追加：strides/pads/dilations(ListInt)、groups(Int)、data_format(String)。
 *        GetInputShape/GetAttrs/GetInputDesc 均走库的真实实现，本文件只准备数据，不打桩。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/types.h"
#include "base/context_builder/op_kernel_run_context_builder.h"

// OpSelectFormat的入口与helper均为内部链接，UT直接include源文件以驱动被测代码（同
// test_extend_conv2d_op_select_format_pass.cpp 的做法；IMPL_OP注册展开为static+counter后缀变量，
// 重复包含安全）
#include "../../../op_graph/extend_conv2d_op_graph.cpp"

using namespace ge;
using Json = nlohmann::json;

namespace {
// ExtendConv2D IR：10个输入、2个输出；OpCheckContext真实布局为values前3槽(Node*/输入计数/输出计数)
// 后再排输入shape，本用例只用到x(0)与filter(1)
constexpr size_t kIrInputs = 10;
constexpr size_t kIrOutputs = 2;
constexpr size_t kCtxHeaderSlots = 3;

std::vector<std::string> SplitComma(const std::string& str)
{
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= str.size()) {
        size_t pos = str.find(',', begin);
        if (pos == std::string::npos) {
            result.push_back(str.substr(begin));
            break;
        }
        result.push_back(str.substr(begin, pos - begin));
        begin = pos + 1;
    }
    return result;
}

std::vector<std::string> GetField(const Json& json, const std::string& classify, const std::string& field)
{
    EXPECT_TRUE(json.contains(classify)) << classify;
    if (!json.contains(classify) || !json[classify].contains(field)) {
        return {};
    }
    return SplitComma(json[classify][field].get<std::string>());
}

gert::Shape MakeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

// 伪造OpCheckContext：compute_node_info(属性/输入desc)走公开builder，输入shape按真实布局挂Chain，
// 各开关按用例裁剪以驱动不同分支（属性按IR顺序截断即模拟"属性缺失"的真实语义）
struct OpCheckContextMocker {
    struct Cfg {
        ge::DataType xDtype = ge::DT_FLOAT16;
        ge::Format xFormat = ge::FORMAT_NCHW;
        std::vector<int64_t> xDims = {1, 4, 56, 56};
        ge::DataType wDtype = ge::DT_FLOAT16;
        ge::Format wFormat = ge::FORMAT_NCHW;
        std::vector<int64_t> wDims = {32, 4, 3, 3};
        std::vector<int64_t> strides = {1, 1, 1, 1};   // attr 0
        std::vector<int64_t> pads = {0, 0, 0, 0};      // attr 1
        std::vector<int64_t> dilations = {1, 1, 1, 1}; // attr 2
        int64_t groups = 1;                            // attr 3
        std::string dataFormat = "NCHW";               // attr 4
        size_t attrCount = 5;     // 少于5时按序截断，未追加的属性index按真实语义取到nullptr
        bool withXDesc = true;    // 不设x的desc，GetInputDesc(0)返回nullptr
        bool withWDesc = true;    // 不设filter的desc，GetInputDesc(1)返回nullptr
        bool withXShape = true;   // 不挂x的shape链，GetInputShape(0)返回nullptr
        bool withWShape = true;   // 不挂filter的shape链，GetInputShape(1)返回nullptr
        bool withNodeInfo = true; // 整体不构建compute_node_info，desc/attrs全为nullptr
    };

    gert::OpKernelContextBuilder builder;
    gert::ContextHolder<gert::KernelContext> holder;
    gert::StorageShape xShape;
    gert::StorageShape filterShape;
    std::vector<uint8_t> buffer;
    std::vector<std::unique_ptr<gert::Chain>> chains;
    int64_t inputCount = static_cast<int64_t>(kIrInputs);
    int64_t outputCount = static_cast<int64_t>(kIrOutputs);
    gert::OpCheckContext* ctx = nullptr;

    explicit OpCheckContextMocker(const Cfg& cfg)
    {
        const void* nodeInfo = nullptr;
        if (cfg.withNodeInfo) {
            builder.OpType("ExtendConv2D").OpName("ExtendConv2D_ut");
            builder.IOInstanceNum(std::vector<uint32_t>(kIrInputs, 1), std::vector<uint32_t>(kIrOutputs, 1));
            if (cfg.withXDesc) {
                builder.InputTensorDesc(0, cfg.xDtype, cfg.xFormat, cfg.xFormat);
            }
            if (cfg.withWDesc) {
                builder.InputTensorDesc(1, cfg.wDtype, cfg.wFormat, cfg.wFormat);
            }
            if (cfg.attrCount > 0) {
                builder.AppendAttr(cfg.strides); // 0: strides
            }
            if (cfg.attrCount > 1) {
                builder.AppendAttr(cfg.pads); // 1: pads
            }
            if (cfg.attrCount > 2) {
                builder.AppendAttr(cfg.dilations); // 2: dilations
            }
            if (cfg.attrCount > 3) {
                builder.AppendAttr(cfg.groups); // 3: groups
            }
            if (cfg.attrCount > 4) {
                builder.AppendAttr(ge::AscendString(cfg.dataFormat.c_str())); // 4: data_format
            }
            holder = builder.Build();
            nodeInfo = holder.GetContext()->GetContext()->compute_node_info;
        }
        xShape.MutableOriginShape() = MakeShape(cfg.xDims);
        filterShape.MutableOriginShape() = MakeShape(cfg.wDims);

        const size_t totalValues = kCtxHeaderSlots + kIrInputs + kIrOutputs;
        buffer.assign(sizeof(KernelRunContext) + (totalValues - 1) * sizeof(void*), 0);
        auto* krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = kCtxHeaderSlots + kIrInputs; // GetInputShape边界：index+3 < input_size
        krc->output_size = kIrOutputs;
        krc->compute_node_info = nodeInfo;
        krc->kernel_extend_info = nullptr;
        krc->output_start = nullptr;
        krc->values[0] = AsChain(&outputCount); // 占位Node*：仅要求非空
        krc->values[1] = AsChain(&inputCount);  // 输入计数
        krc->values[2] = AsChain(&outputCount); // 输出计数
        krc->values[kCtxHeaderSlots] = cfg.withXShape ? AsChain(&xShape) : nullptr;
        krc->values[kCtxHeaderSlots + 1] = cfg.withWShape ? AsChain(&filterShape) : nullptr;
        // 其余输入(shape0/relu_weight0等)的shape不被OpSelectFormat读取，保持空槽
        ctx = reinterpret_cast<gert::OpCheckContext*>(buffer.data());
    }

    ~OpCheckContextMocker() = default; // unique_ptr自动释放Chain

    AsyncAnyValue* AsChain(void* data)
    {
        auto chain = std::make_unique<gert::Chain>();
        chain->Set(data, nullptr);
        chains.push_back(std::move(chain));
        return reinterpret_cast<AsyncAnyValue*>(chains.back().get());
    }
};

// 走入口真实分发，按filter下发的format判定落在哪个分支
std::string RunSelectFormat(const OpCheckContextMocker& mocker)
{
    ge::AscendString result;
    EXPECT_EQ(ops::ExtendConv2dOpSelectFormat(mocker.ctx, result), ge::GRAPH_SUCCESS);
    const char* jsonStr = result.GetString();
    EXPECT_NE(jsonStr, nullptr);
    return jsonStr == nullptr ? std::string() : std::string(jsonStr);
}

void ExpectFilterFormat(const std::string& json, const std::string& expectFormat)
{
    Json parsed = json.empty() ? Json() : Json::parse(json);
    auto formats = GetField(parsed, "input1", "format");
    EXPECT_FALSE(formats.empty());
    for (const auto& format : formats) {
        EXPECT_EQ(format, expectFormat) << "filter format: " << format;
    }
}
} // namespace

class ExtendConv2dOpSelectFormatCtxTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// c04判定的正例：小shape、fp16、groups=1，L1容量判定通过，下发FRACTAL_Z_C04
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_nchw)
{
    OpCheckContextMocker mocker{OpCheckContextMocker::Cfg{}};
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// NHWC布局的c04正例：覆盖入口的NHWC下标分支(cIndex=3/wIndex=2)与CheckC04的NHWC下标计算
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_nhwc)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xFormat = ge::FORMAT_NHWC;
    cfg.xDims = {1, 56, 56, 4}; // NHWC: C=dim[3]=4, W=dim[2]=56
    cfg.wFormat = ge::FORMAT_NHWC;
    cfg.wDims = {3, 3, 4, 32}; // NHWC: kH=dim[1]=3, kW=dim[2]=3
    cfg.dataFormat = "NHWC";
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// fmapC>4不走c04
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_c_gt4)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 5, 56, 56};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// groups!=1不走c04
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_groups_not_one)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.groups = 2;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// fmapW超过32767不走c04
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_w_over_limit)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 4, 56, 32768};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// weight为int8不走c04
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_weight_int8)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.wDtype = ge::DT_INT8;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// L1容量判不过：fmapUsedL1Size超过1M上限
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_l1_overflow)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 4, 1, 400000}; // wIn=400000，fp16时fmapUsedL1Size约3.2M
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// x的desc缺失：GetInputDesc(0)为nullptr，直接跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_x_desc_null)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.withXDesc = false;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// x的shape缺失：GetInputShape(0)为nullptr，跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_x_shape_null)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.withXShape = false;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// compute_node_info整体缺失：desc/attrs全为nullptr，跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_no_node_info)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.withNodeInfo = false;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// groups属性缺失：GetInt(3)为nullptr，跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_groups_attr_missing)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.attrCount = 3; // 只追加strides/pads/dilations
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// fmap维度数不足(wIndex=3)：dim_num=2，跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_dims_too_small)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 4};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// fmapC非正：fmapC=0，跳过c04判定
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_c_not_positive)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 0, 56, 56};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// data_format属性缺失：CheckC04HwSplitModeL1Valid取GetStr(4)为nullptr返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_data_format_missing)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.attrCount = 4; // 缺attr4(data_format)
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// data_format中找不到H/W：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_data_format_no_hw)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.dataFormat = "ND";
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// strides尺寸不足(NCHW下posH=2)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_strides_too_small)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.strides = {1, 1};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// pads尺寸不足(<4)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_pads_too_small)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.pads = {0, 0};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// dilations尺寸不足：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_dilations_too_small)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.dilations = {1, 1};
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// weight的desc缺失：CheckC04HwSplitModeL1Valid入口判weightDesc为nullptr返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_weight_desc_null)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.withWDesc = false;
    cfg.withWShape = false;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// weight的shape缺失：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_weight_shape_null)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.withWShape = false;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// fmap原始格式无H/W(ND)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_ori_format_no_hw)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xFormat = ge::FORMAT_ND;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// fmap原始格式含H/W的位置超过shape维度数(NCDHW取dim[3]/dim[4]但只有4维)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_fmap_dim_lt_ori_format_pos)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xFormat = ge::FORMAT_NCDHW; // fmapPosH=3, fmapPosW=4
    cfg.xDims = {1, 4, 56, 56};     // dim_num=4 <= 4
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// weight原始格式无H/W(ND)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_weight_ori_format_no_hw)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.wFormat = ge::FORMAT_ND;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// weight原始格式含H/W的位置超过shape维度数：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_weight_dim_lt_ori_format_pos)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.wFormat = ge::FORMAT_NCDHW; // weightPosH=3, weightPosW=4
    cfg.wDims = {32, 4, 3, 3};      // dim_num=4 <= 4
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// wOut<=0(wIn小于膨胀后核宽)：返回false
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_false_wout_le_zero)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDims = {1, 4, 3, 4};  // wIn=dim[3]=4
    cfg.wDims = {32, 4, 6, 6}; // wK=dim[3]=6 > wIn+pad，wOut=(4-6)/1+1=-1
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z");
}

// dtype分支：x为int8(GetCubeMkn的int8档与GetBitRatio的1字节档)，weight仍fp16，c04成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_x_dtype_int8)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDtype = ge::DT_INT8;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// dtype分支：x为uint64(GetBitRatio的8字节档、GetCubeMkn的default档)，c04仍成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_x_dtype_uint64)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDtype = ge::DT_UINT64;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// dtype分支：weight为int4(GetBitRatio的1字节档且非int8)，c04成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_weight_dtype_int4)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.wDtype = ge::DT_INT4;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// dtype分支：x为float(GetBitRatio的4字节档)，c04仍成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_x_dtype_float)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDtype = ge::DT_FLOAT;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// dtype分支：x为double(GetBitRatio的DT_DOUBLE档)，c04仍成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_x_dtype_double)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDtype = ge::DT_DOUBLE;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}

// dtype分支：x为bool(GetBitRatio的default档按1字节)，c04仍成立
TEST_F(ExtendConv2dOpSelectFormatCtxTest, c04_true_x_dtype_bool)
{
    OpCheckContextMocker::Cfg cfg;
    cfg.xDtype = ge::DT_BOOL;
    OpCheckContextMocker mocker(cfg);
    ExpectFilterFormat(RunSelectFormat(mocker), "FRACTAL_Z_C04");
}
