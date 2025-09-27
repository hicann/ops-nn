/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_tiling_runtime.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_

#include <map>
#include <nlohmann/json.hpp>
#include <vector>
#include <tiling/platform/platform_ascendc.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"

namespace optiling {
namespace avg_pool3_d_tiling_compile_info {
constexpr uint64_t kInvalidTilingId = std::numeric_limits<uint64_t>::max();

constexpr uint32_t kVarBatchN = 0x0001; // conv2d dx, conv3d, conv3d dx

constexpr uint32_t kVarFmapH = 0x0010; // conv2d dw, conv3d
constexpr uint32_t kVarFmapW = 0x0020; // conv2d dw, conv3d
constexpr uint32_t kVarFmapD = 0x0040; // conv2d dw, conv3d
constexpr uint32_t kVarDedyD = 0x0080; // conv3d dx
constexpr uint32_t kVarDedyH = 0x0100; // conv3d dx
constexpr uint32_t kVarDedyW = 0x0200; // conv3d dx

const std::map<std::string, uint32_t> kVar2Flag = {
    {"batch_n", kVarBatchN}, {"fmap_h", kVarFmapH}, {"fmap_w", kVarFmapW}, {"fmap_d", kVarFmapD}};

enum class CubeTilingType : uint64_t
{
    CUBE_DYNAMIC_SHAPE_TILING,
    CUBE_DEFAULT_TILING,
    CUBE_BINARY_TILING,
};

class CubeCompileInfo {
public:
    CubeCompileInfo() = default;
    virtual ~CubeCompileInfo() = default;

    bool AnalyzeCompileInfo(const char* op_name, const char* compile_info_str);
    bool CheckRangeSize(size_t shape_dim_num) const;
    static void GetChipFeature(
        fe::PlatFormInfos& platform_info, const string& lable, const string& platform_info_key,
        const string& true_value, bool& value);
    static void GetLocalMemSize(
        fe::PlatFormInfos& platform_info, const string& lable, const string& mem_type, uint64_t& size);
    static void GetAICoreIntrinsicDtype(fe::PlatFormInfos& platform_info, const string& intrinsic_name, bool& value);
    void ParseRuntimePlatformInfo(const char* op_name, fe::PlatFormInfos& platform_info);
    uint64_t CheckTilingInRepo(const char* op_name, const int64_t* shape, size_t dim_num, bool conv = false) const;
    uint64_t CheckTilingInCostModel(const char* op_name, const int64_t* shape, size_t dim_num) const;
    uint64_t CheckDefaultTiling(const char* op_name, const int64_t* shape, size_t dim_num) const;
    uint64_t CubeTilingBatch(const char* op_name, const int64_t* shape) const;

    virtual bool AnalyzeExtendInfo(const nlohmann::json& compile_info) = 0;
    bool AnalyzeCommonCompileInfo(const nlohmann::json& compile_info);

    bool correct_range_flag = false;
    CubeTilingType tiling_type = CubeTilingType::CUBE_DYNAMIC_SHAPE_TILING;
    uint64_t default_tiling_id = kInvalidTilingId;
    std::vector<int64_t> default_range;
    std::vector<std::vector<int64_t>> repo_seeds;
    std::vector<std::vector<int64_t>> repo_range;
    std::vector<std::vector<int64_t>> cost_range;
    std::vector<std::vector<int64_t>> batch_range; // for dynamic batch
    std::vector<uint64_t> repo_tiling_ids;
    std::vector<uint64_t> cost_tiling_ids;
    std::vector<uint64_t> batch_tiling_ids; // for dynamic batch
    std::map<uint64_t, uint32_t> block_dim;
    std::string soc_version = "";
    platform_ascendc::SocVersion shortSocVersion;

    uint32_t core_num = 0;
    uint64_t ub_size = 0;
    uint64_t l1_size = 0;
    uint64_t l2_size = 0;
    uint64_t l0a_size = 0;
    uint64_t l0b_size = 0;
    uint64_t l0c_size = 0;
    uint64_t bt_size = 0;
    int32_t cube_freq = 0;
    bool load3d_constraints = true;
    bool intrinsic_data_move_l12ub = true;
    bool intrinsic_matmul_ub_to_ub = false;
    bool intrinsic_conv_ub_to_ub = false;
    bool intrinsic_data_move_l0c2ub = true;
    bool intrinsic_fix_pipe_l0c2out = false;
    bool intrinsic_fix_pipe_l0c2ub = false;
    bool intrinsic_data_move_out2l1_nd2nz = false;
    bool intrinsic_data_move_l12bt_bf16 = false;
};

class Conv3DCompileInfo : public CubeCompileInfo {
public:
    Conv3DCompileInfo() = default;
    ~Conv3DCompileInfo() override = default;

    bool AnalyzeExtendInfo(const nlohmann::json& compile_info) override;
    bool repo_binary_flag = false;
    int32_t binary_mode = 1;
    uint32_t var_bit_flags = 0;
    int64_t fmap_c1 = static_cast<int64_t>(-1);
    bool is_ascend_c = false;
    bool is_regbase = false;
    platform_ascendc::SocVersion platform_soc_version;
};

class Conv3DBackPropInputCompileInfo : public CubeCompileInfo {
public:
    Conv3DBackPropInputCompileInfo() = default;
    ~Conv3DBackPropInputCompileInfo() override = default;

    bool AnalyzeExtendInfo(const nlohmann::json& compile_info) override;
    bool repo_binary_flag = false;
    uint32_t var_bit_flags = 0;
    int64_t dedy_c1 = -1;
    int32_t binary_mode = 1;
    int32_t fusion_mode = 0;
    std::string soc_version = "";
    bool is_ascend_c = false;
};

template <class CompileInfo, size_t range_size>
ge::graphStatus ParseCubeCompileInfo(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("nil", "context is null"), return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    OP_CHECK_IF(op_name == nullptr, OP_LOGE("nil", "compute_node_info is null"), return ge::GRAPH_FAILED);
    auto compile_info = context->GetCompiledInfo<CompileInfo>();
    auto json_str = context->GetCompiledJson();
    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OP_CHECK_IF(
        compile_info == nullptr || json_str == nullptr || platform_info == nullptr,
        OP_LOGE(op_name, "compile_info/json/PlatFormInfos is null"), return ge::GRAPH_FAILED);
    compile_info->ParseRuntimePlatformInfo(op_name, *platform_info);
    OP_CHECK_IF(
        !compile_info->AnalyzeCompileInfo(op_name, json_str), OP_LOGE(op_name, "failed to analyze compile info"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !compile_info->CheckRangeSize(range_size), OP_LOGE(op_name, "repo_range/repo_seeds/cost_range invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CubeTiling(
    const int64_t* input_shape, size_t input_shape_dim_num, const gert::Shape& var_value,
    const CubeCompileInfo& compile_info, gert::TilingContext* context);

std::string TensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor);
std::string DebugTilingContext(const gert::TilingContext* context);

ge::graphStatus TilingForConv3D(
    gert::TilingContext* context, const gert::StorageShape* fmap_shape, const gert::StorageShape* out_shape,
    bool check_c1);

ge::graphStatus TilingForConv3DBpInput(gert::TilingContext* context, size_t dedy_idx, bool check_c1);
} // namespace avg_pool3_d_tiling_compile_info
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_