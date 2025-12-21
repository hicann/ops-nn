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
 * \file avg_pool_cube_tiling.cc
 * \brief
 */
#include "avg_pool_cube_tiling.h"
#include "error_util.h"

#include "graph/utils/type_utils.h"

namespace {
using nlohmann::json;
using std::map;
using std::string;
using std::vector;

constexpr size_t kShapeSize6HD = 6;
constexpr size_t kNDim = 0;
constexpr size_t kDDim = 1;
constexpr size_t kHDim = 3;
constexpr size_t kWDim = 4;

// dynamic batch
const auto kIsShapeInRange1 = [](const int64_t* shape, const vector<int64_t>& range) -> bool {
    return shape[0] >= range[0] && shape[0] <= range[1];
};

// nhw
const auto kIsShapeInRange3 = [](const int64_t* shape, const vector<int64_t>& range) -> bool {
    return shape[0] >= range[0] && shape[0] <= range[1] && shape[1] >= range[2] && shape[1] <= range[3] &&
           shape[2] >= range[4] && shape[2] <= range[5];
};

// ndhw
const auto kIsShapeInRange4 = [](const int64_t* shape, const vector<int64_t>& range) -> bool {
    return shape[0] >= range[0] && shape[0] <= range[1] && shape[1] >= range[2] && shape[1] <= range[3] &&
           shape[2] >= range[4] && shape[2] <= range[5] && shape[3] >= range[6] && shape[3] <= range[7];
};

// skip batch
const auto kCalcDist3 = [](const int64_t* shape, const vector<int64_t>& seeds) -> int64_t {
    return abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]);
};

// skip batch
const auto kCalcDist4 = [](const int64_t* shape, const vector<int64_t>& seeds) -> int64_t {
    return abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]) + abs(shape[3] - seeds[3]);
};
} // namespace

namespace optiling {
namespace avgPool3DTilingCompileInfo {
template <class IsShapeInRange, class CalcDist>
uint64_t MatchRepoTiling(
    const int64_t* shape, const vector<vector<int64_t>>& repo_seeds, const vector<vector<int64_t>>& repo_range,
    const IsShapeInRange& is_shape_in_range, const CalcDist& calc_dist)
{
    int64_t min_distance = std::numeric_limits<int64_t>::max();
    size_t idx = 0;
    size_t tiling_id_idx = kInvalidTilingId;
    while (idx < repo_seeds.size() && idx < repo_range.size()) {
        if (is_shape_in_range(shape, repo_range[idx])) {
            int64_t dist = calc_dist(shape, repo_seeds[idx]);
            if (dist < min_distance) {
                min_distance = dist;
                tiling_id_idx = idx;
            }
        }
        ++idx;
    }

    return tiling_id_idx;
}

template <class IsShapeInRange>
uint64_t MatchCostModelTiling(
    const int64_t* shape, const vector<vector<int64_t>>& cost_range, const vector<uint64_t>& tiling_ids,
    const IsShapeInRange& is_shape_in_range)
{
    for (size_t idx = 0; idx < cost_range.size(); ++idx) {
        if (is_shape_in_range(shape, cost_range[idx])) {
            return tiling_ids[idx];
        }
    }

    return kInvalidTilingId;
}

static CubeTilingType ParseTilingType(const json& compile_info)
{
    if (!compile_info.contains("tiling_type")) {
        return CubeTilingType::CUBE_DYNAMIC_SHAPE_TILING;
    }

    const auto& tiling_type = compile_info["tiling_type"];
    if (tiling_type == "default_tiling") {
        return CubeTilingType::CUBE_DEFAULT_TILING;
    }

    if (tiling_type == "binary") {
        return CubeTilingType::CUBE_BINARY_TILING;
    }

    return CubeTilingType::CUBE_DYNAMIC_SHAPE_TILING;
}

static void GetVarFlagsFromCompileInfo(const nlohmann::json& compile_info, uint32_t& var_bit_flags)
{
    std::vector<std::string> vars = compile_info.at("_vars").begin().value().get<std::vector<std::string>>();
    for (auto it = kVar2Flag.begin(); it != kVar2Flag.end(); ++it) {
        if (std::find(vars.begin(), vars.end(), it->first) != vars.end()) {
            var_bit_flags = var_bit_flags | it->second;
        }
    }
}

bool CubeCompileInfo::AnalyzeCompileInfo(const char* op_name, const char* compile_info_str)
{
    OP_CHECK_IF(compile_info_str == nullptr, OP_LOGE(op_name, "null compile info"), return false);
    OP_LOGD(op_name, "compile info: %s", compile_info_str);
    try {
        auto compile_info = json::parse(compile_info_str);
        if (compile_info.type() == json::value_t::object) {
            OP_CHECK_IF(
                !(AnalyzeExtendInfo(compile_info) && AnalyzeCommonCompileInfo(compile_info)),
                OP_LOGE(op_name, "Parse compile value fail"), return false);
        } else {
            for (size_t idx = 0; idx < compile_info.size(); ++idx) {
                OP_CHECK_IF(
                    !(AnalyzeExtendInfo(compile_info[idx]) && AnalyzeCommonCompileInfo(compile_info[idx])),
                    OP_LOGE(op_name, "Parse compile value fail"), return false);
            }
        }
    } catch (...) {
        OP_LOGE(op_name, "get unknown exception, please check compile info json.");
        return false;
    }

    return true;
}

void CubeCompileInfo::AnalyzeCompileInfoBlockDim(const json& compile_info)
{
    if (compile_info.contains("block_dim")) {
        const auto& tmp_block_dim = compile_info["block_dim"].get<map<string, uint32_t>>();
        bool is_digit = true;
        auto is_digit_func = [&is_digit](char element) {
            is_digit = is_digit && isdigit(element) != 0;
        };
        for (auto it = tmp_block_dim.begin(); it != tmp_block_dim.end(); ++it) {
            is_digit = true;
            std::for_each(it->first.begin(), it->first.end(), is_digit_func);
            if (is_digit) { // filter CORE_NUM in binary mode
                block_dim.insert({stoull(it->first), it->second});
            }
        }
    }
}

bool CubeCompileInfo::AnalyzeCommonCompileInfo(const json& compile_info)
{
    // json structure: {"repo_seeds": {"10114": [..]}, "repo_range": {"10114": [...]}, "cost_range": {"10115": [...]},
    //                  "block_dim": {"10114": 32, "10115": 32}}
    correct_range_flag = compile_info.contains("correct_range_flag") &&
                         compile_info["correct_range_flag"].is_boolean() && compile_info["correct_range_flag"];
    tiling_type = ParseTilingType(compile_info);

    AnalyzeCompileInfoBlockDim(compile_info);

    if (compile_info.contains("repo_seeds") && compile_info.contains("repo_range")) {
        const auto& tmp_repo_seeds = compile_info["repo_seeds"].get<map<string, vector<int64_t>>>();
        const auto& tmp_repo_range = compile_info["repo_range"].get<map<string, vector<int64_t>>>();
        auto repo_seeds_iter = tmp_repo_seeds.begin();
        auto repo_range_iter = tmp_repo_range.begin();

        // maybe in loop, can't reserve space here
        while (repo_seeds_iter != tmp_repo_seeds.end() && repo_range_iter != tmp_repo_range.end()) {
            if (repo_seeds_iter->first != repo_range_iter->first) {
                return false;
            }

            repo_seeds.emplace_back(repo_seeds_iter->second);
            repo_range.emplace_back(repo_range_iter->second);
            repo_tiling_ids.emplace_back(stoull(repo_seeds_iter->first));
            ++repo_seeds_iter;
            ++repo_range_iter;
        }
    }

    if (compile_info.contains("cost_range")) {
        const auto& tmp_cost_range = compile_info["cost_range"].get<map<string, vector<int64_t>>>();
        // maybe in loop, can't reserve space here
        for (auto it = tmp_cost_range.begin(); it != tmp_cost_range.end(); ++it) {
            cost_range.emplace_back(it->second);
            cost_tiling_ids.emplace_back(stoull(it->first));
        }
    }

    // for dynamic batch
    if (compile_info.contains("tiling_range")) {
        const auto& tmp_tiling_range = compile_info["tiling_range"].get<map<string, vector<int64_t>>>();
        // maybe in loop, can't reserve space here
        for (auto it = tmp_tiling_range.begin(); it != tmp_tiling_range.end(); ++it) {
            batch_range.emplace_back(it->second);
            batch_tiling_ids.emplace_back(stoull(it->first));
        }
    }

    if (compile_info.contains("default_range")) {
        const auto& tmp_default_range = compile_info["default_range"].get<map<string, vector<int64_t>>>();
        for (auto it = tmp_default_range.begin(); it != tmp_default_range.end(); ++it) {
            default_range = it->second;
            default_tiling_id = stoull(it->first);
            break;
        }
    }

    return true;
}

void CubeCompileInfo::GetChipFeature(
    fe::PlatFormInfos& platform_info, const string& lable, const string& platform_info_key, const string& true_value,
    bool& value)
{
    std::string temp_str;
    if (platform_info.GetPlatformRes(lable, platform_info_key, temp_str)) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", platform_info_key.c_str(), temp_str.c_str());
        value = (temp_str.find(true_value) != string::npos);
    } else {
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", platform_info_key.c_str());
        value = false;
    }
}

void CubeCompileInfo::GetLocalMemSize(
    fe::PlatFormInfos& platform_info, const string& lable, const string& mem_type, uint64_t& size)
{
    std::string temp_str;
    size = static_cast<uint64_t>(0);
    if (platform_info.GetPlatformRes(lable, mem_type, temp_str)) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", mem_type.c_str(), temp_str.c_str());
        try {
            size = static_cast<uint64_t>(atol(temp_str.c_str()));
        } catch (const std::exception& e) {
            OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "illegal PLATFORM INFO %s: %s", mem_type.c_str(), e.what());
        }
    } else {
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", mem_type.c_str());
    }
}

void CubeCompileInfo::GetAICoreIntrinsicDtype(
    fe::PlatFormInfos& platform_info, const string& intrinsic_name, bool& value)
{
    std::string val;
    (void)platform_info.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsic_name, val);
    if (!val.empty()) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", intrinsic_name.c_str(), val.c_str());
        value = true;
    } else {
        value = false;
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", intrinsic_name.c_str());
    }
}

void CubeCompileInfo::ParseRuntimePlatformInfo(const char* op_name, fe::PlatFormInfos& platform_info)
{
    std::string cube_freq_str;
    core_num = platform_info.GetCoreNum();
    platform_info.GetLocalMemSize(fe::LocalMemType::UB, ub_size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L1, l1_size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L2, l2_size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_A, l0a_size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_B, l0b_size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_C, l0c_size);
    platform_info.GetPlatformRes("AICoreSpec", "cube_freq", cube_freq_str);
    platform_info.GetPlatformRes("version", "SoC_version", soc_version);
    GetLocalMemSize(platform_info, "AICoreSpec", "bt_size", bt_size);
    cube_freq = std::atoi(cube_freq_str.c_str());
    OP_LOGD(
        op_name, "PLATFORM INFO CORE_NUM: %u, UB: %lu, L1: %lu, L2: %lu, L0_A: %lu, L0_B: %lu, L0_C: %lu, BT_SIZE: %lu",
        core_num, ub_size, l1_size, l2_size, l0a_size, l0b_size, l0c_size, bt_size);
    OP_LOGD(op_name, "The platform info of soc version: %s, cube_freq %i", soc_version.c_str(), cube_freq);

    GetChipFeature(
        platform_info, "AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", "bf16", intrinsic_data_move_l12bt_bf16);
    GetChipFeature(platform_info, "AICoreSpec", "load3d_constraints", "1", load3d_constraints);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_l12ub", intrinsic_data_move_l12ub);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_l0c2ub", intrinsic_data_move_l0c2ub);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_fix_pipe_l0c2out", intrinsic_fix_pipe_l0c2out);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_fix_pipe_l0c2ub", intrinsic_fix_pipe_l0c2ub);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_out2l1_nd2nz", intrinsic_data_move_out2l1_nd2nz);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_matmul_ub_to_ub", intrinsic_matmul_ub_to_ub);
    GetAICoreIntrinsicDtype(platform_info, "Intrinsic_conv_ub_to_ub", intrinsic_conv_ub_to_ub);
}

bool CubeCompileInfo::CheckRangeSize(size_t shape_dim_num) const
{
    // for security
    auto check_range = [shape_dim_num](const vector<int64_t>& range) { return range.size() < (shape_dim_num << 1); };
    auto check_seeds = [shape_dim_num](const vector<int64_t>& seeds) { return seeds.size() < shape_dim_num; };
    auto check_batch_range = [](const vector<int64_t>& range) {
        return range.size() < 2;
    }; // range size 2 for dim num 1

    return std::find_if(repo_range.begin(), repo_range.end(), check_range) == repo_range.end() &&
           std::find_if(repo_seeds.begin(), repo_seeds.end(), check_seeds) == repo_seeds.end() &&
           std::find_if(cost_range.begin(), cost_range.end(), check_range) == cost_range.end() &&
           std::find_if(batch_range.begin(), batch_range.end(), check_batch_range) == batch_range.end();
}

uint64_t CubeCompileInfo::CheckTilingInRepo(const char* op_name, const int64_t* shape, size_t dim_num) const
{
    uint64_t tiling_id_idx = kInvalidTilingId;
    switch (dim_num) {
        case 3: { // shape size 3
            // nhw
            tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange3, kCalcDist3);
            break;
        }
        case 4: { // shape size 4
            // ndhw
            tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange4, kCalcDist4);
            break;
        }
        default:
            OP_LOGW(op_name, "not support check dim num %zu in repo", dim_num);
            break;
    }

    uint64_t tiling_id = (tiling_id_idx != kInvalidTilingId) ? repo_tiling_ids[tiling_id_idx] : kInvalidTilingId;
    OP_LOGD(op_name, "get tiling_id %lu from repo range/seeds", tiling_id);
    return tiling_id;
}

uint64_t CubeCompileInfo::CheckTilingInCostModel(const char* op_name, const int64_t* shape, size_t dim_num) const
{
    uint64_t tiling_id = kInvalidTilingId;
    switch (dim_num) {
        case 3: { // shape size 3
            // nhw
            tiling_id = MatchCostModelTiling(shape, cost_range, cost_tiling_ids, kIsShapeInRange3);
            break;
        }
        case 4: { // shape size 4
            // ndhw
            tiling_id = MatchCostModelTiling(shape, cost_range, cost_tiling_ids, kIsShapeInRange4);
            break;
        }
        default:
            OP_LOGW(op_name, "not support check dim num %zu in cost range", dim_num);
            break;
    }

    OP_LOGD(op_name, "get tiling_id %lu from cost_range", tiling_id);
    return tiling_id;
}

uint64_t CubeCompileInfo::CheckDefaultTiling(const char* op_name, const int64_t* shape, size_t dim_num) const
{
    uint64_t tiling_id = kInvalidTilingId;
    switch (dim_num) {
        case 1:
            // dynamic batch
            if (kIsShapeInRange1(shape, default_range)) {
                tiling_id = default_tiling_id;
            }
            break;
        case 3: // shape size 3
            // nhw
            if (kIsShapeInRange3(shape, default_range)) {
                tiling_id = default_tiling_id;
            }
            break;
        case 4: // shape size 4
            // ndhw
            if (kIsShapeInRange4(shape, default_range)) {
                tiling_id = default_tiling_id;
            }
            break;
        default:
            OP_LOGW(op_name, "not support check dim num %zu in default tiling", dim_num);
            break;
    }

    OP_LOGD(op_name, "get tiling_id %lu from default tiling", tiling_id);
    return tiling_id;
}

uint64_t CubeCompileInfo::CubeTilingBatch(const char* op_name, const int64_t* shape) const
{
    uint64_t tiling_id = kInvalidTilingId;
    for (size_t idx = 0; idx < batch_range.size(); ++idx) {
        if (kIsShapeInRange1(shape, batch_range[idx])) {
            tiling_id = batch_tiling_ids[idx];
        }
    }

    OP_LOGD(op_name, "get tiling_id %lu for dynamic batch", tiling_id);
    return tiling_id;
}

bool AvgPool3DCubeCompileInfo::AnalyzeExtendInfo(const json& compile_info)
{
    if (compile_info.contains("fmap_c1")) {
        fmap_c1 = compile_info["fmap_c1"];
    }

    if (compile_info.contains("tiling_type") && compile_info["tiling_type"] == "binary") {
        repo_binary_flag = true;
        if (compile_info.contains("binary_mode")) {
            binary_mode = compile_info["binary_mode"];
        }
    } else {
        GetVarFlagsFromCompileInfo(compile_info, var_bit_flags);
    }

    return true;
}

bool AvgPool3DGradCubeCompileInfo::AnalyzeExtendInfo(const json& compile_info)
{
    if (compile_info.contains("dedy_c1")) {
        dedy_c1 = compile_info["dedy_c1"];
    }

    if (compile_info.contains("fusion_mode")) {
        fusion_mode = compile_info["fusion_mode"];
    }

    if (compile_info.contains("tiling_type") &&
        (compile_info["tiling_type"] == "binary" || compile_info["tiling_type"] == "static")) {
        repo_binary_flag = true;
        if (compile_info.contains("binary_mode")) {
            binary_mode = compile_info["binary_mode"];
        }
    } else {
        GetVarFlagsFromCompileInfo(compile_info, var_bit_flags);
    }

    return true;
}

ge::graphStatus CubeTiling(
    const int64_t* input_shape, size_t intput_shape_dim_num, const gert::Shape& var_value,
    const CubeCompileInfo& compile_info, gert::TilingContext* context)
{
    const char* op_name = context->GetNodeName();
    uint64_t tiling_id = kInvalidTilingId;

    if (compile_info.tiling_type == CubeTilingType::CUBE_DEFAULT_TILING) {
        tiling_id = compile_info.CheckDefaultTiling(op_name, input_shape, intput_shape_dim_num);
    } else if (var_value.GetDimNum() != 1) {
        tiling_id = compile_info.CheckTilingInRepo(op_name, input_shape, intput_shape_dim_num);
        if (tiling_id == kInvalidTilingId) {
            tiling_id = compile_info.CheckTilingInCostModel(op_name, input_shape, intput_shape_dim_num);
        }
    } else {
        tiling_id = compile_info.CubeTilingBatch(op_name, input_shape);
    }

    if (tiling_id == kInvalidTilingId) {
        if (compile_info.correct_range_flag) {
            OP_LOGE(
                op_name,
                "The original range does not meet requirements,"
                "new range is generated during op compile, but the shape is not covered by new range");
        }

        OP_LOGE(op_name, "This shape is not covered by any tiling, please modify range and recompile");
        return ge::GRAPH_FAILED;
    }

    auto it = compile_info.block_dim.find(tiling_id);
    OP_CHECK_IF(
        it == compile_info.block_dim.end(), OP_LOGE(op_name, "failed to get block dim for tiling id %lu", tiling_id),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(it->second);
    context->SetTilingKey(tiling_id);
    auto tiling_data = context->GetRawTilingData();
    for (size_t idx = 0; idx < var_value.GetDimNum(); ++idx) {
        tiling_data->Append(static_cast<int32_t>(var_value.GetDim(idx)));
    }

    return ge::GRAPH_SUCCESS;
}

string TensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << Shape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << Shape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
               .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string DebugTilingContext(const gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << TensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << TensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

static size_t InitVarsValuesGetIdx(const gert::Shape& in_shape, gert::Shape& var_value, int64_t* shape_for_range_match)
{
    size_t idx = 0;
    shape_for_range_match[idx++] = in_shape.GetDim(kNDim);
    if (var_value.GetDimNum() > 1) { // not only dynamic batch
        shape_for_range_match[idx++] = in_shape.GetDim(kDDim);
        shape_for_range_match[idx++] = in_shape.GetDim(kHDim);
        shape_for_range_match[idx++] = in_shape.GetDim(kWDim);
    }
    return idx;
}

static size_t InitVarsValuesForAvgPool3DGradCube(
    uint32_t var_bit_flags, const gert::Shape& in_shape, const gert::Shape& out_shape, gert::Shape& var_value,
    int64_t* shape_for_range_match)
{
    if ((var_bit_flags & kVarBatchN) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(out_shape.GetDim(kNDim));
    }

    if ((var_bit_flags & kVarDedyD) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kDDim));
        var_value.AppendDim(out_shape.GetDim(kDDim));
    }

    if ((var_bit_flags & kVarDedyH) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kHDim));
        var_value.AppendDim(out_shape.GetDim(kHDim));
    }

    if ((var_bit_flags & kVarDedyW) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kWDim));
        var_value.AppendDim(out_shape.GetDim(kWDim));
    }

    return InitVarsValuesGetIdx(in_shape, var_value, shape_for_range_match);
}

static size_t InitVarsValuesForAvgPool3DCube(
    uint32_t var_bit_flags, const gert::Shape& in_shape, const gert::Shape& out_shape, gert::Shape& var_value,
    int64_t* shape_for_range_match)
{
    if ((var_bit_flags & kVarBatchN) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(out_shape.GetDim(kNDim));
    }

    if ((var_bit_flags & kVarFmapD) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kDDim));
        var_value.AppendDim(out_shape.GetDim(kDDim));
    }

    if ((var_bit_flags & kVarFmapH) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kHDim));
        var_value.AppendDim(out_shape.GetDim(kHDim));
    }

    if ((var_bit_flags & kVarFmapW) != static_cast<uint32_t>(0)) {
        var_value.AppendDim(in_shape.GetDim(kWDim));
        var_value.AppendDim(out_shape.GetDim(kWDim));
    }

    return InitVarsValuesGetIdx(in_shape, var_value, shape_for_range_match);
}

ge::graphStatus Tiling4AvgPool3DCube(
    gert::TilingContext* context, const gert::StorageShape* fmap_shape, const gert::StorageShape* out_shape)
{
    const auto op_name = context->GetNodeName();
    const auto compile_info = reinterpret_cast<const AvgPool3DCubeCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(compile_info == nullptr, OP_LOGE(op_name, "compile_info is null"), return ge::GRAPH_FAILED);

    OP_LOGD(op_name, "%s", DebugTilingContext(context).c_str());

    OP_CHECK_IF(
        fmap_shape == nullptr || out_shape == nullptr, OP_LOGE(op_name, "failed to get fmap/out shape"),
        return ge::GRAPH_FAILED);

    const auto& fmap_storage_shape = fmap_shape->GetStorageShape();
    const auto& out_storage_shape = out_shape->GetStorageShape();
    OP_CHECK_IF(
        (fmap_storage_shape.GetDimNum() < kShapeSize6HD) || (out_storage_shape.GetDimNum() < kShapeSize6HD),
        OP_LOGE(op_name, "invalid fmap/out dim size"), return ge::GRAPH_FAILED);

    gert::Shape var_value;
    int64_t shape_for_range_match[4]; // 4: ndhw
    size_t dim_num = InitVarsValuesForAvgPool3DCube(
        compile_info->var_bit_flags, fmap_storage_shape, out_storage_shape, var_value, shape_for_range_match);
    return CubeTiling(shape_for_range_match, dim_num, var_value, *compile_info, context);
}

ge::graphStatus TilingForAvgPool3dGrad(gert::TilingContext* context, size_t dedy_idx)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("nil", "context is null"), return ge::GRAPH_FAILED);
    const auto op_name = context->GetNodeName();
    const auto compile_info = reinterpret_cast<const AvgPool3DGradCubeCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(
        compile_info == nullptr, CUBE_INNER_ERR_REPORT(op_name, "compile_info is null"), return ge::GRAPH_FAILED);

    OP_LOGD(op_name, "%s", DebugTilingContext(context).c_str());

    const auto dedy_shape = context->GetInputShape(dedy_idx);
    const auto fmap_shape = context->GetOutputShape(0);

    OP_CHECK_IF(
        fmap_shape == nullptr || dedy_shape == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "failed to get input/output shape"), return ge::GRAPH_FAILED);

    const auto& fmap_storage_shape = fmap_shape->GetStorageShape();
    const auto& dedy_storage_shape = dedy_shape->GetStorageShape();
    OP_CHECK_IF(
        (fmap_storage_shape.GetDimNum() < kShapeSize6HD) || (dedy_storage_shape.GetDimNum() < kShapeSize6HD),
        CUBE_INNER_ERR_REPORT(op_name, "invalid fmap/out_backprop dim size"), return ge::GRAPH_FAILED);

    gert::Shape var_value;
    int64_t shape_for_range_match[4]; // 4: ndhw
    size_t dim_num = InitVarsValuesForAvgPool3DGradCube(
        compile_info->var_bit_flags, fmap_storage_shape, dedy_storage_shape, var_value, shape_for_range_match);
    return CubeTiling(shape_for_range_match, dim_num, var_value, *compile_info, context);
}

} // namespace avgPool3DTilingCompileInfo
} // namespace optiling
