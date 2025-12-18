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
 * \file test_cube_util.cpp
 */
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <climits>
#include "test_cube_util.h"

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics) {
  string default_hardward_info = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32}})";
  nlohmann::json compile_info_json = nlohmann::json::parse(compile_info_str);
  if (compile_info_json.type() != nlohmann::json::value_t::object) {
    compile_info_json = nlohmann::json::parse(default_hardward_info.c_str());
  }

  map<string, string> soc_info_keys = {{"ai_core_cnt", "CORE_NUM"}, {"l2_size", "L2_SIZE"}, {"cube_core_cnt", "cube_core_cnt"}, {"vector_core_cnt", "vector_core_cnt"}, {"core_type_list", "core_type_list"}};
  soc_infos["core_type_list"] = "AICore";

  for (auto &t : soc_info_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(t.second)) {
      auto &obj_json = compile_info_json["hardware_info"][t.second];
      if (obj_json.is_number_integer()) {
        soc_infos[t.first] = to_string(compile_info_json["hardware_info"][t.second].get<uint32_t>());
      } else if (obj_json.is_string()) {
        soc_infos[t.first] = obj_json;
      }
    }
  }
  map<string, string> aicore_spec_keys = {{"ub_size", "UB_SIZE"},
                                          {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"},
                                          {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},
                                          {"bt_size", "BT_SIZE"},
                                          {"load3d_constraints", "load3d_constraints"},
                                          {"lut_type", "lut_type"}};
  aicore_spec["cube_freq"] = "cube_freq";
  for (auto &t : aicore_spec_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(t.second)) {
      if (t.second == "load3d_constraints" || t.second == "lut_type") {
        aicore_spec[t.first] = compile_info_json["hardware_info"][t.second].get<string>();
      } else {
        aicore_spec[t.first] = to_string(compile_info_json["hardware_info"][t.second].get<uint32_t>());
      }
    }
  }

  std::string intrinsics_keys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub",
                                   "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                   "Intrinsic_matmul_ub_to_ub", "Intrinsic_conv_ub_to_ub",
                                   "Intrinsic_data_move_l12bt", "Intrinsic_mmad"};
  for (string key : intrinsics_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(key) &&
        compile_info_json["hardware_info"][key].get<bool>()) {
      intrinsics[key] = "float16";
      if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
        intrinsics[key] = "bf16";
      }
      if (key.find("Intrinsic_mmad") != string::npos) {
        intrinsics[key] = "s8s4";
      }
    }
  }
}

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics, map<string, string> &soc_version) {
  GetPlatFormInfos(compile_info_str, soc_infos, aicore_spec, intrinsics);
  nlohmann::json compile_info_json = nlohmann::json::parse(compile_info_str);
  if (compile_info_json.type() != nlohmann::json::value_t::object) {
    return;
  }

  map<string, string> soc_version_keys = {{"Short_SoC_version", "socVersion"}};

  for (auto &t : soc_version_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(t.second)) {
      auto &obj_json = compile_info_json["hardware_info"][t.second];
      if (obj_json.is_number_integer()) {
        soc_version[t.first] = to_string(compile_info_json["hardware_info"][t.second].get<uint32_t>());
      } else if (obj_json.is_string()) {
        soc_version[t.first] = obj_json;
      }
    }
  }
}

std::string GetExeDirPath() {
  std::string exe_path("./");
  char path[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
  if (n > 0) {
    path[n] = '\0';
    exe_path.assign(path);
    auto pos = exe_path.find_last_of('/');
    if (pos != std::string::npos) {
      exe_path.erase(pos + 1);
    } else {
      exe_path.assign("./");
    }
  }

  return exe_path;
}