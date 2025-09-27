# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

include(ExternalProject)

if (JSON_INCLUDE)
  return()
endif()

ExternalProject_Add(nlohmann_json
  URL                         https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip
  URL_MD5                     e2f46211f4cf5285412a63e8164d4ba6
  DOWNLOAD_DIR                download/nlohmann_json
  PREFIX                      third_party
  TLS_VERIFY                  OFF
  DOWNLOAD_EXTRACT_TIMESTAMP  OFF
  CONFIGURE_COMMAND           ""
  BUILD_COMMAND               ""
  INSTALL_COMMAND             ""
)

ExternalProject_Get_Property(nlohmann_json SOURCE_DIR)

set(JSON_INCLUDE ${SOURCE_DIR}/include)

add_library(json INTERFACE IMPORTED)
set_target_properties(json PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE}")
target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)
add_dependencies(json nlohmann_json)
