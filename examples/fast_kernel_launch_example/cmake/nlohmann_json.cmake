# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_guard(GLOBAL)

if(json_FOUND)
    return()
endif()

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

include(ExternalProject)
include(FindPackageHandleStandardArgs)

set(JSON_ROOT_DIR       ${CMAKE_BINARY_DIR}/third_party/json)
set(JSON_DOWNLOAD_DIR   ${CMAKE_BINARY_DIR}/third_party/download/json)

find_path(JSON_INCLUDE
    NAMES nlohmann/json.hpp
    PATHS ${JSON_ROOT_DIR}/include
    NO_DEFAULT_PATH
)

find_package_handle_standard_args(json
    FOUND_VAR json_FOUND
    REQUIRED_VARS JSON_INCLUDE
)

if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "json found in build dir: ${JSON_INCLUDE}")
    add_custom_target(nlohmann_json)
else()
    message(STATUS "json not found, download to build dir")

    ExternalProject_Add(nlohmann_json
        URL       https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip
        URL_MD5   e2f46211f4cf5285412a63e8164d4ba6

        DOWNLOAD_DIR  ${JSON_DOWNLOAD_DIR}
        SOURCE_DIR    ${JSON_ROOT_DIR}

        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""

        TLS_VERIFY OFF
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
    )
endif()
