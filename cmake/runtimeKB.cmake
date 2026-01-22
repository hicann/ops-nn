# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK runtime kb to package run #####

set(OPS_RTKB_DIR opp/built-in/data/op)
set(RTKB_FILE_ALL)
foreach(OP_RTKB_CATEGORY ${OP_CATEGORY_LIST})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OP_RTKB_CATEGORY})
        file(GLOB_RECURSE RTKB_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${OP_RTKB_CATEGORY}/*/op_host/config/${compute_unit}/*_runtime_kb.json)
        list(APPEND RTKB_FILE_ALL ${RTKB_FILE})
    endif()
endforeach()

foreach(RTKB_DATA_FILE ${RTKB_FILE_ALL})
    # 获取文件名（不含路径）
    get_filename_component(FILENAME "${RTKB_DATA_FILE}" NAME)

    # 使用正则提取：从开头第一个“_数字_AiCore”之前的部分
    if (FILENAME MATCHES "^([^_]+(_[^_]+)*)_[0-9]+_AiCore_")
        # 提取第一个部分（即SoC）
        set(fileSocPath "${CMAKE_MATCH_1}")
        message(STATUS "Extracted Soc: ${fileSocPath}")
    else()
        message(WARNING "Failed to parse Soc from ${FILENAME}")
        set(fileSocPath "unknown")
    endif()

    install(FILES ${RTKB_DATA_FILE}
        DESTINATION ${OPS_RTKB_DIR}/${fileSocPath}/unified_bank
        OPTIONAL
    )
endforeach()