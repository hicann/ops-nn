# /**
#  * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
#  * and is contributed to the CANN Open Software.
#  *
#  * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
#  * All Rights Reserved.
#  *
#  * Authors (accounts):
#  * - Liang Yanglin <@liang-yanglin>
#  * - Su Tonghua <@sutonghua>
#  *
#  * This program is free software: you can redistribute it and/or modify it.
#  * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#  * You may not use this file except in compliance with the License.
#  * See the LICENSE file at the root of the repository for the full text of the License.
#  *
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
#  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  */

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()