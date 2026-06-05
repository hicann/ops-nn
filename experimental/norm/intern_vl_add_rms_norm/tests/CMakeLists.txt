# Copyright (c) 2026 Huawei Technologies Co., Ltd.

if(NOT DEFINED TEST_DTYPE)
    set(TEST_DTYPE bfloat16_t)
endif()

add_compile_definitions(DTYPE_X=${TEST_DTYPE})


file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
