
# InferShape UT：独立编译，不依赖 CANN faker 框架
add_executable(test_apply_adam_v2_infershape test_apply_adam_v2_infershape.cpp)
target_compile_features(test_apply_adam_v2_infershape PRIVATE cxx_std_17)
target_compile_options(test_apply_adam_v2_infershape PRIVATE -Wall -Wextra)
add_test(NAME test_apply_adam_v2_infershape COMMAND test_apply_adam_v2_infershape)

# 递归包含 arch35 子目录
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
