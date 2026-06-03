# arch22
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93 kirinx90 kirin9030
    AUTO_SYNC false
)

# arch35
add_kernel_sources(
    KERNEL_SRC arch35/foreach_sub_scalar_list.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
