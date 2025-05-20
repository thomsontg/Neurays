add_library(network_lib STATIC
    EvalNetwork.cpp
    EvalNetwork.h
    KernelNetwork.h
)

target_link_libraries(network_lib PRIVATE sutil_7_sdk)