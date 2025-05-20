add_library( scene_lib STATIC
    Photons.h
    Scene.cpp
    Scene.h
    Pipeline.cpp
    Pipeline.h
    Accelerator.cpp
    Accelerator.h
    Denoiser.cpp
    Denoiser.h
)

target_include_directories( scene_lib PRIVATE ${LIB_DIR})
target_include_directories( scene_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(scene_lib 
    ${CUDA_LIBRARIES}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib
    lights_lib
    sutil_7_sdk
    optics_lib
    OpenImageDenoise
    # network_lib
    glad
)