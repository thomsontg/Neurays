
# cmake_policy(SET CMP0048 NEW)
# project(materials_lib VERSION 1.0.0 DESCRIPTION "Material Library")

add_library(lights_lib STATIC
  Lights.cpp
  Lights.h
  Envmap.h
  Envmap.cpp
  SunSky.h
  SunSky.cpp
)

# set_target_properties(materials_lib PROPERTIES VERSION ${PROJECT_VERSION})

target_include_directories(lights_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(lights_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(lights_lib PRIVATE ${LIB_DIR})
target_link_libraries(lights_lib ${CUDA_LIBRARIES} scene_lib)
