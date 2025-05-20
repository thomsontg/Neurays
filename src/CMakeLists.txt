#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

OPTIX_add_sample_executable( 
  App target_name
  app.cpp
  Render/Renderer.h
  Render/Renderer.cpp

  sutil/IO/HDRLoader.h
  sutil/IO/HDRLoader.cpp
  sutil/IO/QuatTrackBall.h
  sutil/IO/QuatTrackBall.cpp

  misc/Callbacks.h
  misc/Globals.h
  misc/Globals.cpp
  misc/Utility.h
  misc/Spline.h

  device/shaders.cu

  # device/network/common_host.cu
  # device/network/cpp_api.cu
  # device/network/cutlass_mlp.cu
  # device/network/encoding.cu
  # device/network/fully_fused_mlp.cu
  # device/network/loss.cu
  # device/network/network.cu
  # device/network/object.cu
  # device/network/optimizer.cu
  # device/network/reduce_sum.cu
  
  # gaussian/Gaussian.h
  # gaussian/Gaussian.cpp
  # gaussian/GaussianAverage.h

  OPTIONS -rdc true
  )
  
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Render/Scene)
add_subdirectory(GUI)
add_subdirectory(Render)
# Our sutil library.  The rules to build it are found in the subdirectory.
add_subdirectory(sutil)
add_subdirectory(sh)
add_subdirectory(network)
# add_subdirectory(gaussian)

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib
  scene_lib
  lights_lib
  GUI_lib
  SH_lib
  OpenImageDenoise
  network_lib
  tinyxml2
  # assimp_lib
  # tetgen
  # Gaussian_lib
  )
