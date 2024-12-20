//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <glad/glad.h> // Needs to be included before gl_interop

#include <GLFW/glfw3.h>

// #include <torch/torch.h>
// #include <torch/script.h>
// #include "network.h"

#include "misc/Utility.h"
#include "Render/Renderer.h"


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  // Global variables
  std::shared_ptr<Globals> gbx = std::make_shared<Globals>();

  // Fill in Global values
  try
  {
    if(argc > 1)
    {
      throw std::runtime_error("Loading scene from command line...\n\n");
    }
    Utility::parse_scene_file("../../models/scene/fountain.xml", *gbx);
  }
  catch(const std::exception& e)
  {
    gbx->xml_loading =  false;
    std::cout << e.what() << std::endl;
    Utility::parse_cmdline(argc, argv, *gbx);
  }

  try
  {
    // Initilize Renderer
    Renderer renderer(gbx);

    // Run Renderer
    renderer.Render();
  }
  catch (std::exception &e)
  {
    std::cout << "\n********ERROR LIST********\n\n"; 
    Utility::print_error(e.what(), "error:");
    std::cout << "\n\n********WARNING LIST********\n\n"; 
    Utility::print_error(e.what(), "warning:");
    std::cout << "\n\n********EVERYTHING********\n\n";
    std::cout << "\n\n" << e.what() << "\n\n"; 
    return 1;
  }
  return 0;
}