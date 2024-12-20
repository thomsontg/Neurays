#pragma once
#include "Renderer.h"

#include <iostream>
#include <GUI/ImGuiTheme.h>
#include <GUI/GUI.h>
#include <misc/Utility.h>
#include <misc/Callbacks.h>

#include <GLFW/glfw3.h>

using namespace std;

/********************************************/
Renderer::Renderer(std::shared_ptr<Globals> _gbx)
    :scene(std::make_shared<Scene>(_gbx.get()))
/********************************************/
{
  callbacks::gbx = _gbx;
  gbx = _gbx;
  state_update_time = std::chrono::duration<double>(0.0);
  render_time = std::chrono::duration<double>(0.0);
  display_time = std::chrono::duration<double>(0.0);
  
  start = std::chrono::steady_clock::now();
    
}

/********************************************/
void Renderer::Render()
/********************************************/
{
  if (!gbx->outfile_selected)
    run_interactive_render();
  else
    run_cmdline_render();

  delete gbx->trackball;
}

/********************************************/
void Renderer::run_interactive_render()
/********************************************/
{
  init_glfw();
  setup_imgui();
  
  Utility::start_data_collection(*gbx);
  
  scene->init_gl_buffer(gbx->rd_width, gbx->rd_height);
  sutil::GLDisplay gl_display;

  // Render loop
  do  render_loop(gl_display);
    while (!glfwWindowShouldClose(gbx->window));

  CUDA_SYNC_CHECK();

  sutil::cleanupUI(gbx->window);
}

/********************************************/
void Renderer::run_cmdline_render()
/********************************************/
{
  if (gbx->output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
  {
    sutil::initGLFW(); // For GL context
    sutil::initGL();
  }

  Utility::updateState(*gbx, *scene);

  cout << "Rendering";
  unsigned int dot = gbx->samples / 20;
  chrono::duration<double> render_time(0.0);
  auto t0 = chrono::steady_clock::now();
  for (unsigned int i = 0; i < gbx->samples; ++i)
  {
    scene->render(scene->get_output_buffer());
    ++gbx->launch_params.subframe_index;
    if ((i + 1) % dot == 0)
    cerr << ".";
  }
  auto t1 = chrono::steady_clock::now();
  render_time = t1 - t0;
  cout << endl << "Time: " << render_time.count() << endl;

  Utility::exportRawImage(*gbx, gbx->outfile);
  Utility::save_Image(*gbx, scene->get_output_buffer(), gbx->outfile);
  if (gbx->output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
  {
    scene->get_output_buffer().deletePBO();
    glfwTerminate();
  }
}

/********************************************/
void Renderer::init_glfw()
/********************************************/
{

  gbx->window = sutil::initUI("Neural Rays", scene->width, scene->height, gbx->custom_title_bar);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glfwSetMouseButtonCallback(gbx->window, callbacks::mouseButtonCallback);
  glfwSetCursorPosCallback(gbx->window, callbacks::cursorPosCallback);
  glfwSetFramebufferSizeCallback(gbx->window, callbacks::windowSizeCallback);
  glfwSetWindowIconifyCallback(gbx->window, callbacks::windowIconifyCallback);
  glfwSetKeyCallback(gbx->window, callbacks::keyCallback);
  glfwSetScrollCallback(gbx->window, callbacks::scrollCallback);
  glfwSetWindowUserPointer(gbx->window, &gbx->launch_params);
  glfwSetWindowPos(gbx->window, 100, 100);
  glfwSetDropCallback(gbx->window, callbacks::windowdropCallback);
  if (gbx->show_panels)
    glfwSetWindowSize(gbx->window, gbx->win_width, gbx->win_height);
  else
    glfwSetWindowSize(gbx->window, gbx->rd_width, gbx->rd_height);

  Utility::load_window_logo(gbx->window);

}

/********************************************/
void Renderer::setup_imgui()
/********************************************/
{
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigDockingWithShift = true;

  // Setup Dear ImGui style
  SetupImGuiStyle();

  GUI::load_GUI_files();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(gbx->window, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // set_cursor(gbx->window);
}

/********************************************/
void Renderer::render_loop(sutil::GLDisplay gl_display)
/********************************************/
{
  setup_video_frame();

  Utility::collect_data(*gbx, *scene);
  
  auto t0 = std::chrono::steady_clock::now();
  glfwPollEvents();

  Utility::updateState(*gbx, *scene);
  auto t1 = std::chrono::steady_clock::now();
  state_update_time += t1 - t0;
  t0 = t1;

  if(gbx->compute_error)
  {
    // std::cout << gbx->error[0]/(gbx->rd_width * gbx->rd_height) << std::endl;
    cudaMemset(gbx->launch_params.error, 0, sizeof(float));
  }

  if (!gbx->pause_render && (gbx->progressive || gbx->launch_params.subframe_index == 0))
  {
    scene->render(scene->get_output_buffer());
  }

  if(gbx->compute_error)
  {
    cudaMemcpy(gbx->error, gbx->launch_params.error, sizeof(float), cudaMemcpyDeviceToHost);
    gbx->error[0] /= gbx->rd_width * gbx->rd_height;
  }

  t1 = std::chrono::steady_clock::now();
  render_time += t1 - t0;
  t0 = t1;

  Utility::display_Frame(scene->get_output_buffer(), gl_display, gbx->window);
  t1 = std::chrono::steady_clock::now();
  display_time += t1 - t0;

  render_GUI();

  glfwSwapBuffers(gbx->window);

  if (!gbx->pause_render && gbx->progressive)
  {
    ++gbx->launch_params.subframe_index;
  }

  if (gbx->save_image)
  {
    Utility::save_Image(*gbx, scene->get_output_buffer(), gbx->outfile);
    gbx->save_image = false;
  }
  if(gbx->save_scene)
  {
    Utility::save_Scene(gbx->outfile, *gbx);
    gbx->save_scene = false;
  }
}

/********************************************/
void Renderer::setup_video_frame()
/********************************************/
{
  if (!gbx->video)
    return;

  if (gbx->launch_params.subframe_index > gbx->video_samples && gbx->step < 1)
  {
    Utility::construct_spline(*gbx, *scene);
  }
  else if (gbx->step >= 1)
  {
    gbx->video = false;
    cout << "<< VIDEO RENDER FINISHED >>" << endl;
  }
}

/********************************************/
void Renderer::render_GUI()
/********************************************/
{

  GUI::beginPanelOverlay(gbx->launch_params, *scene);
  
  if (gbx->first_loop)
  {
    Utility::setup_docking(*gbx);
    gbx->first_loop = false;
  }

  if (gbx->save_as_open)
  {
    gbx->save_image |= GUI::Save_as_modal(*gbx);
  }

  if (gbx->custom_title_bar)
  {
    GUI::mainmenubar(*gbx, *scene);
    GUI::attemptDragWindow(gbx->window);
  }

  if (gbx->stats_on_render)
  {
    sutil::displayStats(state_update_time, render_time, display_time);
  }

  GUI::create_Browser_overlay(*gbx);

  if (gbx->show_panels  && !scene->use_simulator())
  {
    ImGui::Begin("Settings", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    if (!gbx->stats_on_render)
    {
      GUI::createStatisticsOverlay(*gbx, *scene);
    }

    gbx->camera_changed |= GUI::createCameraOverlay(*gbx, *scene);
    gbx->light_changed |= GUI::createLightOverlay(gbx->launch_params, *scene);
    ImGui::End();
    ImGui::Begin("Video");
    gbx->videoset_changed |= GUI::createVideoOverlay(*gbx);
    ImGui::End();

    ImGui::Begin("Materials");
    gbx->material_changed |= GUI::createMaterialOverlay(*gbx, *scene);
    ImGui::End();

    ImGui::Begin("Transform");
    gbx->instance_transformed |= GUI::createTransformOverlay(*gbx, *scene);
    
    ImGui::End();
  }
  GUI::endPanelOverlay();

  Utility::end_data_collection(*gbx);
}