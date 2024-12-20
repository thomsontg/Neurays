#pragma once

#include <iostream>

#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <misc/Utility.h>

#include <sutil/sutil.h>
#include <sutil/IO/QuatTrackBall.h>
#include <sutil/shared/structs.h>

namespace callbacks
{
  std::shared_ptr<Globals> gbx;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  // Ignore mouse events if hovering over ImGui
  if (ImGui::GetCurrentContext() != nullptr &&
      ImGui::GetIO().WantCaptureMouse)
  {
    return;
  }
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  if (action == GLFW_PRESS)
  {
    gbx->mouse_button = button;
    switch (button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
      gbx->trackball->grab_ball(ORBIT_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      gbx->trackball->grab_ball(DOLLY_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      gbx->trackball->grab_ball(PAN_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    }
  }
  else
  {
    gbx->trackball->release_ball();
    gbx->mouse_button = -1;
  }
}

static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
  if (gbx->mouse_button >= 0)
  {
    gbx->trackball->roll_ball(make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
    gbx->camera_changed = true;
  }
}

static void windowSizeCallback(GLFWwindow *window, int32_t res_x, int32_t res_y)
{
  // Keep rendering at the current resolution when the window is minimized.
  if (gbx->minimized)
    return;

  // Output dimensions must be at least 1 in both x and y.
  sutil::ensureMinimumSize(res_x, res_y);
  if (gbx->show_panels)
  {
    gbx->rd_width = res_x - gbx->panel_width;
    gbx->rd_height = res_y;
    gbx->win_width = res_x;
    gbx->win_height = res_y;
  }
  else
  {
    gbx->rd_width = res_x;
    gbx->rd_height = res_y;
    gbx->win_width = res_x + gbx->panel_width;
    gbx->win_height = res_y;
  }

  if (gbx->show_panel_changed)
  {
    gbx->show_panel_changed = false;
  }
  else
  {
    gbx->camera_changed = true;
    gbx->resize_dirty = true;
  }
}

static void windowIconifyCallback(GLFWwindow *window, int32_t iconified)
{
  gbx->minimized = (iconified > 0);
}

static void windowdropCallback(GLFWwindow *window, int count, const char **paths)
{
  for (int i = 0; i < count; i++)
  {
    std::string filename = paths[i];
    std::string file_extension;
    size_t idx = filename.find_last_of('.');
    if (idx < filename.length())
    {
      file_extension = filename.substr(idx, filename.length() - idx);
      Utility::lower_case_string(file_extension);
    }
    if (file_extension == ".obj")
    {
      gbx->dropped_files.push_back(filename);
      gbx->obj_dropped = true;
    }
    else if (file_extension == ".hdr" || file_extension == ".png")
    {
      gbx->env_name = filename;
      std::cout << "Loading : " << filename << std::endl;
      gbx->env_dropped = true;
    }
    else
    {
      printf("File extension for: %s is not supported yet. \n", paths[i]);
    }
  }
}

static void keyCallback(GLFWwindow *window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureKeyboard)
    return;

  if (action == GLFW_PRESS || action == GLFW_REPEAT)
  {
    switch (key)
    {
    case GLFW_KEY_Q: // Quit the program using <Q>
                     // case GLFW_KEY_ESCAPE: // Quit the program using <esc>
      glfwSetWindowShouldClose(window, true);
      break;
    case GLFW_KEY_S: // Save the rendered image using <S>
      if (action == GLFW_PRESS)
        gbx->save_as_open = !gbx->save_as_open;
      break;
    case GLFW_KEY_R: // Toggle progressive rendering using <R>
      if (action == GLFW_PRESS)
      {
        std::cout << "Samples per pixel: " << gbx->launch_params.subframe_index << std::endl;
        gbx->progressive = !gbx->progressive;
        std::cout << "Progressive sampling is " << (gbx->progressive ? "on." : "off.") << std::endl;
      }
      break;
    case GLFW_KEY_Z: // Zoom using 'z' or 'Z'
    {
      int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
      int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
      gbx->cam_const *= rshift || lshift ? 1.05f : 1.0f / 1.05f;
      gbx->camera_changed = true;
      std::cout << "Vertical field of view: " << atanf(gbx->cam_const) * 360.0f * M_1_PIf << std::endl;
      break;
    }
    case GLFW_KEY_O: // Save current view to a file called view using <O> (output)
      if (action == GLFW_PRESS)
        gbx->save_view("view");
      break;
    case GLFW_KEY_V: // Save current view to a file with view number using <V> (output)
      if (action == GLFW_PRESS)
        gbx->save_view(std::to_string(gbx->view));
      gbx->view++;
      break;
    case GLFW_KEY_I: // Load current view from a file called view using <I> (input)
      if (action == GLFW_PRESS)
        gbx->load_view("view");
      break;
    case GLFW_KEY_H: // Change the time of the day in half-hour steps using 'h' or 'H'
    {
      int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
      int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
      gbx->time_of_day += rshift || lshift ? -0.5f : 0.5f;
      gbx->sky_changed = true;
      std::cout << "The solar time is now: " << gbx->time_of_day << std::endl;
      break;
    }
    case GLFW_KEY_C: // Change the cloud cover using 'c' or 'C'
    {
      int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
      int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
      gbx->clouds += rshift || lshift ? -0.05f : 0.05f;
      gbx->clouds = clamp(gbx->clouds, 0.0f, 1.0f);
      gbx->sky_changed = true;
      std::cout << "The cloud cover is now: " << gbx->clouds * 100.0f << "%" << std::endl;
      break;
    }
    case GLFW_KEY_E: // Export scene data as xml using 'e'
    {
      if (action == GLFW_PRESS)
      {
        Utility::save_Scene("temp", *gbx);
      }
      break;
    }
    case GLFW_KEY_P:
    {
      if (action == GLFW_PRESS)
      {
        gbx->show_panels = !gbx->show_panels;
        gbx->show_panel_changed = !gbx->show_panel_changed;
        if (gbx->show_panels)
          glfwSetWindowSize(window, gbx->win_width, gbx->rd_height);
        else
          glfwSetWindowSize(window, gbx->rd_width, gbx->rd_height);
      }
      break;
    }
    case GLFW_KEY_KP_ADD: // Increment the angle of incidence using '+'
    {
      gbx->theta_i = fminf(gbx->theta_i + 1.0f, 90.0f);
      std::cout << "Angle of incidence: " << static_cast<int>(gbx->theta_i) << std::endl;
      gbx->light_changed = true;
      break;
    }
    case GLFW_KEY_KP_SUBTRACT: // Decrement the angle of incidence using '-'
    {
      gbx->theta_i = fmaxf(gbx->theta_i - 1.0f, -90.0f);
      std::cout << "Angle of incidence: " << static_cast<int>(gbx->theta_i) << std::endl;
      gbx->light_changed = true;
      break;
    }
    }
  }
}

static void scrollCallback(GLFWwindow *window, double xscroll, double yscroll)
{
  // Ignore mouse events if hovering over ImGui
  if (ImGui::GetCurrentContext() != nullptr &&
      ImGui::GetIO().WantCaptureMouse)
  {
    return;
  }
  gbx->trackball->eye_dist += yscroll * 10;
  gbx->trackball->eye_dist = fmaxf(gbx->trackball->eye_dist, 0.01f);
  gbx->camera_changed = true;
}

} // namespace callbacks
