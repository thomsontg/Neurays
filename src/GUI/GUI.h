#pragma once

#include <string.h>
#include <functional>
#include <vector>
#include <map>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imfilebrowser.h>

#include <Render/Scene/Scene.h>
#include <sutil/shared/structs.h>
#include <sutil/sutil.h>
#include <sutil/IO/QuatTrackBall.h>
#include <GUI/ImGui_helpers.h>

#include <GLFW/glfw3.h>

namespace GUI
{

  struct Style
  {
    float ctrlPerc = 0.7f;     // Percentage the value control takes in the pane
    float ctrlMaxSize = 500.f; // Max pixel size control will take
    float ctrlMinSize = 50.f;  // Minimum size of the control
    float dragMinSize = 150.f; // Minimum size to be on one line
  };

  struct Image_Textures
  {
    // Logo variables
    int _width = 0;
    int _height = 0;
    GLuint _texture = 0;
  };

  static Style style;

  ImFont *head;

  ImGui::FileBrowser Env_browser;
  ImGui::FileBrowser Scene_browser;
  ImGui::FileBrowser Reference_browser;

  void load_head_font()
  {
    ImGuiIO &io = ImGui::GetIO();
    std::string fontname = std::string(SOURCE_DIR) + "/src/GUI/fonts/Ginga.ttf";
    head = io.Fonts->AddFontFromFileTTF(fontname.c_str(), 42);
  }

  constexpr auto text = IM_COL32(192, 192, 192, 255);
  constexpr auto textDarker = IM_COL32(128, 128, 128, 255);

  std::string get_shader_from_illum(std::vector<std::string> &shader_list, int &illum)
  {

    if (illum - shader_list.size() >= 2 && illum > 2)
    {
      return shader_list[illum - 3];
    }
    else
    {
      return "lambertian";
    }
  }

  int get_illum_from_shader(std::vector<std::string> &shader_list, std::string &shader)
  {
    auto it = std::find(shader_list.begin(), shader_list.end(), shader);
    if ( it != shader_list.end())
    {
      return it - shader_list.begin() + 3;
    }
    else
    {
      return 0;
    }
  }

  ImU32 ColorWithMultipliedValue(const ImColor &color, float multiplier)
  {
    const ImVec4 &colRow = color.Value;
    float hue, sat, val;
    ImGui::ColorConvertRGBtoHSV(colRow.x, colRow.y, colRow.z, hue, sat, val);
    return ImColor::HSV(hue, sat, std::min(val * multiplier, 1.0f));
  }

  void DrawButtonImage(const GLuint &imageNormal, const GLuint &imageHovered, const GLuint &imagePressed,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,
                       ImVec2 rectMin, ImVec2 rectMax)
  {
    auto *drawList = ImGui::GetForegroundDrawList();
    if (ImGui::IsItemActive())
      ImGui::Image((void *)(intptr_t)imagePressed, ImVec2(35, 35), ImVec2(0, 0), ImVec2(1, 1), ImVec4(tintPressed, tintPressed, tintPressed, tintPressed));
    else if (ImGui::IsItemHovered())
      ImGui::Image((void *)(intptr_t)imagePressed, ImVec2(35, 35), ImVec2(0, 0), ImVec2(1, 1), ImVec4(tintHovered, tintHovered, tintHovered, tintHovered));
    else
      ImGui::Image((void *)(intptr_t)imagePressed, ImVec2(35, 35), ImVec2(0, 0), ImVec2(1, 1), ImVec4(tintNormal, tintNormal, tintNormal, tintNormal));
  };

  // Window buttons
  const ImU32 buttonColN = ColorWithMultipliedValue(text, 0.9f);
  const ImU32 buttonColH = ColorWithMultipliedValue(text, 1.2f);
  const ImU32 buttonColP = textDarker;
  Image_Textures m_logo, m_close, m_minimize, m_maximise, m_iconify;

  double s_xpos = 0, s_ypos = 0;
  int w_xsiz = 0, w_ysiz = 0;
  int dragState = 0;

  void TextCentered(std::string text)
  {
    auto windowWidth = ImGui::GetWindowSize().x;
    auto textWidth = ImGui::CalcTextSize(text.c_str()).x;

    ImGui::SetCursorPosX((windowWidth - textWidth) * 0.5f);
    // ImGui::SetCursorPosY(ImGui::GetCursorPosY()-18);
    ImGui::Text(text.c_str());
  }

  void DrawButtonImage(const GLuint &imageNormal, const GLuint &imageHovered, const GLuint &imagePressed,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,
                       ImRect rectangle)
  {
    DrawButtonImage(imageNormal, imageHovered, imagePressed, tintNormal, tintHovered, tintPressed, rectangle.Min, rectangle.Max);
  };

  void DrawButtonImage(const GLuint &image,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,
                       ImVec2 rectMin, ImVec2 rectMax)
  {
    DrawButtonImage(image, image, image, tintNormal, tintHovered, tintPressed, rectMin, rectMax);
  };

  void DrawButtonImage(const GLuint &image,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,
                       ImRect rectangle)
  {
    DrawButtonImage(image, image, image, tintNormal, tintHovered, tintPressed, rectangle.Min, rectangle.Max);
  };

  void DrawButtonImage(const GLuint &imageNormal, const GLuint &imageHovered, const GLuint &imagePressed,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed)
  {
    DrawButtonImage(imageNormal, imageHovered, imagePressed, tintNormal, tintHovered, tintPressed, ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  };

  void DrawButtonImage(const GLuint &image,
                       ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed)
  {
    DrawButtonImage(image, image, image, tintNormal, tintHovered, tintPressed, ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  };

  bool ToggleButton(const char *str_id, bool &v)
  {
    bool changed = false;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList *draw_list = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f;

    if (changed |= ImGui::InvisibleButton(str_id, ImVec2(width, height)))
      v = !v;
    ImU32 col_bg;
    if (ImGui::IsItemHovered())
      col_bg = v ? IM_COL32(247, 255, 154, 162 + 20) : IM_COL32(230, 50, 75, 162 + 20);
    else
      col_bg = v ? IM_COL32(247, 255, 154, 162) : IM_COL32(230, 50, 75, 162);

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
    draw_list->AddCircleFilled(ImVec2(v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));
    return changed;
  }

  void load_img(std::string filename, Image_Textures &img)
  {
    std::string logo = std::string(SOURCE_DIR) + "/src/assets/" + filename;
    bool ret = sutil::LoadTextureFromFile(logo.c_str(), &(img._texture), &(img._width), &(img._height));
    IM_ASSERT(ret);
  }

  void attemptDragWindow(GLFWwindow *window)
  {
    if (glfwGetMouseButton(window, 0) == GLFW_PRESS && dragState == 0)
    {
      glfwGetCursorPos(window, &s_xpos, &s_ypos);
      glfwGetWindowSize(window, &w_xsiz, &w_ysiz);
      dragState = 1;
    }
    if (glfwGetMouseButton(window, 0) == GLFW_PRESS && dragState == 1)
    {
      double c_xpos, c_ypos;
      int w_xpos, w_ypos;
      glfwGetCursorPos(window, &c_xpos, &c_ypos);
      glfwGetWindowPos(window, &w_xpos, &w_ypos);
      if (
          s_xpos >= 0 && s_xpos <= ((double)w_xsiz - 170) &&
          s_ypos >= 0 && s_ypos <= 25)
      {
        glfwSetWindowPos(window, w_xpos + (c_xpos - s_xpos), w_ypos + (c_ypos - s_ypos));
      }
      if (
          s_xpos >= ((double)w_xsiz - 15) && s_xpos <= ((double)w_xsiz) &&
          s_ypos >= ((double)w_ysiz - 15) && s_ypos <= ((double)w_ysiz))
      {
        glfwSetWindowSize(window, w_xsiz + (c_xpos - s_xpos), w_ysiz + (c_ypos - s_ypos));
      }
    }
    if (glfwGetMouseButton(window, 0) == GLFW_RELEASE && dragState == 1)
    {
      dragState = 0;
    }
  }

  void load_GUI_files()
  {
    // Load Images
    load_img("close_w.png", GUI::m_close);
    load_img("minimise_w.png", GUI::m_minimize);
    load_img("maximise_w.png", GUI::m_maximise);
    load_img("iconify_w.png", GUI::m_iconify);

    // Setup Browser Overlay
    Scene_browser = ImGui::FileBrowser();
    Scene_browser.SetTitle("Load Scene");
    Scene_browser.SetTypeFilters({".xml"});
    Scene_browser.SetWindowSize(400, 300);

    Env_browser = ImGui::FileBrowser();
    Env_browser.SetTitle("Load Environment");
    Env_browser.SetTypeFilters({".hdr", ".png"});
    Env_browser.SetWindowSize(400, 300);

    Reference_browser = ImGui::FileBrowser();
    Reference_browser.SetTitle("Load Reference");
    Reference_browser.SetTypeFilters({".exr"});
    Reference_browser.SetWindowSize(400, 300);

    // Load Fonts
    // load_head_font();
  }

  // Main Menu
  void mainmenubar(Globals &gbx, Scene &scene)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 8));
    if (ImGui::BeginMainMenuBar())
    {
      if (ImGui::BeginMenu("File"))
      {
        if (ImGui::MenuItem("Save as"))
        {
          gbx.save_as_open = true;
        }  

        if(ImGui::MenuItem("Load Scene"))
        {
          Scene_browser.Open();
        }

        if(ImGui::MenuItem("Load Environment"))
        {
          Env_browser.Open();
        }

        if(ImGui::MenuItem("Load Reference"))
        {
          Reference_browser.Open();
        }
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Settings"))
      {
        ImGui::PopStyleVar();

        ToggleButton("Stats on Rendering", (bool &)gbx.stats_on_render);
        ImGui::SameLine();
        ImGui::Text("Stats on Rendering");

        ToggleButton("Pause Rendering", (bool &)gbx.pause_render);
        ImGui::SameLine();
        ImGui::Text("Pause Rendering");

        ToggleButton("Update Bounding Box", (bool &)gbx.update_bbox);
        ImGui::SameLine();
        ImGui::Text("Update Bounding Box");

        ToggleButton("Use Denoiser", (bool &)scene.use_denoiser);
        ImGui::SameLine();
        ImGui::Text("Use Denoiser");

        ImGui::PushItemWidth(130.0f);
        if (ImGui::SliderInt("Max Depth", (int *)&gbx.launch_params.max_depth, 1, 16))
          ;
        ImGui::PopItemWidth();
        ImGui::EndMenu();

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 8));
      }
      ImGui::PushStyleColor(ImGuiCol_Button, {0.15f, 0.15f, 0.15f, 0.00f});
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.15f, 0.15f, 0.15f, 0.62f});
      if (ImGui::Button("Toggle Panel"))
      {
        gbx.show_panels = !gbx.show_panels;
        gbx.show_panel_changed = !gbx.show_panel_changed;
        if (gbx.show_panels)
          glfwSetWindowSize(gbx.window, gbx.win_width, gbx.rd_height);
        else
          glfwSetWindowSize(gbx.window, gbx.rd_width, gbx.rd_height);
      }
      ImGui::PopStyleColor();
      ImGui::PopStyleColor();

      // Close button
      ImGui::SetCursorPosX(ImGui::GetWindowSize().x - 40);

      const int iconWidth = m_close._width;
      const int iconHeight = m_close._height;
      if (ImGui::InvisibleButton("Close", ImVec2(35, 35)))
        glfwSetWindowShouldClose(gbx.window, true);

      ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x - 40, 0));
      DrawButtonImage(m_close._texture, text, ColorWithMultipliedValue(text, 1.4f), buttonColP);

      // Maximise button
      ImGui::SetCursorPosX(ImGui::GetWindowSize().x - 80);
      int max = glfwGetWindowAttrib(gbx.window, GLFW_MAXIMIZED);
      if (ImGui::InvisibleButton("Maximize", ImVec2(35, 35)))
      {
        if (max == GLFW_TRUE)
          glfwRestoreWindow(gbx.window);
        else
          glfwMaximizeWindow(gbx.window);
      };
      ImGui::SetCursorPosX(ImGui::GetWindowSize().x - 80);
      DrawButtonImage(max == GLFW_TRUE ? m_minimize._texture : m_maximise._texture, buttonColN, buttonColH, buttonColP);

      // Iconify
      ImGui::SetCursorPosX(ImGui::GetWindowSize().x - 120);
      int iconified = glfwGetWindowAttrib(gbx.window, GLFW_ICONIFIED);
      if (ImGui::InvisibleButton("Minimize", ImVec2(35, 35)))
      {
        // TODO: move this stuff to a better place, like Window class
        if (iconified == GLFW_TRUE)
          glfwRestoreWindow(gbx.window);
        else
          glfwIconifyWindow(gbx.window);
      }

      ImGui::SetCursorPosX(ImGui::GetWindowSize().x - 120);
      DrawButtonImage(m_iconify._texture, buttonColN, buttonColH, buttonColP);

      // Centered name
      // ImGui::PushFont(head);
      TextCentered("Neural Rays");
      // ImGui::PopFont();

      ImGui::EndMainMenuBar();
      ImGui::PopStyleVar();
    }
  }

  bool Save_as_modal(Globals &gbx)
  {
    bool changed = false;
    ImGui::SetNextWindowSize(ImVec2(300, 145)); // 300, 138 for regular window
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, {0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_TitleBg, {0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, {0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 1.0f});
    ImGui::Begin("Save Menu", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
    ImGui::Text("File Name:");
    static char str0[128];
    strcpy(str0, gbx.outfile.c_str());
    ImGui::SameLine();
    if (ImGui::InputText("##", str0, 128, ImGuiInputTextFlags_CharsNoBlank | ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue))
    {
      changed |= true;
    }
    gbx.outfile = str0;

    ImGui::RadioButton("PNG", (int *)&gbx.file_type,
                       (int)PNG);
    ImGui::SameLine();
    ImGui::RadioButton("PPM", (int *)&gbx.file_type,
                       (int)PPM);
    ImGui::SameLine();
    ImGui::RadioButton("EXR", (int *)&gbx.file_type,
                       (int)EXR);
    ImGui::SameLine();
    ImGui::RadioButton("RAW", (int *)&gbx.file_type,
                       (int)RAW);

    changed |= ImGui::Button("Save");
    if (changed)
    {
      gbx.save_as_open = false;
    }

    ImGui::SameLine();
    if (ImGui::Button("Exit"))
    {
      gbx.save_as_open = false;
    }

    ImGui::SameLine(ImGui::GetWindowWidth()-100);
    if(ImGui::Button("Save Scene"))
    {
      gbx.save_scene = true;
      gbx.save_as_open = false;
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    return changed;
  }

  void show_property_label(const std::string &text, std::string description)
  {

    if (text.back() == '\n')
    {
      ImGui::TextWrapped("%s", text.c_str());
    }
    else
    {
      // Indenting the text to be right justified
      float current_indent = ImGui::GetCursorPos().x;
      const ImGuiStyle &imstyle = ImGui::GetStyle();
      const ImGuiWindow *window = ImGui::GetCurrentWindow();

      float control_width = std::min(
          (ImGui::GetWindowWidth() - imstyle.IndentSpacing) * style.ctrlPerc,
          style.ctrlMaxSize);
      control_width -= window->ScrollbarSizes.x;
      control_width = std::max(control_width, style.ctrlMinSize);

      float available_width = ImGui::GetContentRegionAvail().x;
      float avaiable_text_width =
          available_width - control_width - imstyle.ItemInnerSpacing.x;
      ImVec2 text_size = ImGui::CalcTextSize(
          text.c_str(), text.c_str() + text.size(), false, avaiable_text_width);
      float indent = current_indent + available_width - control_width -
                     text_size.x - imstyle.ItemInnerSpacing.x;

      ImGui::AlignTextToFramePadding();
      ImGui::NewLine();
      ImGui::SameLine(indent);
      ImGui::PushTextWrapPos(indent + avaiable_text_width);
      ImGui::TextWrapped("%s", text.c_str());
      ImGui::PopTextWrapPos();
      ImGui::SameLine();
    }
  }

  bool Custom(const std::string &label, std::string description,
              std::function<bool()> show_content)
  {
    show_property_label(label, description);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);

    bool changed = show_content();
    return changed;
  }

  bool beginPanelOverlay(LaunchParams &launch_params, Scene &scene)
  {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiDockNodeFlags nodeFlags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGui::DockSpaceOverViewport(NULL, nodeFlags);
    return true;
  }

  bool createStatisticsOverlay(Globals &gbx, Scene &scene)
  {
    bool changed = false;
    if (ImGui::CollapsingHeader("Statistics", NULL))
    {
      ImGui::TextWrapped("Frame : %d\nApplication average %.3f ms/frame (%.1f FPS)",
                         gbx.launch_params.subframe_index,
                         1000.0f / ImGui::GetIO().Framerate,
                         ImGui::GetIO().Framerate);
    }

    ImGui::PushItemWidth(38.f);
    ImGui::Text("Render Resolution : ");
    ImGui::SameLine();
    ImGui::DragInt("w", &gbx.rd_width, 1.0f, 500, 5000, NULL, ImGuiSliderFlags_AlwaysClamp);
    changed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine();
    ImGui::Text("X");
    ImGui::SameLine();
    ImGui::DragInt("h", &gbx.rd_height, 1.0f, 500, 5000, NULL, ImGuiSliderFlags_AlwaysClamp);
    changed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::PopItemWidth();
    
    if(gbx.compute_error)
    {
      ImGui::TextWrapped("MSE error : %.6f\n", 
                        gbx.error[0]);
    }

    if(ImGui::Button("Compute Error"))
    {
      if(gbx.reference_name == "")
      {
        Reference_browser.Open();
      }
      else
      {
        gbx.compute_error = !gbx.compute_error;
        if(gbx.compute_error)
        {
          gbx.rd_width = gbx.launch_params.ref_width;
          gbx.rd_height = gbx.launch_params.ref_height;
          gbx.resize_dirty = true;
          gbx.gen_type = GEN_ERROR;
          gbx.error_file.open("errors_list.txt");
        }
        else
        {
          gbx.gen_type = GEN_NONE;
          gbx.error_file.close();
          std::cout << "Error file saved." << std::endl;
        }
      }
    }
    ImGui::SameLine();
    ToggleButton("Pause Rendering", (bool &)gbx.compute_error);


    if(ImGui::Button("Reset"))
    {
      gbx.reset = true;
    }

    int win_width = gbx.rd_width + gbx.panel_width;
    if (changed)
    {
      glfwSetWindowSize(gbx.window, win_width, gbx.rd_height);
    }

    return changed;
  }

  // Camera
  bool createCameraOverlay(Globals &gbx , Scene &scene)
  {
    bool changed = false;
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // TODO express this in up, center and lookat
      (void)Custom("Center", "Center of camera interest", [&]
                   { return ImGui::DragFloat3("Center", &gbx.trackball->get_centre().x, 0.01, NULL, NULL, "%.3f"); });
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      (void)Custom("Rotation", "Rotation", [&]
                   { return ImGui::DragFloat4("Rotation", (float *)&gbx.trackball->get_rotation(),
                                              0.01, NULL, NULL, "%.3f"); });
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      (void)Custom("Eye Dist", "Eye Dist", [&]
                   { return ImGui::DragFloat("Eye Dist", gbx.trackball->get_eye_dist(), 0.01, 0.1, NULL,
                                             "%.5f"); });
      changed |= ImGui::IsItemDeactivatedAfterEdit();
    }
    return changed;
  }

  bool createLightOverlay(LaunchParams &launch_params, Scene &scene)
  {
    bool changed = false;
    if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::Text("Light");
      changed |= ImGui::RadioButton("Simple", (int *)&launch_params.lightType,
                                    (int)LIGHTING_SIMPLE);
      ImGui::SameLine();
      changed |= ImGui::RadioButton("Envmap", (int *)&launch_params.lightType,
                                    (int)LIGHTING_ENVMAP);
      changed |= ImGui::RadioButton("Arealight", (int *)&launch_params.lightType,
                                    (int)LIGTHING_AREALIGHT);
      ImGui::SameLine();
      changed |= ImGui::RadioButton("Sun-Sky", (int *)&launch_params.lightType,
                                    (int)LIGHTING_SUNSKY);

      // if (launch_params.lightType == LIGHTING_SIMPLE) {
      //   for (unsigned int i = 0; i < scene.lights.size(); i++) {
      //     SimpleLight &light = scene.lights[i];
      //     ImGui::Text("Light %d", i);
      //     ImGui::Text("Point");
      //     ImGui::SameLine();
      //     changed |= ToggleButton("Light flip", (bool&)light.type);
      //     ImGui::SameLine();
      //     ImGui::Text("Directional");

      //     if (light.type == DIRECTIONALLIGHT) {
      //       changed |= ImGui::SliderFloat3("Light Direction",
      //                                      (float *)&scene.lights[0].direction.x,
      //                                      -1.f, 1.f);
      //     } else if (light.type == POINTLIGHT) {
      //       changed |= ImGui::SliderFloat3("Light Position",
      //                                      (float *)&scene.lights[0].position.x,
      //                                      -100.f, 100.f);
      //     }
      //     changed |= ImGui::SliderFloat3(
      //         "Light Emission", (float *)&scene.lights[0].emission.x, 0.f, 50.f);
      //   }
      //   if(ImGui::Button("+"))
      //   {
      //     SimpleLight l;
      //     l.direction = make_float3(1.0f);
      //     l.emission = make_float3(1.0f);
      //     l.position = make_float3(1.0f);
      //     l.type = DIRECTIONALLIGHT;
      //     scene.lights.push_back(l);
      //     changed |= true;
      //   }
      //   ImGui::SameLine();
      //   if(ImGui::Button("-") && scene.lights.size() > 0)
      //   {
      //     scene.lights.pop_back();
      //     changed |= true;
      //   }
      // }

      // Miss
      ImGui::Text("Miss");
      changed |= ImGui::RadioButton("Constant", (int *)&launch_params.missType,
                                    (int)MISS_CONSTANT);
      ImGui::SameLine();
      changed |= ImGui::RadioButton("Env", (int *)&launch_params.missType,
                                    (int)MISS_ENVMAP);
      // ImGui::SameLine();
      changed |= ImGui::RadioButton("SunSky", (int *)&launch_params.missType,
                                    (int)MISS_SUNSKY);
      ImGui::SameLine();
      changed |= ImGui::RadioButton("Direction", (int *)&launch_params.missType,
                                    (int)MISS_DIRECTION);

      if (launch_params.missType == MISS_CONSTANT)
      {
        changed |= ImGui::ColorEdit3(
            "Miss color", reinterpret_cast<float *>(&launch_params.miss_color));
      }
      if (launch_params.missType == MISS_ENVMAP ||
          launch_params.lightType == LIGHTING_ENVMAP)
      {
        (void)Custom("Env scale", "Env Scale", [&]
                     { return ImGui::InputFloat("Env Scale ", &launch_params.env_scale, 0.01f,
                                                1.f, "%.2f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();
      }
    }
    return changed;
  }

  bool createMaterialOverlay(Globals &gbx, Scene &scene)
  {
    bool changed = false;
    const auto &meshes = scene.get_meshes();
    const auto &instances = scene.get_instances();
    const std::vector<MtlData> &materials = scene.get_materials();
    const std::vector<std::string> &material_names = scene.get_material_names();
    const std::vector<std::string> all_list = scene.get_all_mtl();
    const std::map<std::string, Medium> media = scene.get_media();
    const std::map<std::string, Interface> interface = scene.get_interfaces();

    ImGuiTreeNodeFlags flags = meshes.size() > 2 ? NULL : ImGuiTreeNodeFlags_DefaultOpen;
    ImGui::BeginChild("Children");
    for (const auto &instance : instances)
    {
      const auto mesh = meshes[instance.get()->mesh_idx];
      std::string meshname = mesh->name.empty() ? "No_file" : mesh->name;
      if (meshname == "No_file")
        continue;
      meshname += " " + std::to_string(instance.get()->instance_id);
      if (ImGui::CollapsingHeader(meshname.c_str(), flags))
      {
        std::string selectedItem = mesh->material_name;
        int mtl_idx = instance->material_id;
        if (ImGui::BeginCombo("Material List", instance->material_name.c_str()))
        {
          for (int i = 0; i < all_list.size(); i++)
          {
            const bool isSelected = (selectedItem == all_list[i]);
            if (ImGui::Selectable(all_list[i].c_str(), isSelected))
            {
              selectedItem = all_list[i];
              instance->material_name = selectedItem;
            }
            changed |= ImGui::IsItemDeactivatedAfterEdit();

            // Set the initial focus when opening the combo
            // (scrolling + keyboard navigation focus)
            if (isSelected)
            {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }
        gbx.materialname = selectedItem;

        ImGui::InputText("Material Name", &instance->material_name[0], 128, ImGuiInputTextFlags_CharsNoBlank | ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        std::string illum = meshname + "_illum";
        (void)Custom("Illum", "Illum", [&]
                     { return ImGui::InputInt(illum.c_str(), &instance->material_illum); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        std::string selected_shader = get_shader_from_illum(scene.get_shaderlist(), instance->material_illum);

        if(ImGui::BeginCombo("Shader", selected_shader.c_str()))
        {
          for(std::string shader : scene.get_shaderlist())
          {            
            const bool isSelected = (selected_shader == shader);
            if(ImGui::Selectable(shader.c_str(), isSelected))
            {
              selected_shader = shader;
              instance->material_illum = get_illum_from_shader(scene.get_shaderlist(), selected_shader);
            }
            
            changed |= ImGui::IsItemDeactivatedAfterEdit();

            // Set the initial focus when opening the combo
            // (scrolling + keyboard navigation focus)
            if (isSelected)
            {
              ImGui::SetItemDefaultFocus();
            }

          }
        
          ImGui::EndCombo();
        }

        MtlData mtl = scene.get_materialdata(selectedItem);
        std::string noise_scale = meshname + "_noise_scale";
        (void)Custom("Noise", "Noise", [&]
                      { return ImGui::InputFloat(noise_scale.c_str(), &mtl.noise_scale, 0.01f,
                                                1.0f, "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        std::string density = meshname + "_density";
        (void)Custom("Density", "Density", [&]
                      { return ImGui::InputFloat(density.c_str(), &mtl.density, 0.01f,
                                                1.0f, "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::Text("Invert Geometry");
        ImGui::SameLine();
        changed |= ToggleButton("density_invert", gbx.launch_params.invert_distances);

        std::string geometry_center = "_geom_center";
        (void)Custom("Center", "Center", [&]
                      { return ImGui::DragFloat3(geometry_center.c_str(), &gbx.launch_params.geom_center.x, 0.01f, -1000.0f,
                                                1000.f, "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

          changed |= ImGui::ColorEdit3("Base color", reinterpret_cast<float *>(&mtl.rho_d));


        std::string mat_scale = meshname + "_material_scale";
        float tmp_scale = mtl.material_scale;
        (void)Custom("Material Scale", "Material Scale", [&]
          { return ImGui::InputFloat(mat_scale.c_str(), &tmp_scale, 0.01f,
              1.0f, "%.3f"); });
        if(ImGui::IsItemDeactivatedAfterEdit())
        {
          mtl.update_scale(tmp_scale);
          changed |= true;
        }

          std::string scattering = meshname + "_scattering";
        (void)Custom("Scattering", "Scattering", [&]
                      { return ImGui::InputFloat3(scattering.c_str(), &mtl.scat.x,
                                                  "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();
        std::string absorption = meshname + "_absorption";
        (void)Custom("Absorption", "Absorption", [&]
                      { return ImGui::InputFloat3(absorption.c_str(), &mtl.absp.x,
                                                  "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();
        std::string asym = meshname + "_asym";
        (void)Custom("Asym", "Asym", [&]
                      { return ImGui::InputFloat3(asym.c_str(), &mtl.asym.x, "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::InputFloat("Shininess", &mtl.shininess, 0.1f);
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        std::string roughness = meshname + "_roughness_d";
        (void)Custom("Roughnes", "Roughness", [&]
                      { return ImGui::InputFloat(roughness.c_str(), &mtl.shininess, 0.01f,
                                                1.0f, "%.3f"); });
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        gbx.mtldata = mtl;
      }
    }
    ImGui::EndChild();
    return changed;
  }

  bool createTransformOverlay(Globals &gbx, Scene &scene)
  {
    bool changed = false;
    const auto &instances = scene.get_instances();
    const auto &meshes = scene.get_meshes();
    const std::vector<std::string> &material_names = scene.get_material_names();

    int idx = 0;
    for (const auto &mesh : meshes)
    {
      ImGui::Text(mesh.get()->name.c_str());
      ImGui::SameLine();
      ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 90);
      if (ImGui::Button("Clear"))
      {
        gbx.mesh_idx = idx;
        gbx.delete_all_instance = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("+"))
      {
        gbx.mesh_idx = idx;
        gbx.add_instance = true;
      }
      idx++;
    }

    ImGuiTreeNodeFlags flags = instances.size() > 2 ? NULL : ImGuiTreeNodeFlags_DefaultOpen;
    ImGui::BeginChild("Children");
    int instance_idx = 0;
    for (const auto &instance : instances)
    {
      const auto &mesh = meshes[instance->mesh_idx];
      std::string meshname = mesh->name;
      if (meshname.empty())
        meshname = "##";
      meshname += " " + std::to_string(instance.get()->instance_id);
      if (ImGui::CollapsingHeader(meshname.c_str(), flags))
      {
        bool geo_changed = false;
        std::string scale_id =
            meshname + "_scale" + std::to_string(instance_idx);
        (void)Custom("Scale", "Scale Instance", [&]
                     { return ImGui::DragFloat3(scale_id.c_str(),
                                                &instance->transforms.scale.x, 0.01f, -1000.0f,
                                                1000.f, "%.3f"); });

        geo_changed |= ImGui::IsItemDeactivatedAfterEdit();
        std::string translate_id =
            meshname + "_translate" + std::to_string(instance_idx);
        (void)Custom("Translate", "Translate Instance", [&]
                     { return ImGui::DragFloat3(translate_id.c_str(),
                                                &instance->transforms.translate.x, 0.01f,
                                                -1000.f, 1000.f, "%.3f"); });
        geo_changed |= ImGui::IsItemDeactivatedAfterEdit();
        std::string rotate_id =
            meshname + "_rotate" + std::to_string(instance_idx);
        (void)Custom("Rotation", "RotationInstance", [&]
                     { return ImGui::DragFloat3(rotate_id.c_str(),
                                                &instance->transforms.rotate.x, 0.01f, -1000.f,
                                                1000.f, "%.3f"); });
        geo_changed |= ImGui::IsItemDeactivatedAfterEdit();
        instance->transforms.dirty = geo_changed;
        changed |= geo_changed;
        if (ImGui::Button("Delete"))
        {
          gbx.mesh_idx = instance_idx;
          gbx.delete_instance = true;
        }
      }
      instance_idx++;
    }
    ImGui::EndChild();

    return changed;
  }

  bool createVideoOverlay(Globals &gbx)
  {
    bool changed = false;

    ImGui::InputFloat("Video Length (in sec)", &gbx.vid_len, 0.01f, 1.0f, "%.3f");

    ImGui::InputInt("Keyframes stored", &gbx.keypoints);
    if (ImGui::IsItemDeactivatedAfterEdit() && gbx.keypoints < 3)
      gbx.keypoints = 3;

    ImGui::InputInt("Samples per frame", &gbx.video_samples);

    ImGui::RadioButton("PNG", (int *)&gbx.file_type,
                       (int)PNG);
    ImGui::SameLine();
    ImGui::RadioButton("PPM", (int *)&gbx.file_type,
                       (int)PPM);
    ImGui::SameLine();
    ImGui::RadioButton("EXR", (int *)&gbx.file_type,
                       (int)EXR);
    ImGui::SameLine();
    ImGui::RadioButton("RAW", (int *)&gbx.file_type,
                       (int)RAW);

    ImGui::Checkbox("Save Video", (bool *)&gbx.video_save);

    changed |= ImGui::Button("Record Video");

    return changed;
  }

  void create_Browser_overlay(Globals &gbx)
  {
    Scene_browser.Display();
    if (Scene_browser.HasSelected())
    {
      gbx.scene_name = Scene_browser.GetSelected().string();
      gbx.scene_dropped = true;
      Scene_browser.ClearSelected();
    }

    Env_browser.Display();
    if (Env_browser.HasSelected())
    {
      std::cout << "Loading Env : " << Env_browser.GetSelected().string() << std::endl;
      gbx.env_name = Env_browser.GetSelected().string();
      gbx.env_dropped = true;
      Env_browser.ClearSelected();
    }

    Reference_browser.Display();
    if (Reference_browser.HasSelected())
    {
      std::cout << "Loading Reference Image : " << Reference_browser.GetSelected().string() << std::endl;
      gbx.reference_name = Reference_browser.GetSelected().string();
      gbx.ref_dropped = true;
      Reference_browser.ClearSelected();
    }
  }

  void endPanelOverlay()
  {
    ImGui::Render();
    ImGui::EndFrame();
    ImGui::UpdatePlatformWindows();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

} // namespace GUI
