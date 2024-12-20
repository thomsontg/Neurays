#pragma once

#include <string>

#include <sutil/sutilapi.h>

#include <functional>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>

#include <sutil/math/vec_math.h>

//--------------------------------------------------------------------------------------------------
// Creating a window panel
// - Panel will be on the left or the right side of the window.
// - It fills the height of the window, and stays on the side it was created.
class Panel /*static*/
{
  static ImGuiID dockspaceID;

public:
  // Side where the panel will be
  enum class Side
  {
    Left,
    Right,
  };

  // Starting the panel, equivalent to ImGui::Begin for a window. Need
  // ImGui::end()
  SUTILAPI static void Begin(Side side = Side::Right, float alpha = 0.5f,
                             const char *name = nullptr, float aspect_ratio = 1.5f);

  // Mirror begin but can use directly End()
  SUTILAPI static void End() { ImGui::End(); }

  // Return the position and size of the central display
  SUTILAPI static void CentralDimension(ImVec2 &pos, ImVec2 &size)
  {
    auto dock_main = ImGui::DockBuilderGetCentralNode(dockspaceID);
    if (dock_main)
    {
      pos = dock_main->Pos;
      size = dock_main->Size;
    }
  }
};

enum class Flags
{
  Normal = 0,       // Normal display
  Disabled = 1 << 0 // Grayed out
};

SUTILAPI bool Custom(const std::string &label, std::string description,
                     std::function<bool()> show_content, Flags flags = Flags::Normal);
