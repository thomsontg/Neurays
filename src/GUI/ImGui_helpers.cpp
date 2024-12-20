
#include <string>

#include <GLFW/glfw3.h>

#include "ImGui_helpers.h"

ImGuiID Panel::dockspaceID{0};

void Panel::Begin(Panel::Side side, float alpha, const char *name, float aspect_ratio)
{
  // Keeping the unique ID of the dock space
  Panel::dockspaceID = ImGui::GetID("DockSpace");
  // The dock need a dummy window covering the entire viewport.
  ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  // All flags to dummy window
  ImGuiWindowFlags host_window_flags = 0;
  host_window_flags |= ImGuiWindowFlags_NoTitleBar |
                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
  host_window_flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking;
  host_window_flags |=
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  host_window_flags |= ImGuiWindowFlags_NoBackground;
  // Starting dummy window
  char label[32];
  ImFormatString(label, IM_ARRAYSIZE(label), "DockSpaceViewport_%08X",
                 viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin(label, NULL, host_window_flags);
  ImGui::PopStyleVar(3);

  // The central node is transparent, so that when UI is draw after, the image
  // is visible Auto Hide Bar, no title of the panel Center is not dockable,
  // that is for the scene
  ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode |
                                      ImGuiDockNodeFlags_AutoHideTabBar |
                                      ImGuiDockNodeFlags_NoDockingInCentralNode;

  // Default panel/window is name setting
  std::string dock_name("Settings");
  if (name != nullptr)
  {
    dock_name = name;
  }

  // Building the splitting of the dock space is done only once
  if (!ImGui::DockBuilderGetNode(dockspaceID))
  {

    ImGui::DockBuilderRemoveNode(dockspaceID);
    ImGui::DockBuilderAddNode(dockspaceID,
                              dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceID, viewport->Size);
    ImGuiID dock_main_id = dockspaceID;

    // Slitting all 4 directions, targetting (320 pixel * DPI) panel width, (180
    // pixel * DPI) panel height.
    const float xRatio = clamp(320.0f / viewport->WorkSize[0], 0.01f, 0.499f);
    const float yRatio = clamp(180.0f / viewport->WorkSize[1], 0.01f, 0.499f);
    ImGuiID id_left, id_right, id_up, id_down;
    // Note, for right, down panels, we use the n / (1 - n) formula to correctly
    // split the space remaining from the left, up panels.
    id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, xRatio,
                                          nullptr, &dock_main_id);
    id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right,
                                           xRatio / (1 - xRatio), nullptr,
                                           &dock_main_id);
    id_up = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, yRatio,
                                        nullptr, &dock_main_id);
    id_down = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down,
                                          yRatio / (1 - yRatio), nullptr,
                                          &dock_main_id);

    ImGui::DockBuilderDockWindow(
        side == Panel::Side::Left ? dock_name.c_str() : "Dock_left", id_left);
    ImGui::DockBuilderDockWindow(side == Panel::Side::Right ? dock_name.c_str()
                                                            : "Dock_right",
                                 id_right);
    ImGui::DockBuilderDockWindow("Dock_up", id_up);
    ImGui::DockBuilderDockWindow("Dock_down", id_down);
    ImGui::DockBuilderDockWindow("Scene", dock_main_id); // Center

    ImGui::DockBuilderFinish(dock_main_id);
  }

  // Setting the panel to blend with alpha
  ImVec4 col = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(col.x, col.y, col.z, alpha));

  ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), dockspaceFlags);
  ImGui::PopStyleColor();
  ImGui::End();

  // The panel
  if (alpha < 1)
    ImGui::SetNextWindowBgAlpha(
        alpha); // For when the panel becomes a floating window
  ImGui::Begin(dock_name.c_str());
}

struct Style
{
  float ctrlPerc = 0.7f;     // Percentage the value control takes in the pane
  float ctrlMaxSize = 500.f; // Max pixel size control will take
  float ctrlMinSize = 50.f;  // Minimum size of the control
  float dragMinSize = 150.f; // Minimum size to be on one line
};

static Style style;

void show_property_label(const std::string &text, std::string description,
                         Flags flags)
{
  if (flags == Flags::Disabled)
  {
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
  }

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

  if (flags == Flags::Disabled)
    ImGui::PopStyleColor();

  /* ImGuiH::tooltip(description, false, 0.0f); */
}

bool Custom(const std::string &label, std::string description, std::function<bool()> show_content, Flags flags)
{
  /* ImGui::PushID(value); */
  show_property_label(label, description, flags);
  if (flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);

  bool changed = show_content();

  if (flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
  /* ImGui::PopID(); */
  return changed;
}
