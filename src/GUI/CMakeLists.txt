add_library(GUI_lib SHARED
  GUI.h
  GUI.cpp
  ImGui_helpers.h
  ImGui_helpers.cpp
  ImGuiTheme.h
)


target_include_directories(GUI_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(GUI_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(GUI_lib PUBLIC ${LIB_DIR})
target_link_libraries(GUI_lib imgui sutil_7_sdk)
