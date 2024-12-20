#pragma once

#include <iostream>
#include <imgui/imgui.h>

inline void SetupImGuiStyle()
{
	ImGuiIO &io = ImGui::GetIO();
	std::string fontname = std::string(SOURCE_DIR) + "/src/GUI/fonts/SourceSansPro.ttf";
	io.Fonts->AddFontFromFileTTF(fontname.c_str(), 18);
	auto &style = ImGui::GetStyle();

	style.Alpha = 1.0f;
	style.DisabledAlpha = 0.6000000238418579f;
	style.WindowPadding = ImVec2(13.89999961853027f, 11.10000038146973f);
	style.WindowRounding = 0.0f;
	style.WindowBorderSize = 1.0f;
	style.WindowMinSize = ImVec2(20.0f, 32.0f);
	style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
	style.WindowMenuButtonPosition = ImGuiDir_None;
	style.ChildRounding = 11.10000038146973f;
	style.ChildBorderSize = 1.0f;
	style.PopupRounding = 11.10000038146973f;
	style.PopupBorderSize = 1.0f;
	style.FramePadding = ImVec2(7.400000095367432f, 5.199999809265137f);
	style.FrameRounding = 8.0f;
	style.FrameBorderSize = 0.0f;
	style.ItemSpacing = ImVec2(2.900000095367432f, 4.599999904632568f);
	style.ItemInnerSpacing = ImVec2(2.0f, 6.300000190734863f);
	style.CellPadding = ImVec2(6.099999904632568f, 3.099999904632568f);
	style.IndentSpacing = 11.10000038146973f;
	style.ColumnsMinSpacing = 0.0f;
	style.ScrollbarSize = 20.0f;
	style.ScrollbarRounding = 11.10000038146973f;
	style.GrabMinSize = 11.10000038146973f;
	style.GrabRounding = 11.10000038146973f;
	style.TabRounding = 4.099999904632568f;
	style.TabBorderSize = 1.0f;
	style.TabMinWidthForCloseButton = 11.10000038146973f;
	style.ColorButtonPosition = ImGuiDir_Right;
	style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
	style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

	style.Colors[ImGuiCol_Text] = ImVec4(0.9490196108818054f, 0.8901960849761963f, 0.7764706015586853f, 1.0f);
	style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.7553647756576538f, 0.704525351524353f, 0.60623699426651f, 1.0f);
	style.Colors[ImGuiCol_WindowBg] = ImVec4(0.2078431397676468f, 0.2117647081613541f, 0.2039215713739395f, 1.0f);
	style.Colors[ImGuiCol_ChildBg] = ImVec4(0.2603841722011566f, 0.266094446182251f, 0.2546740770339966f, 1.0f);
	style.Colors[ImGuiCol_PopupBg] = ImVec4(0.4034335017204285f, 0.3774614036083221f, 0.3272486329078674f, 0.6266094446182251f);
	style.Colors[ImGuiCol_Border] = ImVec4(9.999999974752427e-07f, 0.0f, 0.0f, 1.0f);
	style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.8798283338546753f, 0.0f, 0.0f, 0.0f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.5278970003128052f);
	style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_TitleBg] = ImVec4(0.7553647756576538f, 0.704525351524353f, 0.60623699426651f, 1.0f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.3562231659889221f, 0.3186964988708496f, 0.2461456209421158f, 1.0f);
	style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.3562231659889221f, 0.3186964988708496f, 0.2461456209421158f, 0.5236051082611084f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.1764705926179886f, 0.2117647081613541f, 0.2039215713739395f, 1.0f);
	style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.01960784383118153f, 0.01960784383118153f, 0.01960784383118153f, 0.1673820018768311f);
	style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.3133047223091125f, 0.3133015930652618f, 0.3133015930652618f, 1.0f);
	style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.407843142747879f, 0.407843142747879f, 0.407843142747879f, 1.0f);
	style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.5098039507865906f, 0.5098039507865906f, 0.5098039507865906f, 1.0f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.9678018093109131f, 1.0f, 0.6051502227783203f, 0.6351931095123291f);
	style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.9678018093109131f, 1.0f, 0.6051502227783203f, 0.6351931095123291f);
	style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.9678018093109131f, 1.0f, 0.6051502227783203f, 0.6351931095123291f);
	style.Colors[ImGuiCol_Button] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.5278970003128052f);
	style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_Header] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.5278970003128052f);
	style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_Separator] = ImVec4(0.0f, 1.430511474609375e-05f, 1.0f, 0.0f);
	style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.09803921729326248f, 0.4000000059604645f, 0.7490196228027344f, 0.0f);
	style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.09803921729326248f, 0.4000000059604645f, 0.7490196228027344f, 0.0f);
	style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.2078431397676468f, 0.3607843220233917f, 0.4901960790157318f, 1.0f);
	style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.4235294163227081f, 0.3568627536296844f, 0.4823529422283173f, 1.0f);
	style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.7529411911964417f, 0.4235294163227081f, 0.5176470875740051f, 1.0f);
	style.Colors[ImGuiCol_Tab] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_TabHovered] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.5278970003128052f);
	style.Colors[ImGuiCol_TabActive] = ImVec4(0.54935622215271f, 0.07780580967664719f, 0.1432993412017822f, 0.6351931095123291f);
	style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.06666667014360428f, 0.1019607856869698f, 0.1450980454683304f, 0.9724000096321106f);
	style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.9725490212440491f, 0.6941176652908325f, 0.5843137502670288f, 1.0f);
	style.Colors[ImGuiCol_PlotLines] = ImVec4(0.9490196108818054f, 0.8901960849761963f, 0.7764706015586853f, 1.0f);
	style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 0.4274509847164154f, 0.3490196168422699f, 1.0f);
	style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.3347639441490173f, 0.1379284858703613f, 0.1987532824277878f, 1.0f);
	style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 0.6000000238418579f, 0.0f, 1.0f);
	style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.1882352977991104f, 0.1882352977991104f, 0.2000000029802322f, 1.0f);
	style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3490196168422699f, 1.0f);
	style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.2274509817361832f, 0.2274509817361832f, 0.2470588237047195f, 1.0f);
	style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.05999999865889549f);
	style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 0.3499999940395355f);
	style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 0.0f, 0.8999999761581421f);
	style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.51f, 0.65f, 0.60f, 1.00f);
	style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.9019607901573181f, 0.196078434586525f, 0.294117659330368f, 0.6351931095123291f);
	style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
	style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.11f, 0.13f, 0.13f, 0.35f);
}
