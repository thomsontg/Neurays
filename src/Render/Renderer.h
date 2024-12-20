#pragma once

#include <Scene.h>
#include <misc/Globals.h>
#include <sutil/shared/GLDisplay.h>

class Renderer
{
public:
    Renderer(std::shared_ptr<Globals> gbx);

    void Render();

    void init_glfw();

    void setup_imgui();

    void render_loop(sutil::GLDisplay gl_display);

    void setup_video_frame();

    void render_GUI();

    void run_interactive_render();
    
    void run_cmdline_render();

private:

    // sutil::GLDisplay *gl_display = nullptr;

    std::chrono::time_point<std::chrono::steady_clock> start;
  
    std::chrono::duration<double> state_update_time;
    std::chrono::duration<double> render_time;
    std::chrono::duration<double> display_time;

    std::shared_ptr<Scene> scene;

    std::shared_ptr<Globals> gbx;
};