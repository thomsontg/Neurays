#pragma once

#include <vector>
#include "sutil/IO/QuatTrackBall.h"
#include <sutil/spline.h>

struct SplineData
{
  std::vector<double> X;
  // vector<QuatTrackBall> Y;
  std::vector<double> eye_dist;
  std::vector<double> qinc_qv_x;
  std::vector<double> qinc_qv_y;
  std::vector<double> qinc_qv_z;
  std::vector<double> qinc_qw;
  std::vector<double> qrot_qv_x;
  std::vector<double> qrot_qv_y;
  std::vector<double> qrot_qv_z;
  std::vector<double> qrot_qw;

  void append_data(QuatTrackBall &trackball, int &count)
  {
    X.push_back(count);

    eye_dist.push_back(trackball.eye_dist);

    qinc_qv_x.push_back(trackball.qinc.qv.x);
    qinc_qv_y.push_back(trackball.qinc.qv.y);
    qinc_qv_z.push_back(trackball.qinc.qv.z);
    qinc_qw.push_back(trackball.qinc.qw);

    qrot_qv_x.push_back(trackball.qrot.qv.x);
    qrot_qv_y.push_back(trackball.qrot.qv.y);
    qrot_qv_z.push_back(trackball.qrot.qv.z);
    qrot_qw.push_back(trackball.qrot.qw);
  }
};
