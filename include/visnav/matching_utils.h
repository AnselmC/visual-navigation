/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <memory>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>
#include <visnav/ex1.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();
  E = visnav::helpers::get_skew_symmetric_matrix(t_0_1.normalized()) * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    const Eigen::Vector3d x_o_l = cam1->unproject(p0_2d);
    const Eigen::Vector3d x_o_r = cam2->unproject(p1_2d);
    double epipolar_constraint = x_o_l.transpose() * E * x_o_r;
    if (std::abs(epipolar_constraint) < epipolar_error_threshold) {
      md.inliers.push_back(md.matches[j]);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  opengv::bearingVectors_t bearingVectors0, bearingVectors1;
  for (size_t i = 0; i < md.matches.size(); i++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[i].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[i].second];
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);
    bearingVectors0.push_back(p0_3d);
    bearingVectors1.push_back(p1_3d);
  }
  opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors0,
                                                        bearingVectors1);
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();
  // get the first result
  std::vector<int> inliers = ransac.inliers_;
  opengv::transformation_t transformation = ransac.model_coefficients_;

  adapter.sett12(transformation.col(3));
  adapter.setR12(transformation.block<3, 3>(0, 0));
  transformation = opengv::relative_pose::optimize_nonlinear(adapter, inliers);

  ran.sac_model_->selectWithinDistance(transformation, ransac_thresh, inliers);

  if ((int)inliers.size() >= ransac_min_inliers) {
    for (auto& inlier : inliers) {
      md.inliers.push_back(md.matches[inlier]);
    }
  }
  Sophus::SE3d T_i_j(transformation.block<3, 3>(0, 0), transformation.col(3));
  md.T_i_j = T_i_j;
}
}  // namespace visnav
