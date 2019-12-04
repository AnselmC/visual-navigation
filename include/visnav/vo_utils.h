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

#include <algorithm>
#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  for (auto& landmark : landmarks) {
    TrackId trackid = landmark.first;
    Eigen::Vector3d p3d = landmark.second.p;
    Eigen::Vector3d p3d_c = current_pose.inverse() * p3d;
    if (p3d_c[2] >= cam_z_threshold) {
      Eigen::Vector2d p2d_c = cam->project(p3d_c);
      if (p2d_c[0] >= 0 and p2d_c[0] <= 752 and p2d_c[1] >= 0 and
          p2d_c[1] <= 480) {
        projected_points.push_back(p2d_c);
        projected_track_ids.push_back(trackid);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations
  // of the landmarks. The feature_match_max_dist and
  // feature_match_test_next_best should be used to filter outliers the same
  // way as in exercise 3.

  // go through every feature

  for (size_t i = 0; i < kdl.corners.size(); i++) {
    FeatureId featureid0 = i;
    Eigen::Vector2d corner = kdl.corners.at(featureid0);
    std::bitset<256> descriptor = kdl.corner_descriptors.at(featureid0);
    std::vector<std::pair<Eigen::Vector2d, TrackId>> candidate_points;
    // select landmark candidates based on 2d distance
    for (size_t j = 0; j < projected_points.size(); j++) {
      Eigen::Vector2d point = projected_points.at(j);
      double distance = std::sqrt(std::pow(point[0] - corner[0], 2) +
                                  std::pow(point[1] - corner[1], 2));
      if (distance <= match_max_dist_2d) {
        candidate_points.push_back(
            std::make_pair(point, projected_track_ids.at(j)));
      }
    }

    // go through candidate matches to find smallest distance
    std::vector<int> distances;
    for (auto& candidate : candidate_points) {
      TrackId trackid = candidate.second;
      Landmark landmark = landmarks.at(trackid);
      FeatureTrack obs = landmark.obs;

      int min_dist = 256;
      for (auto& ob : obs) {
        TimeCamId tcid = ob.first;
        FeatureId featureid = ob.second;
        std::bitset<256> ob_descriptor =
            feature_corners.at(tcid).corner_descriptors.at(featureid);
        int hamming_dist = (descriptor ^ ob_descriptor).count();
        if (hamming_dist < min_dist) {
          min_dist = hamming_dist;
        }
      }
      distances.push_back(min_dist);
    }
    // get best match from distances
    // distance needs to be smaller than feature_match_max_dist and at least
    // feature_match_test_next_best bigger than second smallest distance
    if (distances.size() > 0) {
      int smallest_dist, second_smallest_dist;
      auto min_element = std::min_element(distances.begin(), distances.end());
      smallest_dist = *min_element;
      int min_idx = min_element - distances.begin();
      std::sort(distances.begin(), distances.end());
      if (distances.size() > 1) {
        second_smallest_dist = distances.at(1);
      } else {
        second_smallest_dist = smallest_dist * feature_match_max_dist + 1;
      }
      if (smallest_dist < feature_match_max_dist and
          smallest_dist * feature_match_test_next_best < second_smallest_dist) {
        TrackId match = candidate_points.at(min_idx).second;
        md.matches.push_back(std::make_pair(featureid0, match));
      }
    }
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  // TODO SHEET 5: Find the pose (T_w_c) and the inliers using the landmark to
  // keypoints matches and PnP. This should be similar to the localize_camera
  // in exercise 4 but in this execise we don't explicitelly have tracks.
  opengv::points_t points3d;  // contains landmarks, i.e. 3d points
  opengv::bearingVectors_t bearingVectors;  // contains 3d direction vectors
                                            // of unprojected keypoints
  double ranc_thresh =
      1.0 -
      std::cos(std::atan(reprojection_error_pnp_inlier_threshold_pixel / 500.));
  for (auto& match : md.matches) {
    FeatureId featureid = match.first;
    TrackId trackid = match.second;
    Eigen::Vector2d p2d = kdl.corners.at(featureid);
    Eigen::Vector3d v3d = cam->unproject(p2d);
    Eigen::Vector3d p3d = landmarks.at(trackid).p;
    bearingVectors.push_back(v3d);
    points3d.push_back(p3d);
  }
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors,
                                                        points3d);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ranc_thresh;
  ransac.computeModel();
  // get the result
  inliers = ransac.inliers_;
  opengv::transformation_t transformation = ransac.model_coefficients_;
  adapter.sett(transformation.col(3));
  adapter.setR(transformation.block<3, 3>(0, 0));
  transformation = opengv::absolute_pose::optimize_nonlinear(adapter, inliers);
  ransac.sac_model_->selectWithinDistance(transformation, ranc_thresh, inliers);
  T_w_c = Sophus::SE3d(transformation.block<3, 3>(0, 0), transformation.col(3));
  inliers = ransac.inliers_;
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  auto cam0 = calib_cam.intrinsics.at(tcidl.second).get();
  auto cam1 = calib_cam.intrinsics.at(tcidr.second).get();
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // add observations to existing landmarks
  std::vector<std::pair<FeatureId, FeatureId>> existingFeatures;
  int inlier_index = 0;
  bool foundFirst = false;
  for (auto& match : md.matches) {
    FeatureId featureid0 = match.first;
    TrackId trackid = match.second;

    if (std::find(inliers.begin(), inliers.end(), inlier_index) !=
        inliers.end()) {
      landmarks.at(trackid).obs.insert(std::make_pair(tcidl, featureid0));
      foundFirst = true;
    }
    for (auto& stereo_match : md_stereo.inliers) {
      if (stereo_match.first == featureid0) {
        FeatureId featureid1 = stereo_match.second;
        landmarks.at(trackid).obs.insert(std::make_pair(tcidr, featureid1));
        if(foundFirst){
          existingFeatures.push_back(std::make_pair(featureid0, featureid1));
        }
      }
    }
    inlier_index++;
    foundFirst = false;
  }
  // add new landmarks
  for (auto& stereo_match : md_stereo.inliers) {
     if (std::find(existingFeatures.begin(), existingFeatures.end(),
                  stereo_match) != existingFeatures.end()) {
      continue;  // already a landmark
    }
    Eigen::Vector2d p2d0 = kdl.corners.at(stereo_match.first);
    Eigen::Vector2d p2d1 = kdr.corners.at(stereo_match.second);
    Eigen::Vector3d p3d0 = cam0->unproject(p2d0);
    Eigen::Vector3d p3d1 = cam1->unproject(p2d1);
    opengv::bearingVectors_t bearingVectors0, bearingVectors1;
    bearingVectors0.push_back(p3d0);
    bearingVectors1.push_back(p3d1);
    opengv::relative_pose::CentralRelativeAdapter adapter(
        bearingVectors0, bearingVectors1, t_0_1, R_0_1);

    Eigen::Vector3d p3d0_tri = opengv::triangulation::triangulate(adapter, 0);
    Eigen::Vector3d p3d_world = T_w_c0 * p3d0_tri;
    Landmark new_landmark;
    new_landmark.p = p3d_world;
    new_landmark.obs.insert(std::make_pair(tcidl, stereo_match.first));
    new_landmark.obs.insert(std::make_pair(tcidr, stereo_match.second));
    landmarks.insert(std::make_pair(next_landmark_id, new_landmark));
    next_landmark_id++;
  }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  // TODO SHEET 5: Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs. The ids of all the keyframes that are currently in the
  // optimization should be stored in kf_frames. Removed keyframes should be
  // removed from cameras and landmarks with no left observations should be
  // moved to old_landmarks.
  while ((int)kf_frames.size() > max_num_kfs) {
    FrameId frameid = *kf_frames.begin();
    TimeCamId tcid0 = std::make_pair(frameid, 0);
    TimeCamId tcid1 = std::make_pair(frameid, 1);
    kf_frames.erase(frameid);
    cameras.erase(tcid0);
    cameras.erase(tcid1);
    for (auto it = landmarks.cbegin(); it != landmarks.cend();) {
      // remove associated observations
      for (auto obs_it = it->second.obs.cbegin();
           obs_it != it->second.obs.cend();) {
        auto obs = *obs_it;
        if (obs.first == tcid0 or obs.first == tcid1) {
          obs_it = landmarks.at(it->first).obs.erase(obs_it);
        } else {
          ++obs_it;
        }
      }
      // move landmarks with no more observations to old_landmarks
      if (it->second.obs.size() == 0) {
        old_landmarks.insert(*it);
        it = landmarks.erase(it);
      } else {
        ++it;
      }
    }
  }
}
}  // namespace visnav
