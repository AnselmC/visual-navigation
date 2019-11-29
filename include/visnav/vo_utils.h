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
    if (p3d_c[2] > cam_z_threshold) {
      Eigen::Vector2d p2d_c = cam->project(p3d_c);
      projected_points.push_back(p2d_c);
      projected_track_ids.push_back(trackid);
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
  for (size_t featureid0 = 0; featureid0 < kdl.corners.size(); featureid0++) {
    Eigen::Vector2d corner = kdl.corners.at(featureid0);
    std::bitset<256> descriptor = kdl.corner_descriptors.at(featureid0);
    std::vector<std::pair<Eigen::Vector2d, TrackId>> candidate_points;
    // select landmark candidates based on 2d distance
    for (size_t j = 0; j < projected_points.size(); j++) {
      Eigen::Vector2d point = projected_points.at(j);
      double distance = std::sqrt(std::pow(point[0] - corner[0], 2) +
                                  std::pow(point[1] - corner[1], 2));
      if (distance <= max_match_dist_2d) {
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
        TimeCamId tcid = obs.first;
        FeatureId featureid = obs.second;
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
    int smallest_dist, second_smallest_dist;
    smallest_dist = std::min_element(distances.begin(), distances.end());
    int min_idx = distances.get_index(smallest_dist);
    std::sort(distances.begin(), distances.end());
    second_smallest_dist = distances.at(1);
    if (smallest_dist <= feature_match_max_dist and
        smallest_dist * feature_match_test_next_best <= second_smallest_dist) {
      // valid match
      // TODO: what is featureid1? Doesn't make sense to me..
      // would only make sense to save track id
      TrackID match = candidate_points.at(min_idx).second;
      md.matches.push_back(std::make_pair<featureid0, trackid>);
    }

    return;
  }

  // TODO SHEET 5: Find the pose (T_w_c) and the inliers using the landmark to
  // keypoints matches and PnP. This should be similar to the localize_camera
  // in exercise 4 but in this execise we don't explicitelly have tracks.
  UNUSED(cam);
  UNUSED(kdl);
  UNUSED(landmarks);
  UNUSED(reprojection_error_pnp_inlier_threshold_pixel);
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains landmark to map
  // matches for the left camera (camera 0). The inliers vector contains all
  // inliers in md that were used to compute the pose T_w_c0. For all inliers
  // add the observations to the existing landmarks (if the left point is in
  // md_stereo.inliers then add both observations). For all stereo
  // observations that were not added to the existing landmarks triangulate
  // and add new landmarks. Here next_landmark_id is a running index of the
  // landmarks, so after adding a new landmark you should always increase
  // next_landmark_id by 1.
  UNUSED(tcidl);
  UNUSED(tcidr);
  UNUSED(kdl);
  UNUSED(kdr);
  UNUSED(T_w_c0);
  UNUSED(calib_cam);
  UNUSED(inliers);
  UNUSED(md_stereo);
  UNUSED(md);
  UNUSED(landmarks);
  UNUSED(next_landmark_id);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
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
    UNUSED(max_num_kfs);
    UNUSED(cameras);
    UNUSED(landmarks);
    UNUSED(old_landmarks);
  }
}  // namespace visnav
