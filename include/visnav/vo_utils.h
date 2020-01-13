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
                       Landmarks& landmarks, std::vector<TrackId> kf_lms,
                       TrackId& next_landmark_id) {
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
    kf_lms.push_back(trackid);

    if (std::find(inliers.begin(), inliers.end(), inlier_index) !=
        inliers.end()) {
      landmarks.at(trackid).obs.insert(std::make_pair(tcidl, featureid0));
      foundFirst = true;
    }
    for (auto& stereo_match : md_stereo.inliers) {
      if (stereo_match.first == featureid0) {
        FeatureId featureid1 = stereo_match.second;
        landmarks.at(trackid).obs.insert(std::make_pair(tcidr, featureid1));
        if (foundFirst) {
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
    kf_lms.push_back(++next_landmark_id);
  }
}

void remove_kf(FrameId current_kf, Cameras& cameras, Landmarks& old_landmarks,
               Landmarks& landmarks) {
  // remove keyframe from keyframes

  // kf_frames.erase(current_kf);
  TimeCamId tcidl = std::pair<FrameId, CamId>(current_kf, 0);
  TimeCamId tcidr = std::pair<FrameId, CamId>(current_kf, 1);
  cameras.erase(tcidl);
  cameras.erase(tcidr);

  // remove this keyframe if stored in observations of existing landmarks
  for (auto it = landmarks.cbegin(); it != landmarks.cend();) {
    // remove associated observations
    for (auto obs_it = it->second.obs.cbegin();
         obs_it != it->second.obs.cend();) {
      auto obs = *obs_it;
      if (obs.first == tcidl or obs.first == tcidr) {
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

void compute_covisibility(const Keyframes& kf_frames, const int& min_weight,
                          CovisibilityGraph& cov_graph) {
  cov_graph.clear();
  for (auto& kf : kf_frames) {
    std::vector<TrackId> lm = kf.second;
    std::vector<std::tuple<FrameId, int>> weights;
    for (auto& other_kf : kf_frames) {
      if (kf.first == other_kf.first) continue;
      std::vector<TrackId> other_lm = other_kf.second;
      int weight = 0;
      for (const TrackId& tid : lm) {
        for (const TrackId& other_tid : other_lm) {
          if (tid == other_tid) weight++;
        }
      }
      if (weight >= min_weight) {
        weights.push_back(std::make_tuple(other_kf.first, weight));
      }
    }
    cov_graph.insert(std::make_pair(kf.first, weights));
  }
}

void make_keyframe_decision(bool& take_keyframe,
                            const int& max_frames_since_last_kf,
                            const int& frames_since_last_kf,
                            const int& new_kf_min_inliers, const int& min_kfs,
                            const double& max_kref_overlap,
                            const bool& mapping_busy, const MatchData& md,
                            const Keyframes& kf_frames) {
  if (kf_frames.size() < (uint)min_kfs) {
    take_keyframe = !mapping_busy;
    return;
  }

  int max_count = 0;
  for (auto& kf : kf_frames) {
    int count = 0;
    std::vector<TrackId> lms = kf.second;
    for (auto& match : md.matches) {
      TrackId trackId = match.second;
      bool kf_sees_landmark =
          std::find(lms.begin(), lms.end(), trackId) != lms.end();
      if (kf_sees_landmark) {
        count++;
      }
    }
    if (count > max_count) {
      max_count = count;
    }
  }

  // TODO: change AND to OR and ensure mapping/optimization is interrupted
  // if frames_since_last_kf > 20
  bool cond1 = !mapping_busy;
  //! mapping_busy && frames_since_last_kf > max_frames_since_last_kf;
  bool cond2 = md.matches.size() > (uint)new_kf_min_inliers;
  bool cond3 =
      (double)max_count / (double)md.matches.size() <= max_kref_overlap;
  std::cout << max_count << std::endl;
  std::cout << md.matches.size() << std::endl;
  std::cout << "Condition 1 fulfilled: " << cond1 << std::endl;
  std::cout << "Condition 2 fulfilled: " << cond2 << std::endl;
  std::cout << "Condition 3 fulfilled: " << cond3 << std::endl;
  take_keyframe = cond1 && cond2 && cond3;
}

void add_new_keyframe(const FrameId& new_kf,
                      const std::vector<TrackId>& kf_landmarks,
                      Keyframes& kf_frames) {
  kf_frames.insert(std::make_pair(new_kf, kf_landmarks));
}
double calculate_translation_error(const Sophus::SE3d& groundtruth_pose,
                                   const Sophus::SE3d& estimated_pose) {
  double trans_error =
      (groundtruth_pose.translation() - estimated_pose.translation()).norm();
  return trans_error;
}

double calculate_absolute_pose_error(const Sophus::SE3d& groundtruth_pose,
                                     const Sophus::SE3d& estimated_pose) {
  double ape = ((groundtruth_pose.inverse() * estimated_pose).matrix() -
                Eigen::Matrix4d::Identity())
                   .norm();
  return ape;
}

double calculate_relative_pose_error(const Sophus::SE3d& gt_pose,
                                     const Sophus::SE3d& gt_pose_prev,
                                     const Sophus::SE3d& est_pose,
                                     const Sophus::SE3d& est_pose_prev) {
  Sophus::SE3d gt_tf = gt_pose_prev.inverse() * gt_pose;
  Sophus::SE3d est_tf = est_pose_prev.inverse() * est_pose;

  double rpe = calculate_absolute_pose_error(gt_tf, est_tf);
  return rpe;
}

// void add_to_cov_graph(const FrameId& new_kf, const Keyframes&
// kf_frames,
//                       CovisibilityGraph& cov_graph) {
//   std::vector<std::tuple<FrameId, int>> new_weights;
//   for (auto& node : cov_graph) {
//     FrameId kf = node.first;
//     auto weights = node.second;
//     int curr_weight = 0;
//     for (TrackId& trackid : kf_frames.at(kf)) {
//       if (kf != new_kf) {
//         std::vector<TrackId> new_lms = kf_frames.at(new_kf);
//         if (std::find(new_lms.begin(), new_lms.end(), trackid) !=
//             new_lms.end()) {
//           curr_weight++;
//         }
//       }
//     }
//     if (curr_weight >= min_cov_weight) {
//       new_weights.push_back(std::make_tuple(kf, curr_weight));
//     }
//   }
//   cov_graph.insert(std::make_pair(new_kf, new_weights));
// }

void remove_old_keyframes(Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks, Keyframes& kf_frames,
                          const int& min_kfs,
                          const double& max_redundant_obs_count) {
  if (kf_frames.size() < (uint)min_kfs) return;
  for (auto current_kf = kf_frames.begin(); current_kf != kf_frames.end();) {
    std::vector<TrackId> current_landmarks = (*current_kf).second;
    int overlap_count = 0;
    for (auto current_landmark = current_landmarks.cbegin();
         current_landmark != current_landmarks.cend(); current_landmark++) {
      // if at least three other kf observe this landmark, increment the
      // overlap_count
      std::set<FrameId> unique_frameIds;
      Landmark lm = landmarks.at(*current_landmark);
      for (auto& obs : lm.obs) {
        unique_frameIds.insert(obs.first.first);
      }

      if (unique_frameIds.size() > 3) {
        overlap_count++;
      }
    }
    double overlap_percentage =
        (double)overlap_count / (double)current_landmarks.size();
    if (overlap_percentage >= max_redundant_obs_count) {
      remove_kf((*current_kf).first, cameras, old_landmarks, landmarks);
      current_kf = kf_frames.erase(current_kf);
    } else {
      current_kf++;
    }
  }
}
}  // namespace visnav
