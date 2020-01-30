#pragma once

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {
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

// get subset of Landmarks with TrackId stored in landmark_ids
// from landmarks into landmarks_subset
void get_landmark_subset(const Landmarks& landmarks,
                         const std::set<TrackId>& landmark_ids,
                         Landmarks& landmarks_subset) {
  auto start = std::chrono::high_resolution_clock::now();
  landmarks_subset.clear();
  for (auto& lid : landmark_ids) {
    if (landmarks.find(lid) != landmarks.end()) {
      landmarks_subset.emplace(lid, landmarks.at(lid));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Getting landmark subset took: " << time_taken
            << std::setprecision(9) << " sec" << std::endl;
}

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double d_min, const double d_max,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();
  auto start = std::chrono::high_resolution_clock::now();

  for (auto& landmark : landmarks) {
    TrackId trackid = landmark.first;
    Eigen::Vector3d p3d = landmark.second.p;
    Eigen::Vector3d p3d_c = current_pose.inverse() * p3d;
    if (p3d_c.norm() >= d_min && p3d_c.norm() <= d_max) {
      Eigen::Vector2d p2d_c = cam->project(p3d_c);
      if (p2d_c[0] >= 0 and p2d_c[0] <= 752 and p2d_c[1] >= 0 and
          p2d_c[1] <= 480) {
        projected_points.emplace_back(p2d_c);
        projected_track_ids.emplace_back(trackid);
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "project took: " << time_taken << std::setprecision(9) << " sec"
            << std::endl;
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

  auto start = std::chrono::high_resolution_clock::now();
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
        candidate_points.emplace_back(
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
      distances.emplace_back(min_dist);
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
        TrackId tid = candidate_points.at(min_idx).second;
        md.matches.emplace_back(std::make_pair(featureid0, tid));
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "matching took: " << time_taken << std::setprecision(9) << " sec"
            << std::endl;
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  auto start = std::chrono::high_resolution_clock::now();
  if (md.matches.size() < 4) {
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
    bearingVectors.emplace_back(v3d);
    points3d.emplace_back(p3d);
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
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "localization took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}

int get_kfs_shared_landmarks(const Landmarks& landmarks, const MatchData& md,
                             const int min_weight_k1, std::set<FrameId>& k1) {
  auto start = std::chrono::high_resolution_clock::now();
  int max_count = 0;
  // want to find all keyframes that see landmarks of candidate
  // to do this we iterate through all landmarks the candidate sees
  // and get all keyframes that observe this landmark as well (obs of landmarks
  // - obs however is a timecamid and not a frameid) for every keyframe we add
  // we would also like to know how many observations are shared(update weight)
  // go through matched landmarks
  std::map<FrameId, int> k_weight;
  for (auto& match : md.matches) {
    TrackId tid = match.second;
    Landmark lm = landmarks.at(tid);
    // go through all observing cams
    bool found_first = false;
    for (auto& ob : lm.obs) {
      // don't increment for second camera of same kf(same frameid but different
      // camid)
      if (!found_first) {
        if (k_weight.find(ob.first.first) != k_weight.end()) {
          k_weight[ob.first.first]++;
        } else {
          k_weight.emplace(ob.first.first, 1);
        }
        found_first = true;
      }
    }
    for (auto& kw : k_weight) {
      int count = kw.second;
      FrameId fid = kw.first;
      if (count > min_weight_k1) {
        k1.emplace(fid);
      }
      if (count > max_count) {
        max_count = count;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Get kfs shared lms took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
  return max_count;
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       const double d_min, const double d_max,
                       Landmarks& landmarks, std::set<TrackId>& lm_ids,
                       TrackId& next_landmark_id) {
  auto start = std::chrono::high_resolution_clock::now();
  auto cam0 = calib_cam.intrinsics.at(tcidl.second).get();
  auto cam1 = calib_cam.intrinsics.at(tcidr.second).get();
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();
  lm_ids.clear();

  // add observations to existing landmarks
  std::vector<std::pair<FeatureId, FeatureId>> existingFeatures;
  int inlier_index = 0;
  bool foundFirst = false;
  for (auto& match : md.matches) {
    FeatureId featureid0 = match.first;
    TrackId trackid = match.second;

    if (std::find(inliers.begin(), inliers.end(), inlier_index) !=
        inliers.end()) {
      landmarks.at(trackid).obs.emplace(tcidl, featureid0);
      lm_ids.emplace(trackid);
      foundFirst = true;
    }
    for (auto& stereo_match : md_stereo.inliers) {
      if (stereo_match.first == featureid0) {
        FeatureId featureid1 = stereo_match.second;
        landmarks.at(trackid).obs.emplace(tcidr, featureid1);
        lm_ids.emplace(trackid);
        if (foundFirst) {
          existingFeatures.emplace_back(featureid0, featureid1);
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
    bearingVectors0.emplace_back(p3d0);
    bearingVectors1.emplace_back(p3d1);
    opengv::relative_pose::CentralRelativeAdapter adapter(
        bearingVectors0, bearingVectors1, t_0_1, R_0_1);

    Eigen::Vector3d p3d0_tri = opengv::triangulation::triangulate(adapter, 0);
    Eigen::Vector3d p3d_world = T_w_c0 * p3d0_tri;
    if (p3d_world.norm() >= d_min && p3d_world.norm() <= d_max) {
      Landmark new_landmark;
      new_landmark.p = p3d_world;
      new_landmark.obs.emplace(tcidl, stereo_match.first);
      new_landmark.obs.emplace(tcidr, stereo_match.second);
      landmarks.emplace(next_landmark_id, new_landmark);
      lm_ids.emplace(next_landmark_id++);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Add new landmarks took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}

void remove_kf(FrameId current_kf, Cameras& cameras, Landmarks& landmarks) {
  auto start = std::chrono::high_resolution_clock::now();
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
    // remove landmarks with no more observations
    if (it->second.obs.size() == 0) {
      it = landmarks.unsafe_erase(it);
    } else {
      ++it;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Remove kf took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}
double get_max_pose_difference(const Sophus::SE3d& ref_pose,
                               const Cameras& cameras,
                               const Connections& neighbors) {
  // TODO: remove magic number
  double max_diff = 0;
  for (auto& neighbor : neighbors) {
    TimeCamId tcidl(neighbor.first, 0);
    Sophus::SE3d other_pose = cameras.at(tcidl).T_w_c;
    double diff = calculate_absolute_pose_error(ref_pose, other_pose);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}
std::set<FrameId> get_loop_closure_candidates(const FrameId& new_kf,
                                              const Sophus::SE3d& pose,
                                              const Cameras& cameras,
                                              const Connections& neighbors,
                                              const Keyframes& kf_frames,
                                              const double& max_diff) {
  std::set<FrameId> candidates;
  for (auto& kf : kf_frames) {
    if (neighbors.find(kf.first) != neighbors.end() || kf.first == new_kf)
      continue;
    TimeCamId other_tcidl(kf.first, 0);
    Sophus::SE3d other_pose = cameras.at(other_tcidl).T_w_c;
    double diff = calculate_absolute_pose_error(pose, other_pose);
    if (diff <= max_diff) {
      candidates.emplace(kf.first);
    }
  }
  return candidates;
}
TrackId get_corresponding_landmark(const TimeCamId& tcid,
                                   const Keyframes& kf_frames,
                                   const Landmarks& landmarks,
                                   const FeatureId& fid) {
  LandmarkIds possible_landmarks = kf_frames.at(tcid.first);
  for (auto& lmid : possible_landmarks) {
    FeatureTrack obs = landmarks.at(lmid).obs;
    if (obs.find(tcid) != obs.end()) {
      if (obs.at(tcid) == fid) {
        return lmid;
      }
    }
  }
  // orb feature does not correspond to mappoint
  return -1;
}
LandmarkMatchData get_landmark_correspondences(const TimeCamId& tcid0,
                                               const TimeCamId& tcid1,
                                               const MatchData& md_features,
                                               const Keyframes& kf_frames,
                                               const Landmarks& landmarks) {
  // md_features.matches contains pairs of feature ids
  LandmarkMatchData lmmd;
  for (auto& feature_corr : md_features.matches) {
    FeatureId fid0 = feature_corr.first;
    FeatureId fid1 = feature_corr.second;
    TrackId tid0 =
        get_corresponding_landmark(tcid0, kf_frames, landmarks, fid0);
    if (tid0 == -1) continue;
    TrackId tid1 =
        get_corresponding_landmark(tcid1, kf_frames, landmarks, fid1);
    if (tid1 == -1) continue;
    if (tid0 == tid1) continue;
    lmmd.matches.emplace_back(tid0, tid1);
  }
  return lmmd;
}
void compute_similarity_transform(
    const LandmarkMatchData& lmmd, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& S_i_j, std::vector<int>& inliers) {
  opengv::points_t points1;
  opengv::points_t points2;
  if (lmmd.matches.size() < 3) {
    S_i_j = Sophus::SE3d();
    return;
  }
  for (auto& match : lmmd.matches) {
    Eigen::Vector3d p1 = landmarks.at(match.first).p;
    Eigen::Vector3d p2 = landmarks.at(match.second).p;
    points1.emplace_back(p1);
    points2.emplace_back(p2);
  }
  opengv::point_cloud::PointCloudAdapter adapter(points1, points2);
  opengv::sac::Ransac<opengv::sac_problems::point_cloud::PointCloudSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::point_cloud::PointCloudSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::point_cloud::PointCloudSacProblem(adapter));

  double ranc_thresh =
      1.0 -
      std::cos(std::atan(reprojection_error_pnp_inlier_threshold_pixel / 500.));
  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ranc_thresh;
  ransac.computeModel(0);  // 0 comes from docs - no idea what it means
  // get the result
  inliers = ransac.inliers_;
  opengv::transformation_t transformation = ransac.model_coefficients_;
  adapter.sett12(transformation.col(3));
  adapter.setR12(transformation.block<3, 3>(0, 0));
  transformation = opengv::point_cloud::optimize_nonlinear(adapter, inliers);
  ransac.sac_model_->selectWithinDistance(transformation, ranc_thresh, inliers);
  S_i_j = Sophus::SE3d(transformation.block<3, 3>(0, 0), transformation.col(3));
  inliers = ransac.inliers_;
}
FrameId perform_matching(const Keyframes& kf_frames,
                         const std::set<FrameId> candidates,
                         const TimeCamId& tcid_new,
                         const Corners& feature_corners,
                         const Landmarks& landmarks, const OrbSLAMOptions& opts,
                         LandmarkMatchData& lmmd) {
  lmmd.matches.clear();
  FrameId final_candidate = -1;
  int num_best_inliers = 0;
  for (auto& candidate : candidates) {
    // TODO: only match features that are landmarks already
    TimeCamId tcid_candidate(candidate, 0);
    KeypointsData kd_new = feature_corners.at(tcid_new);
    KeypointsData kd_candidate = feature_corners.at(tcid_candidate);
    MatchData md_features;
    matchDescriptors(kd_new.corner_descriptors, kd_candidate.corner_descriptors,
                     md_features.matches, opts.feature_match_max_dist,
                     opts.feature_match_test_next_best);
    std::cout << "Num matches: " << md_features.matches.size() << std::endl;
    LandmarkMatchData local_lmmd = get_landmark_correspondences(
        tcid_new, tcid_candidate, md_features, kf_frames, landmarks);
    std::cout << "Num matches that are landmarks: " << local_lmmd.matches.size()
              << std::endl;
    // std::vector<int> inliers;
    // Sophus::SE3d sim_transform;
    // compute_similarity_transform(
    //    local_lmmd, landmarks,
    //    opts.reprojection_error_pnp_inlier_threshold_pixel, sim_transform,
    //    inliers);
    // std::cout << "NUM INLIERS: " << inliers.size() << std::endl;
    // if (int(inliers.size()) >= opts.min_inliers_loop_closing &&
    //    int(inliers.size()) > num_best_inliers) {
    //  lmmd.matches.clear();
    //  final_candidate = candidate;
    //  num_best_inliers = inliers.size();
    //  for (auto& inlier : inliers) {
    //    lmmd.matches.push_back(local_lmmd.matches.at(inlier));
    //  }
    //}
    if (int(local_lmmd.matches.size()) >= opts.min_inliers_loop_closing &&
        int(local_lmmd.matches.size()) > num_best_inliers) {
      final_candidate = candidate;
      num_best_inliers = local_lmmd.matches.size();
      lmmd.matches = local_lmmd.matches;
    }
  }
  return final_candidate;
}
void get_neighbor_landmarks_and_ids(const Keyframes& kf_frames,
                                    const Connections& neighbors,
                                    LandmarkIds& local_lm_ids,
                                    std::set<FrameId>& neighbor_ids) {
  for (auto& neighbor : neighbors) {
    neighbor_ids.emplace(neighbor.first);
    LandmarkIds lms = kf_frames.at(neighbor.first);
    local_lm_ids.insert(lms.begin(), lms.end());
  }
}
void get_cov_map(const FrameId kf, const Keyframes& kf_frames,
                 const CovisibilityGraph& cov_graph, LandmarkIds& local_lms,
                 std::set<FrameId>& cov_frames) {
  // get all landmarks from keyframes connected in cov graph
  if (cov_graph.empty()) return;
  auto start = std::chrono::high_resolution_clock::now();
  cov_frames.clear();
  local_lms.clear();
  LandmarkIds lm_ids;
  auto neighbors = cov_graph.at(kf);

  get_neighbor_landmarks_and_ids(kf_frames, neighbors, local_lms, cov_frames);
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Get cov map took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}
void get_local_map(const MatchData& md_prev, const Landmarks& landmarks,
                   const Keyframes& kf_frames,
                   const CovisibilityGraph& cov_graph, const int min_weight_k1,
                   Landmarks& local_landmarks) {
  auto start = std::chrono::high_resolution_clock::now();
  if (kf_frames.empty()) return;
  std::set<FrameId> k1;
  get_kfs_shared_landmarks(landmarks, md_prev, min_weight_k1, k1);
  std::set<FrameId> local_lm_ids;
  std::cout << "K1 size: " << k1.size() << std::endl;
  Connections neighbors;
  for (auto& kf : k1) {
    auto lmids = kf_frames.at(kf);
    local_lm_ids.insert(lmids.begin(), lmids.end());
    Connections curr_neighbors = cov_graph.at(kf);
    Connections new_neighbors;
    std::set_difference(curr_neighbors.begin(), curr_neighbors.end(),
                        neighbors.begin(), neighbors.end(),
                        std::inserter(new_neighbors, new_neighbors.begin()));
    neighbors.insert(new_neighbors.begin(), new_neighbors.end());
    std::set<FrameId> neighbor_ids;
    get_neighbor_landmarks_and_ids(kf_frames, new_neighbors, local_lm_ids,
                                   neighbor_ids);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Get local map took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
  std::cout << "Local map contains " << local_lm_ids.size() << " points."
            << std::endl;
  get_landmark_subset(landmarks, local_lm_ids, local_landmarks);
}

void old_make_keyframe_decision(bool& take_keyframe, const bool mapping_busy,
                                const int new_kf_min_inliers,
                                const MatchData& md) {
  take_keyframe = !mapping_busy && int(md.matches.size()) < new_kf_min_inliers;
  return;
}
void make_keyframe_decision(bool& take_keyframe, const Landmarks& landmarks,
                            const int max_frames_since_last_kf,
                            const int frames_since_last_kf,
                            const int new_kf_min_inliers, const int min_kfs,
                            const int min_weight_k1,
                            const double max_kref_overlap,
                            const bool mapping_busy, const MatchData& md,
                            const Keyframes& kf_frames) {
  if (int(kf_frames.size()) < min_kfs ||
      frames_since_last_kf > max_frames_since_last_kf) {
    take_keyframe = !mapping_busy;
    std::cout << "Mapping busy: " << !take_keyframe << std::endl;
    return;
  }

  std::set<FrameId> k1;

  // TODO: change AND to OR and ensure mapping/optimization is interrupted
  // if frames_since_last_kf > 20
  bool cond1 = !mapping_busy;
  if (!cond1) {
    std::cout << "Mapping busy..." << std::endl;
    take_keyframe = false;
    return;
  }  //! mapping_busy && frames_since_last_kf > max_frames_since_last_kf;
  bool cond2 = int(md.matches.size()) > new_kf_min_inliers;
  if (!cond2) {
    std::cout << "Not enough matches..." << std::endl;
    take_keyframe = false;
    return;
  }
  int max_count = get_kfs_shared_landmarks(landmarks, md, min_weight_k1, k1);
  bool cond3 =
      (double)max_count / (double)md.matches.size() <= max_kref_overlap;

  if (!cond3) {
    std::cout << "Not enough new points..." << std::endl;
  }
  take_keyframe = cond3;
}
void add_to_cov_graph(const FrameId& new_kf, const Keyframes& kf_frames,
                      const Cameras& cameras, const int min_weight,
                      RelativeTransforms& relative_transforms,
                      CovisibilityGraph& cov_graph) {
  Connections connections;
  TimeCamId tcid(new_kf, 0);
  Sophus::SE3d T_w_cnew = cameras.at(tcid).T_w_c;
  std::cout << "Size: " << kf_frames.size() << std::endl;
  std::cout << "Getting landmarks at " << new_kf << std::endl;
  LandmarkIds new_lms = kf_frames.at(new_kf);
  auto start = std::chrono::high_resolution_clock::now();
  double time_taken;
  for (auto& node : cov_graph) {
    FrameId kf = node.first;
    if (kf == new_kf) continue;
    LandmarkIds curr_lms = kf_frames.at(kf);
    TimeCamId tcid_curr(kf, 0);
    int curr_weight = 0;
    for (const TrackId& trackid : curr_lms) {
      if (new_lms.find(trackid) != new_lms.end()) {
        curr_weight++;
      }
    }

    if (curr_weight >= min_weight) {
      Sophus::SE3d T_w_curr = cameras.at(tcid_curr).T_w_c;
      Sophus::SE3d T_curr_new = T_w_curr.inverse() * T_w_cnew;
      connections.emplace(kf, curr_weight);
      relative_transforms.emplace(std::make_pair(new_kf, kf), T_curr_new);
      relative_transforms.emplace(std::make_pair(kf, new_kf),
                                  T_curr_new.inverse());
      node.second[new_kf] = curr_weight;
    }
  }
  cov_graph[new_kf] = connections;
  auto end = std::chrono::high_resolution_clock::now();
  time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Insert to cov graph took: " << time_taken
            << std::setprecision(9) << " sec" << std::endl;
}
void add_new_keyframe(const FrameId& new_kf, const std::set<TrackId>& lm_ids,
                      const Cameras& cameras, const int min_weight,
                      RelativeTransforms& relative_transforms,
                      Keyframes& kf_frames, CovisibilityGraph& cov_graph) {
  auto start = std::chrono::high_resolution_clock::now();
  kf_frames.emplace(new_kf, lm_ids);
  add_to_cov_graph(new_kf, kf_frames, cameras, min_weight, relative_transforms,
                   cov_graph);
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Add new keyframe took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}

void remove_from_cov_graph(const FrameId& old_kf,
                           CovisibilityGraph& cov_graph) {
  auto neighbors = cov_graph.at(old_kf);
  for (auto& neighbor : neighbors) {
    auto& curr_neighbors = cov_graph.at(neighbor.first);
    curr_neighbors.erase(old_kf);
  }
  cov_graph.erase(old_kf);
}

void merge_landmarks(const FrameId& kf, const LandmarkMatchData& lmmd,
                     const Cameras& cameras, const int min_weight,
                     RelativeTransforms& relative_transforms,
                     CovisibilityGraph& cov_graph, Keyframes& kf_frames,
                     Landmarks& landmarks, std::set<FrameId> prev_lm_ids,
                     Landmarks& old_landmarks) {
  old_landmarks.clear();
  auto neighbors = cov_graph.at(kf);
  neighbors.emplace(kf, 0);  // 0 b/c weight doesn't matter
  // iterate over all kf currently selected in covisibility graph
  for (auto& neighbor : neighbors) {
    // iterate through all landmark pairs (first is the landmark to be merged
    // into second)
    LandmarkIds& lms = kf_frames.at(neighbor.first);
    for (auto& lm_pair : lmmd.matches) {
      // TODO make more efficient
      if (prev_lm_ids.find(lm_pair.first) != prev_lm_ids.end()) {
        prev_lm_ids.erase(lm_pair.first);
        prev_lm_ids.insert(lm_pair.second);
      }
      if (lms.find(lm_pair.first) != lms.end()) {  // neighbor sees old landmark
        // change observations in kf_frames
        lms.erase(lm_pair.first);
        lms.emplace(lm_pair.second);
        TimeCamId tcidl(neighbor.first, 0),
            tcidr(neighbor.first,
                  1);  // need to check whether left, right or both cams see it
        FeatureTrack& obs = landmarks.at(lm_pair.first).obs;
        old_landmarks.emplace(*landmarks.find(lm_pair.first));
        auto obs_left = obs.find(tcidl);
        auto obs_right = obs.find(tcidr);
        // erase old and insert new observations
        if (obs_left != obs.end()) {
          landmarks.at(lm_pair.second).obs.emplace(*obs_left);
          obs.erase(obs_left);
        }
        if (obs_right != obs.end()) {
          landmarks.at(lm_pair.second).obs.emplace(*obs_right);
          obs.erase(obs_right);
        }
        if (obs.size() == 0) {  // remove landmark
          landmarks.unsafe_erase(lm_pair.first);
        }
      }
    }
    // works like update if kf already exists
    add_to_cov_graph(neighbor.first, kf_frames, cameras, min_weight,
                     relative_transforms, cov_graph);
  }
}
void remove_redundant_keyframes(Cameras& cameras, Landmarks& landmarks,
                                Keyframes& kf_frames,
                                CovisibilityGraph& cov_graph,
                                std::set<FrameId>& cov_frames,
                                const int min_kfs,
                                const double max_redundant_obs_count) {
  auto start = std::chrono::high_resolution_clock::now();
  for (auto current_kf = kf_frames.begin(); current_kf != kf_frames.end();) {
    if (int(kf_frames.size()) < min_kfs) return;
    LandmarkIds current_landmarks = current_kf->second;
    int overlap_count = 0;
    for (auto& current_landmark : current_landmarks) {
      // if at least three other kf observe this landmark, increment the
      // overlap_count
      std::set<FrameId> unique_frameIds;
      // TODO this shouldn't have to happen
      if (landmarks.find(current_landmark) == landmarks.end()) continue;
      Landmark lm = landmarks.at(current_landmark);
      for (auto& obs : lm.obs) {
        unique_frameIds.emplace(obs.first.first);
      }

      if (unique_frameIds.size() > 3) {
        overlap_count++;
      }
    }
    double overlap_percentage =
        (double)overlap_count / (double)current_landmarks.size();
    if (overlap_percentage >= max_redundant_obs_count) {
      std::cout << "REMOVING KF: " << current_kf->first << std::endl;
      remove_kf(current_kf->first, cameras, landmarks);
      remove_from_cov_graph(current_kf->first, cov_graph);
      if (cov_frames.count(current_kf->first) > 0) {
        cov_frames.erase(current_kf->first);
      }
      current_kf = kf_frames.unsafe_erase(current_kf);
    } else {
      current_kf++;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Rm keyframes took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}

void project_match_localize(const Calibration& calib_cam,
                            const Corners& feature_corners,
                            const OrbSLAMOptions& os_opts,
                            const KeypointsData& kdl,
                            const Landmarks& landmarks,
                            std::vector<int>& inliers,
                            Sophus::SE3d& current_pose, MatchData& md) {
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points;
  std::vector<TrackId> projected_track_ids;
  auto start = std::chrono::high_resolution_clock::now();
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    os_opts.d_min, os_opts.d_max, projected_points,
                    projected_track_ids);

  find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                         projected_track_ids, os_opts.match_max_dist_2d,
                         os_opts.feature_match_max_dist,
                         os_opts.feature_match_test_next_best, md);

  localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
                  os_opts.reprojection_error_pnp_inlier_threshold_pixel, md,
                  current_pose, inliers);
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e9;
  std::cout << "Project match loc took: " << time_taken << std::setprecision(9)
            << " sec" << std::endl;
}
}  // Namespace visnav
