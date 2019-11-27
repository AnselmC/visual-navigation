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

#include <fstream>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const TimeCamId& tcid0,
                                   const TimeCamId& tcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<TimeCamId> tcids = {tcid0, tcid1};
  if (!GetTracksInImages(tcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  // only contains trackids for which no landmark previously existed -> how this
  // would happen idk
  std::vector<TrackId> new_track_ids;

  auto corners0 = feature_corners.find(tcid0)->second.corners;
  auto corners1 = feature_corners.find(tcid1)->second.corners;
  int cam_id0 = tcid0.second;
  int cam_id1 = tcid1.second;
  auto cam0 = calib_cam.intrinsics.at(cam_id0).get();
  auto cam1 = calib_cam.intrinsics.at(cam_id1).get();
  // transformations from cameras to world
  Sophus::SE3d T_w_c0 = cameras.find(tcid0)->second.T_w_c;
  Sophus::SE3d T_w_c1 = cameras.find(tcid1)->second.T_w_c;
  // transformations from camera 1 to camera 0
  Sophus::SE3d T_0_1 = T_w_c0.inverse() * T_w_c1;

  Eigen::Matrix3d rotation = T_0_1.rotationMatrix();
  Eigen::Vector3d translation = T_0_1.translation();

  opengv::bearingVectors_t bearingVectors0, bearingVectors1;
  // go through all shared tracks
  for (TrackId& track_id : shared_track_ids) {
    bearingVectors0.clear();
    bearingVectors1.clear();
    // get track
    FeatureTrack feature_track = feature_tracks.find(track_id)->second;
    // get feature ids for cameras
    FeatureId feature_id0 = feature_track.find(tcid0)->second;
    FeatureId feature_id1 = feature_track.find(tcid1)->second;
    // get 2d points corresponding to feature ids
    Eigen::Vector2d p2d0 = corners0.at(feature_id0);
    Eigen::Vector2d p2d1 = corners1.at(feature_id1);
    // unproject to get 3d direction vector
    Eigen::Vector3d p3d0 = cam0->unproject(p2d0);
    Eigen::Vector3d p3d1 = cam1->unproject(p2d1);
    bearingVectors0.push_back(p3d0);
    bearingVectors1.push_back(p3d1);
    opengv::relative_pose::CentralRelativeAdapter adapter(
        bearingVectors0, bearingVectors1, translation, rotation);

    // triangulate points
    Eigen::Vector3d p3d0_tri = opengv::triangulation::triangulate(adapter, 0);
    // if z-value is negative, point is behind camera 0 -> invalid
    if (p3d0_tri[2] >= 0) {
      Eigen::Vector3d p3d_world = T_w_c0 * p3d0_tri;
      // check if track is already a landmark
      auto search = landmarks.find(track_id);
      if (search == landmarks.end()) {
        // if landmark doesn't yet exist, create a new landmark with 3d point,
        // add observations and insert it
        new_track_ids.push_back(track_id);
        Landmark new_landmark;
        new_landmark.p = p3d_world;
        new_landmark.obs.insert(std::make_pair(tcid0, feature_id0));
        new_landmark.obs.insert(std::make_pair(tcid1, feature_id1));
        landmarks.insert(std::make_pair(track_id, new_landmark));
      } else {  // landmark exists already
        // insert new observation(s)
        search->second.obs.insert(std::make_pair(tcid0, feature_id0));
        search->second.obs.insert(std::make_pair(tcid1, feature_id1));
      }
    }
  }
  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const TimeCamId& tcid0,
                                       const TimeCamId& tcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  int frame_id0 = tcid0.first;
  int frame_id1 = tcid1.first;
  int cam_id0 = tcid0.second;
  int cam_id1 = tcid1.second;
  if (!(frame_id0 == frame_id1 && cam_id0 != cam_id1)) {
    std::cerr << "Images " << tcid0 << " and " << tcid1
              << " don't for a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  Camera cam0, cam1;
  cam0.T_w_c =
      Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  cam1.T_w_c = calib_cam.T_i_c.at(cam_id1);
  cameras.insert(std::make_pair(tcid0, cam0));
  cameras.insert(std::make_pair(tcid1, cam1));
  add_new_landmarks_between_cams(tcid0, tcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const TimeCamId& tcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  opengv::points_t points3d;
  opengv::bearingVectors_t bearingVectors;
  auto cam = calib_cam.intrinsics.at(tcid.second).get();
  auto corners = feature_corners.find(tcid)->second.corners;
  for (const TrackId& track_id : shared_track_ids) {
    FeatureTrack feature_track = feature_tracks.find(track_id)->second;
    FeatureId feature_id = feature_track.find(tcid)->second;
    Eigen::Vector2d p2d = corners.at(feature_id);
    Eigen::Vector3d p3d = cam->unproject(p2d);
    bearingVectors.push_back(p3d);
    Eigen::Vector3d point = landmarks.find(track_id)->second.p;
    points3d.push_back(point);
  }

  double ransac_thresh =
      1.0 -
      std::cos(std::atan(reprojection_error_pnp_inlier_threshold_pixel / 500.));
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
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();
  // get the result
  std::vector<int> inliers = ransac.inliers_;
  opengv::transformation_t transformation = ransac.model_coefficients_;
  adapter.sett(transformation.col(3));
  adapter.setR(transformation.block<3, 3>(0, 0));
  transformation = opengv::absolute_pose::optimize_nonlinear(adapter, inliers);
  ransac.sac_model_->selectWithinDistance(transformation, ransac_thresh,
                                          inliers);
  std::copy(ransac.inliers_.begin(), ransac.inliers_.end(),
            std::back_inserter(inlier_track_ids));
  T_w_c = Sophus::SE3d(transformation.block<3, 3>(0, 0), transformation.col(3));
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
  };

  // Run bundle adjustment to optimize cameras, points, and optionally
  // intrinsics
  void bundle_adjustment(const Corners& feature_corners,
                         const BundleAdjustmentOptions& options,
                         const std::set<TimeCamId>& fixed_cameras,
                         Calibration& calib_cam, Cameras& cameras,
                         Landmarks& landmarks) {
    ceres::Problem problem;

    const int DIM_RESIDUAL = 2;
    const int DIM_INTR = 8;
    const int DIM_VECTOR = 3;
    Sophus::test::LocalParameterizationSE3* local_parameterization =
        new Sophus::test::LocalParameterizationSE3;
    ceres::LossFunction* loss_function;
    if (options.use_huber) {
      loss_function = new ceres::HuberLoss(options.huber_parameter);
    }
    // Optimize over all landmarks
    for (auto& landmark : landmarks) {
      // get 3d world point
      Eigen::Vector3d* p3d = &landmark.second.p;
      // Optimize over all positions
      for (auto& obs : landmark.second.obs) {
        TimeCamId tcid = obs.first;
        CamId cam_id = tcid.second;
        // get camera to world transformation
        Sophus::SE3d* T_w_c = &cameras.find(tcid)->second.T_w_c;
        FeatureId feature_id = obs.second;
        // get camera intrinsics
        auto intr = calib_cam.intrinsics.at(cam_id).get();
        // get camera name: why do we even need this? Aren't the intrinsics
        // already an impl of a specific camera?
        std::string cam_model = intr->name();
        // get corresponding 2d point from camera
        Eigen::Vector2d p_2d =
            feature_corners.find(tcid)->second.corners[feature_id];

        // create cost functor from cam_model and 2d point
        BundleAdjustmentReprojectionCostFunctor* costFunctor =
            new BundleAdjustmentReprojectionCostFunctor(p_2d, cam_model);

        // create cost function
        ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
            BundleAdjustmentReprojectionCostFunctor, DIM_RESIDUAL,
            Sophus::SE3d::num_parameters, DIM_INTR, DIM_VECTOR>(costFunctor);

        // add residual block
        problem.AddResidualBlock(costFunction, loss_function, T_w_c->data(),
                                 p3d->data(), intr->data());
        // set parameterization of T_w_c
        problem.SetParameterization(T_w_c->data(), local_parameterization);

        // set intrinsics constant
        if (!options.optimize_intrinsics) {
          problem.SetParameterBlockConstant(intr->data());
        }
        // fix certain cameras
        if (fixed_cameras.find(tcid) != fixed_cameras.end()) {
          problem.SetParameterBlockConstant(T_w_c->data());
        }
      }
    }
    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = options.max_num_iterations;
    ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);
    switch (options.verbosity_level) {
      // 0: silent
      case 1:
        std::cout << summary.BriefReport() << std::endl;
        break;
      case 2:
        std::cout << summary.FullReport() << std::endl;
        break;
    }
  }

  }  // namespace visnav
