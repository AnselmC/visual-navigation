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

#include<math.h>

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Helpers for skewsymmetric operations
template <class T>
Eigen::Matrix<T, 3, 3> get_skew_symmetric_matrix(
    const Eigen::Matrix<T, 3, 1>& vector) {
  auto result = Eigen::Matrix<T, 3, 3>();
  result << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
      vector(0), 0;
  return result;
}

template <class T>
Eigen::Matrix<T, 3, 1> get_vector_from_skew_symmetric_matrix(
    const Eigen::Matrix<T, 3, 3>& mat) {
  auto result = Eigen::Matrix<T, 3, 1>();
  result << -mat(1, 2), mat(0, 2), -mat(0, 1);
  return result;
}

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& phi) {
  // TODO SHEET 1: implement
  double length = phi.norm();
  auto normalized = phi;
  normalized.normalize();
  auto identity = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> result =
      cos(length) * identity +
      (1 - cos(length)) * (normalized * normalized.transpose()) +
      sin(length) * get_skew_symmetric_matrix(normalized);
  return result;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> result;
  double theta = acos(0.5 * (mat.trace() - 1));
  if (theta == 0.0) {
    result << 0, 0, 0;
    return result;
  }
  Eigen::Matrix<T, 3, 3> log =
      (theta / (2 * sin(theta))) * (mat - mat.transpose());
  result = get_vector_from_skew_symmetric_matrix(log);
  return result;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> rho, phi;
  rho << xi(0), xi(1), xi(2);
  phi << xi(3), xi(4), xi(5);
  Eigen::Matrix<T, 3, 3> exp_phi = user_implemented_expmap(phi);
  double length = phi.norm();
  auto normalized = phi;
  normalized.normalize();
  auto identity = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> jacobian;

  if (length == 0.0) {
    jacobian << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  } else {
    jacobian =
        (sin(length) / length) * identity +
        (1 - (sin(length) / length)) * (normalized * normalized.transpose()) +
        ((1 - cos(length)) / length) * get_skew_symmetric_matrix(normalized);
  }

  Eigen::Matrix<T, 4, 4> result;
  result.template block<3, 3>(0, 0) = exp_phi;
  result.template block<3, 1>(0, 3) = jacobian * rho;
  result.row(3) = Eigen::Matrix<T, 1, 4>(0, 0, 0, 0);
  result(3, 3) = 1;
  return result;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  UNUSED(mat);
  return Eigen::Matrix<T, 6, 1>();
  }

}  // namespace visnav
