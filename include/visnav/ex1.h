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

#include <math.h>

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

namespace helpers {
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

template <class T>
Eigen::Matrix<T, 3, 3> get_jacobian(const Eigen::Matrix<T, 3, 1>& phi) {
  T length = phi.norm();
  auto normalized = phi;  // preserve the original phi for later use
  normalized.normalize();
  auto identity = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> jacobian;

  if (length == 0.0) {  // dividing by zero will result in nans -> use Taylor
                        // expansion approximations
    T sin_term =
        1 - (length * length / 6) *
                (1 - (length * length / 20) * (1 - (length * length / 42)));
    T cos_term =
        (length / 2) *
        (1 - (length * length / 12) *
                 (1 - (length * length / 30) * (1 - (length * length / 56))));
    jacobian << sin_term * identity +
                    (1 - sin_term) * (normalized * normalized.transpose()) +
                    cos_term * get_skew_symmetric_matrix(normalized);
  } else {
    jacobian =
        (sin(length) / length) * identity +
        (1 - (sin(length) / length)) * (normalized * normalized.transpose()) +
        ((1 - cos(length)) / length) * get_skew_symmetric_matrix(normalized);
  }
  return jacobian;
}
template <class T>
Eigen::Matrix<T, 3, 3> get_jacobian_inverse(const Eigen::Matrix<T, 3, 1>& phi) {
  T length = phi.norm();
  auto normalized = phi;  // preserve the original phi for later use
  normalized.normalize();
  auto identity = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> jacobian_inverse;
  auto skew_symmetric_phi = get_skew_symmetric_matrix(phi);
  T approximation_term;
  if (length == 0.0) {  // use taylor expansion
    std::cout << "using taylor approx" << std::endl;
    approximation_term = 0;
  } else {
    approximation_term = 1 / (length * length) *
                         (1 - (length * sin(length) / (2 * (1 - cos(length)))));
  }
  jacobian_inverse =
      identity - 0.5 * skew_symmetric_phi +
      approximation_term * (skew_symmetric_phi * skew_symmetric_phi);

  return jacobian_inverse;
}
}  // namespace helpers

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& phi) {
  T length = phi.norm();
  auto normalized = phi;
  normalized.normalize();
  auto identity = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> result =
      cos(length) * identity +
      (1 - cos(length)) * (normalized * normalized.transpose()) +
      sin(length) * helpers::get_skew_symmetric_matrix(normalized);
  return result;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  Eigen::Matrix<T, 3, 1> result;
  T theta = acos(0.5 * (mat.trace() - 1));
  if (theta == 0.0) {  // dividing by zero will result in nans
    result << 0, 0, 0;
    return result;
  }
  Eigen::Matrix<T, 3, 3> log =
      (theta / (2 * sin(theta))) * (mat - mat.transpose());
  result = helpers::get_vector_from_skew_symmetric_matrix(log);
  return result;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  Eigen::Matrix<T, 3, 1> rho, phi;
  Eigen::Matrix<T, 4, 4> result;
  rho = xi.head(3);
  phi = xi.tail(3);
  Eigen::Matrix<T, 3, 3> exp_phi = user_implemented_expmap(phi);
  Eigen::Matrix<T, 3, 3> jacobian = helpers::get_jacobian(phi);
  result.template block<3, 3>(0, 0) = exp_phi;
  result.template block<3, 1>(0, 3) = jacobian * rho;
  result.row(3) = Eigen::Matrix<T, 1, 4>(0, 0, 0, 1);
  return result;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  Eigen::Matrix<T, 6, 1> result;
  Eigen::Matrix<T, 3, 3> rotation = mat.template block<3, 3>(0, 0);
  Eigen::Matrix<T, 3, 1> translation = mat.template block<3, 1>(0, 3);
  Eigen::Matrix<T, 3, 1> phi = user_implemented_logmap(rotation);
  Eigen::Matrix<T, 3, 3> jacobian_inverse = helpers::get_jacobian_inverse(phi);
  Eigen::Matrix<T, 3, 1> rho = jacobian_inverse * translation;
  result << rho, phi;
  return result;
}

}  // namespace visnav
