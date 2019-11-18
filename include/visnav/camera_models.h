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

#include <cassert>
#include <memory>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {
namespace helpers {
template <typename Scalar>
Scalar pow(const Scalar& num, int exp) {
  Scalar result = num;
  for (int i = 0; i < exp; i++) {
    result *= num;
  }
  return result;
}

}  // namespace helpers
template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() { param.setZero(); }

  PinholeCamera(const VecN& p) { param = p; }

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);

    const Scalar& x = Scalar(p[0]);
    const Scalar& y = Scalar(p[1]);
    const Scalar& z = Scalar(p[2]);

    Vec2 res;

    assert(z > Scalar(0));
    res << fx * (x / z) + cx, fy * (y / z) + cy;
    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);

    const Scalar& u = Scalar(p[0]);
    const Scalar& v = Scalar(p[1]);

    Vec3 res;

    Scalar mx = (u - cx) / fx;
    Scalar my = (v - cy) / fy;
    Scalar length = ceres::sqrt(mx * mx + my * my + Scalar(1));
    res << mx, my, Scalar(1);
    res /= length;
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  // NOTE: For convenience for serialization and handling different camera
  // models in ceres functors, we use the same parameter vector size for all of
  // them, even if that means that for some certain entries are unused /
  // constant 0.
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() { param.setZero(); }

  ExtendedUnifiedCamera(const VecN& p) { param = p; }

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);
    const Scalar& alpha = Scalar(param[4]);
    const Scalar& beta = Scalar(param[5]);

    const Scalar& x = Scalar(p[0]);
    const Scalar& y = Scalar(p[1]);
    const Scalar& z = Scalar(p[2]);

    Vec2 res;

    Scalar d = ceres::sqrt(beta * (x * x + y * y) + z * z);
    Scalar denominator = alpha * d + (Scalar(1) - alpha) * z;
    res << fx * (x / denominator) + cx, fy * (y / denominator) + cy;

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);
    const Scalar& alpha = Scalar(param[4]);
    const Scalar& beta = Scalar(param[5]);

    const Scalar& u = Scalar(p[0]);
    const Scalar& v = Scalar(p[1]);

    Vec3 res;

    Scalar mx = (u - cx) / fx;
    Scalar my = (v - cy) / fy;
    Scalar rSquared = mx * mx + my * my;
    Scalar mz =
        (Scalar(1) - beta * alpha * alpha * rSquared) /
        (alpha * ceres::sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) *
                                             beta * rSquared) +
         (Scalar(1) - alpha));
    res << mx, my, mz;
    return res.normalized();
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() { param.setZero(); }

  DoubleSphereCamera(const VecN& p) { param = p; }

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0,
        0;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);
    const Scalar& xi = Scalar(param[4]);
    const Scalar& alpha = Scalar(param[5]);

    const Scalar& x = Scalar(p[0]);
    const Scalar& y = Scalar(p[1]);
    const Scalar& z = Scalar(p[2]);

    Vec2 res;

    Scalar d1 = ceres::sqrt(x * x + y * y + z * z);
    Scalar d2 = ceres::sqrt(x * x + y * y + (xi * d1 + z) * (xi * d1 + z));
    Scalar denominator = alpha * d2 + (Scalar(1) - alpha) * (xi * d1 + z);
    res << fx * x / denominator + cx, fy * y / denominator + cy;

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = Scalar(param[0]);
    const Scalar& fy = Scalar(param[1]);
    const Scalar& cx = Scalar(param[2]);
    const Scalar& cy = Scalar(param[3]);
    const Scalar& xi = Scalar(param[4]);
    const Scalar& alpha = Scalar(param[5]);

    const Scalar& u = Scalar(p[0]);
    const Scalar& v = Scalar(p[1]);
    Vec3 res;

    Scalar mx = (u - cx) / fx;
    Scalar my = (v - cy) / fy;
    Scalar rSquared = mx * mx + my * my;
    Scalar mz =
        (Scalar(1) - alpha * alpha * rSquared) /
        (alpha * ceres::sqrt(Scalar(1) -
                             (Scalar(2) * alpha - Scalar(1)) * rSquared) +
         Scalar(1) - alpha);
    Scalar scalar =
        (mz * xi + ceres::sqrt(mz * mz + (Scalar(1) - xi * xi) * rSquared)) /
        (mz * mz + rSquared);
    res << scalar * mx, scalar * my, scalar * mz - xi;

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() { param.setZero(); }

  KannalaBrandt4Camera(const VecN& p) { param = p; }

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    Scalar r = ceres::sqrt(x * x + y * y);
    Scalar theta = ceres::atan2(r, z);
    Scalar theta2 = theta * theta;
    Scalar theta4 = theta2 * theta2;
    Scalar theta6 = theta2 * theta4;
    Scalar theta8 = theta4 * theta4;
    Scalar d = theta * (Scalar(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 +
                        k4 * theta8);

    if (r > 1e-8) {
      res << fx * d * x / r + cx, fy * d * y / r + cy;
    } else {
      res << fx * x / z + cx, fy * y / z + cy;
    }

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& u = p[0];
    const Scalar& v = p[1];

    Vec3 res;

    Scalar mx = (u - cx) / fx;
    Scalar my = (v - cy) / fy;
    Scalar ru = ceres::sqrt(mx * mx + my * my);

    // Approximation of theta via Newton's method

    Scalar theta = ru;  // initial guess
    Scalar theta2 = theta * theta;
    Scalar theta4 = theta2 * theta2;
    Scalar theta6 = theta2 * theta4;
    Scalar theta8 = theta4 * theta4;
    Scalar d, dDeriv;
    for (int i = 0; i < 3; i++) {
      theta2 = theta * theta;
      theta4 = theta2 * theta2;
      theta6 = theta2 * theta4;
      theta8 = theta4 * theta4;
      d = theta * (Scalar(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 +
                   k4 * theta8) -
          ru;
      dDeriv = Scalar(1) + Scalar(3) * k1 * theta2 + Scalar(5) * k2 * theta4 +
               Scalar(7) * k3 * theta6 + Scalar(9) * k4 * theta8;
      theta -= (d / dDeriv);
    }

    if (ru > 1e-8) {
      res << ceres::sin(theta) * mx / ru, ceres::sin(theta) * my / ru,
          ceres::cos(theta);
    } else {
      res << mx, my, ceres::cos(theta);
    }

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }
};

}  // namespace visnav
