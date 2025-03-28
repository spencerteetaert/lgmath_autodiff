/**
 * \file CommonMath.hpp
 * \brief Header file for some common math functions
 * \details defines some constants, angle-based functions, and comparison
 * functions.
 *
 * \author Sean Anderson, ASRL (autodiff support by Spencer Teetaert) 
 */
#pragma once

#include <Eigen/Core>

namespace lgmath {

/** Define various pi functions -- found from matlab w long precision */
namespace constants {

const double TWO_PI = 6.283185307179586;          // 2*pi
const double PI = 3.141592653589793;              // pi
const double PI_DIV_TWO = 1.570796326794897;      // pi/2
const double PI_DIV_FOUR = 0.785398163397448;     // pi/4
const double ONE_DIV_PI = 0.318309886183791;      // 1/pi
const double ONE_DIV_TWO_PI = 0.159154943091895;  // 1/2*pi
const double DEG2RAD = 0.017453292519943;         // pi/180
const double RAD2DEG = 57.295779513082323;        // 180/pi

}  // namespace constants

/// Common math functions
namespace common {

/** \brief moves a radian value to the range -pi, pi */
double angleMod(double radians);

/** \brief converts from degrees to radians */
double deg2rad(double degrees);

/** \brief converts from radians to degrees */
double rad2deg(double radians);

/** \brief compares if doubles are near equal */
bool nearEqual(double a, double b, double tol = 1e-6);

/** \brief compares if (double) Eigen matrices are near equal */
bool nearEqual(Eigen::MatrixXd A, Eigen::MatrixXd B, double tol = 1e-6);

/** \brief compares if radian angles are near equal */
bool nearEqualAngle(double radA, double radB, double tol = 1e-6);

/** \brief compares if axis angles are near equal */
bool nearEqualAxisAngle(Eigen::Matrix<double, 3, 1> aaxis1,
                        Eigen::Matrix<double, 3, 1> aaxis2, double tol = 1e-6);

/** \brief compares if lie algebra is near equal */
bool nearEqualLieAlg(Eigen::Matrix<double, 6, 1> vec1,
                     Eigen::Matrix<double, 6, 1> vec2, double tol = 1e-6);
}  // namespace common
}  // namespace lgmath

#ifdef USE_AUTODIFF
#ifdef USE_AUTODIFF_BACKWARD

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using autodiff::var;

namespace lgmath {
namespace common {
bool nearEqual(var a, var b, double tol);

bool nearEqualAngle(var radA, var radB, double tol);

template <typename Derived>
std::enable_if_t<std::is_same<typename Derived::Scalar, var>::value, bool>
nearEqual(const Eigen::MatrixBase<Derived>& A,
          const Eigen::MatrixBase<Derived>& B, double tol) {
  return common::nearEqual(A.template cast<double>(), B.template cast<double>(),
                           tol);
}

template <typename Derived>
std::enable_if_t<std::is_same<typename Derived::Scalar, var>::value, bool>
nearEqualAxisAngle(const Eigen::MatrixBase<Derived>& aaxis1,
                   const Eigen::MatrixBase<Derived>& aaxis2, double tol) {
  return common::nearEqualAxisAngle(aaxis1.template cast<double>(),
                                    aaxis2.template cast<double>(), tol);
}

template <typename Derived>
std::enable_if_t<std::is_same<typename Derived::Scalar, var>::value, bool>
nearEqualLieAlg(const Eigen::MatrixBase<Derived>& vec1,
                const Eigen::MatrixBase<Derived>& vec2, double tol) {
  return common::nearEqualLieAlg(vec1.template cast<double>(),
                                 vec2.template cast<double>(), tol);
}
}  // namespace common
}  // namespace lgmath
namespace autodiff {
namespace reverse {
namespace detail {
template <typename T>
AUTODIFF_DEVICE_FUNC bool isfinite(const Variable<T>& x) {
  return std::isfinite(double(x));
}

template <typename T>
AUTODIFF_DEVICE_FUNC bool isinf(const Variable<T>& x) {
  return std::isinf(double(x));
}

template <typename T>
AUTODIFF_DEVICE_FUNC bool isnan(const Variable<T>& x) {
  return std::isnan(double(x));
}

template <typename T>
AUTODIFF_DEVICE_FUNC Variable<T> fabs(const Variable<T>& x) {
  if (x >= 0) {
    return x;
  } else {
    return -x;
  }
}

template <typename T>
AUTODIFF_DEVICE_FUNC ExprPtr<T> fabs(const ExprPtr<T>& x) {
  if (x >= 0) {
    return x;
  } else {
    return -x;
  }
}

// template <typename T>
// AUTODIFF_DEVICE_FUNC ExprPtr<T> fabs(const ExprPtr<T>& x) {
//   if (x >= 0) {
//     return x;
//   } else {
//     return -x;
//   }
// }
}  // namespace detail
}  // namespace reverse
}  // namespace autodiff

#else
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace lgmath {
namespace common {
inline bool nearEqual(autodiff::real1st a, autodiff::real1st b, double tol) {
  return std::fabs(double(a) - double(b)) <= tol;
}

inline bool nearEqualAngle(autodiff::real1st radA, autodiff::real1st radB,
                    double tol) {
  return common::nearEqual(common::angleMod(double(radA) - double(radB)), 0.0,
                           tol);
}

template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, autodiff::real1st>::value, bool>
nearEqual(const Eigen::MatrixBase<Derived>& A,
          const Eigen::MatrixBase<Derived>& B, double tol) {
  return common::nearEqual(A.template cast<double>(), B.template cast<double>(),
                           tol);
}

template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, autodiff::real1st>::value, bool>
nearEqualAxisAngle(const Eigen::MatrixBase<Derived>& aaxis1,
                   const Eigen::MatrixBase<Derived>& aaxis2, double tol) {
  return common::nearEqualAxisAngle(aaxis1.template cast<double>(),
                                    aaxis2.template cast<double>(), tol);
}

template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, autodiff::real1st>::value, bool>
nearEqualLieAlg(const Eigen::MatrixBase<Derived>& vec1,
                const Eigen::MatrixBase<Derived>& vec2, double tol) {
  return common::nearEqualLieAlg(vec1.template cast<double>(),
                                 vec2.template cast<double>(), tol);
}
}  // namespace common
}  // namespace lgmath

namespace autodiff {
namespace detail {
template <size_t N, typename T>
AUTODIFF_DEVICE_FUNC bool isfinite(const Real<N, T>& x) {
  return std::isfinite(double(x));
}

template <size_t N, typename T>
AUTODIFF_DEVICE_FUNC bool isinf(const Real<N, T>& x) {
  return std::isinf(double(x));
}

template <size_t N, typename T>
AUTODIFF_DEVICE_FUNC bool isnan(const Real<N, T>& x) {
  return std::isnan(double(x));
}

template <size_t N, typename T>
AUTODIFF_DEVICE_FUNC Real<N, T> fabs(const Real<N, T>& x) {
  if (x.val() >= 0) {
    return x;
  } else {
    return -x;
  }
}
}  // namespace detail
}  // namespace autodiff

#endif
#endif
