/**
 * \file CommonMath.hpp
 * \brief Header file for some common math functions
 * \details defines some constants, angle-based functions, and comparison
 * functions.
 *
 * \author Sean Anderson, ASRL
 */
#pragma once

#include <Eigen/Core>

#if USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#endif

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

#if USE_AUTODIFF
namespace diff {

bool nearEqual(autodiff::real a, autodiff::real b, double tol);

bool nearEqual(autodiff::MatrixXreal A, autodiff::MatrixXreal B, double tol);

bool nearEqualAngle(autodiff::real radA, autodiff::real radB, double tol);

bool nearEqualAxisAngle(autodiff::Vector3real aaxis1,
                        autodiff::Vector3real aaxis2, double tol);

bool nearEqualLieAlg(autodiff::VectorXreal vec1, autodiff::VectorXreal vec2,
                     double tol);
}  // namespace diff
#endif // USE_AUTODIFF

}  // namespace common
}  // namespace lgmath
