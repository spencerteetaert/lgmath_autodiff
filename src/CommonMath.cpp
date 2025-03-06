/**
 * \file CommonMath.cpp
 * \brief Implementation file for some common math functions
 * \details defines some constants, angle-based functions, and comparison
 * functions.
 *
 * \author Sean Anderson, ASRL
 */
#include <lgmath/CommonMath.hpp>

#include <math.h>

#if INCLUDE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#endif 

namespace lgmath {
namespace common {

double angleMod(double radians) {
  return (double)(radians - (constants::TWO_PI *
                             rint(radians * constants::ONE_DIV_TWO_PI)));
}

double deg2rad(double degrees) {
  return (double)(degrees * constants::DEG2RAD);
}

double rad2deg(double radians) {
  return (double)(radians * constants::RAD2DEG);
}

bool nearEqual(double a, double b, double tol) { return fabs(a - b) <= tol; }

bool nearEqual(Eigen::MatrixXd A, Eigen::MatrixXd B, double tol) {
  bool near = true;
  near = near & (A.cols() == B.cols());
  near = near & (A.rows() == B.rows());
  for (int j = 0; j < A.cols(); j++) {
    for (int i = 0; i < A.rows(); i++) {
      near = near & nearEqual(A(i, j), B(i, j), tol);
    }
  }
  return near;
}

bool nearEqualAngle(double radA, double radB, double tol) {
  return nearEqual(angleMod(radA - radB), 0.0, tol);
}

bool nearEqualAxisAngle(Eigen::Matrix<double, 3, 1> aaxis1,
                        Eigen::Matrix<double, 3, 1> aaxis2, double tol) {
  bool near = true;

  // get angles
  double a1 = aaxis1.norm();
  double a2 = aaxis2.norm();

  // check if both angles are near zero
  if (fabs(a1) < tol && fabs(a2) < tol) {
    return true;
  } else {  // otherwise, compare normalized axis

    // compare each element of axis
    Eigen::Matrix<double, 3, 1> axis1 = aaxis1 / a1;
    Eigen::Matrix<double, 3, 1> axis2 = aaxis1 / a2;
    for (int i = 0; i < 3; i++) {
      near = near & nearEqual(axis1(i), axis2(i), tol);
    }

    // compare wrapped angles
    near = near & nearEqualAngle(a1, a2, tol);
    return near;
  }
}

bool nearEqualLieAlg(Eigen::Matrix<double, 6, 1> vec1,
                     Eigen::Matrix<double, 6, 1> vec2, double tol) {
  bool near = true;
  near = near & nearEqualAxisAngle(vec1.tail<3>(), vec2.tail<3>(), tol);
  near = near & nearEqual(vec1.head<3>(), vec2.head<3>(), tol);
  return near;
}

#if INCLUDE_AUTODIFF

namespace diff {
  
bool nearEqual(autodiff::real a, autodiff::real b, double tol) {
  return std::fabs(a.val() - b.val()) <= tol;
}

bool nearEqual(autodiff::MatrixXreal A, autodiff::MatrixXreal B, double tol) {
  Eigen::MatrixXd A_d(A.rows(), A.cols()); 
  A_d = A.cast<double>();
  Eigen::MatrixXd B_d(B.rows(), B.cols()); 
  B_d = B.cast<double>();
  return common::nearEqual(A_d, B_d, tol);
}

bool nearEqualAngle(autodiff::real radA, autodiff::real radB, double tol) {
  return common::nearEqual(common::angleMod(radA.val() - radB.val()), 0.0, tol);
}

bool nearEqualAxisAngle(autodiff::Vector3real aaxis1,
                        autodiff::Vector3real aaxis2, double tol) {
  Eigen::Vector3d aaxis1_d = aaxis1.cast<double>();
  Eigen::Vector3d aaxis2_d = aaxis2.cast<double>();
  return common::nearEqualAxisAngle(aaxis1_d, aaxis2_d, tol);
}

bool nearEqualLieAlg(autodiff::VectorXreal vec1, autodiff::VectorXreal vec2,
                     double tol) {
  Eigen::VectorXd vec1_d(vec1.size()); 
  vec1_d = vec1.cast<double>();
  Eigen::VectorXd vec2_d(vec2.size()); 
  vec2_d = vec2.cast<double>();
  return common::nearEqualLieAlg(vec1_d, vec2_d, tol);
}
}  // namespace diff
#endif

}  // namespace common
}  // namespace lgmath
