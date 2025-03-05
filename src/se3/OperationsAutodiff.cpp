/**
 * \file Operations.cpp
 * \brief Implementation file for the SE3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#include <lgmath/se3/OperationsAutodiff.hpp>

#include <stdio.h>
#include <stdexcept>

#include <Eigen/Dense>

#include <lgmath/so3/OperationsAutodiff.hpp>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace lgmath {
namespace se3 {

namespace diff {

autodiff::Matrix4real hat(const autodiff::Vector3real& rho,
                          const autodiff::Vector3real& aaxis) {
  autodiff::Matrix4real mat = autodiff::Matrix4real::Zero();
  mat.topLeftCorner<3, 3>() = so3::diff::hat(aaxis);
  mat.topRightCorner<3, 1>() = rho;
  return mat;
}

autodiff::Matrix4real hat(const autodiff::VectorXreal& xi) {
  assert(xi.size() == 6);
  return hat(xi.head<3>(), xi.tail<3>());
}

autodiff::MatrixXreal curlyhat(const autodiff::Vector3real& rho,
                               const autodiff::Vector3real& aaxis) {
  autodiff::MatrixXreal mat(6, 6);
  mat.setZero();
  mat.topLeftCorner<3, 3>() = mat.bottomRightCorner<3, 3>() =
      so3::diff::hat(aaxis);
  mat.topRightCorner<3, 3>() = so3::diff::hat(rho);
  return mat;
}

autodiff::MatrixXreal curlyhat(const autodiff::VectorXreal& xi) {
  assert(xi.size() == 6);
  return curlyhat(xi.head<3>(), xi.tail<3>());
}

// Eigen::Matrix<double, 4, 6> point2fs(const autodiff::Vector3real& p, double
// scale)
// {
//   Eigen::Matrix<double, 4, 6> mat = Eigen::Matrix<double, 4, 6>::Zero();
//   mat.topLeftCorner<3, 3>() = scale * autodiff::Matrix3real::Identity();
//   mat.topRightCorner<3, 3>() = -so3::diff::hat(p);
//   return mat;
// }

// Eigen::Matrix<double, 6, 4> point2sf(const autodiff::Vector3real& p, double
// scale)
// {
//   Eigen::Matrix<double, 6, 4> mat = Eigen::Matrix<double, 6, 4>::Zero();
//   mat.bottomLeftCorner<3, 3>() = -so3::diff::hat(p);
//   mat.topRightCorner<3, 1>() = p;
//   return mat;
// }

void vec2tran_analytical(const autodiff::Vector3real& rho_ba,
                         const autodiff::Vector3real& aaxis_ba,
                         autodiff::Matrix3real* out_C_ab,
                         autodiff::Vector3real* out_r_ba_ina) {
  // Check pointers
  if (out_C_ab == NULL) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2tran_analytical");
  }
  if (out_r_ba_ina == NULL) {
    throw std::invalid_argument(
        "Null pointer out_r_ba_ina in vec2tran_analytical");
  }

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, rotation is Identity
    *out_C_ab = autodiff::Matrix3real::Identity();
    *out_r_ba_ina = rho_ba;
  } else {
    // Normal analytical solution
    autodiff::Matrix3real J_ab;

    // Use rotation identity involving jacobian, as we need it to
    // convert rho_ba to the proper translation
    so3::diff::vec2rot(aaxis_ba, out_C_ab, &J_ab);

    // Convert rho_ba (twist-translation) to r_ba_ina
    *out_r_ba_ina = J_ab * rho_ba;
  }
}

void vec2tran_numerical(const autodiff::Vector3real& rho_ba,
                        const autodiff::Vector3real& aaxis_ba,
                        autodiff::Matrix3real* out_C_ab,
                        autodiff::Vector3real* out_r_ba_ina,
                        unsigned int numTerms) {
  // Check pointers
  if (out_C_ab == NULL) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2tran_numerical");
  }
  if (out_r_ba_ina == NULL) {
    throw std::invalid_argument(
        "Null pointer out_r_ba_ina in vec2tran_numerical");
  }

  // Init 4x4 transformation
  autodiff::Matrix4real T_ab = autodiff::Matrix4real::Identity();

  // Incremental variables
  autodiff::VectorXreal xi_ba(6);
  xi_ba << rho_ba, aaxis_ba;
  autodiff::Matrix4real x_small = se3::diff::hat(xi_ba);
  autodiff::Matrix4real x_small_n = autodiff::Matrix4real::Identity();

  // Loop over sum up to the specified numTerms
  for (unsigned int n = 1; n <= numTerms; n++) {
    x_small_n = x_small_n * x_small / double(n);
    T_ab += x_small_n;
  }

  // Fill output
  *out_C_ab = T_ab.topLeftCorner<3, 3>();
  *out_r_ba_ina = T_ab.topRightCorner<3, 1>();
}

void vec2tran(const autodiff::VectorXreal& xi_ba,
              autodiff::Matrix3real* out_C_ab,
              autodiff::Vector3real* out_r_ba_ina, unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  if (numTerms == 0) {
    // Analytical solution
    vec2tran_analytical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab,
                        out_r_ba_ina);
  } else {
    // Numerical solution (good for testing the analytical solution)
    vec2tran_numerical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab, out_r_ba_ina,
                       numTerms);
  }
}

autodiff::Matrix4real vec2tran(const autodiff::VectorXreal& xi_ba,
                               unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  // Get rotation and translation
  autodiff::Matrix3real C_ab;
  autodiff::Vector3real r_ba_ina;
  vec2tran(xi_ba, &C_ab, &r_ba_ina, numTerms);

  // Fill output
  autodiff::Matrix4real T_ab = autodiff::Matrix4real::Identity();
  T_ab.topLeftCorner<3, 3>() = C_ab;
  T_ab.topRightCorner<3, 1>() = r_ba_ina;
  return T_ab;
}

autodiff::VectorXreal tran2vec(const autodiff::Matrix3real& C_ab,
                               const autodiff::Vector3real& r_ba_ina) {
  // Init
  autodiff::VectorXreal xi_ba(6);

  // Get axis angle from rotation matrix
  autodiff::Vector3real aaxis_ba = so3::diff::rot2vec(C_ab);

  // Get twist-translation vector using Jacobian
  autodiff::Vector3real rho_ba = so3::diff::vec2jacinv(aaxis_ba) * r_ba_ina;

  // Return se3 algebra vector
  xi_ba << rho_ba, aaxis_ba;
  return xi_ba;
}

autodiff::VectorXreal tran2vec(const autodiff::Matrix4real& T_ab) {
  return tran2vec(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

autodiff::MatrixXreal tranAd(const autodiff::Matrix3real& C_ab,
                             const autodiff::Vector3real& r_ba_ina) {
  autodiff::MatrixXreal adjoint_T_ab(6, 6);
  adjoint_T_ab.setZero();

  adjoint_T_ab.topLeftCorner<3, 3>() = adjoint_T_ab.bottomRightCorner<3, 3>() =
      C_ab;
  adjoint_T_ab.topRightCorner<3, 3>() = so3::diff::hat(r_ba_ina) * C_ab;
  return adjoint_T_ab;
}

autodiff::MatrixXreal tranAd(const autodiff::Matrix4real& T_ab) {
  return tranAd(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

autodiff::Matrix3real vec2Q(const autodiff::Vector3real& rho_ba,
                            const autodiff::Vector3real& aaxis_ba) {
  // Construct scalar terms
  const autodiff::real ang = aaxis_ba.norm();
  const autodiff::real ang2 = ang * ang;
  const autodiff::real ang3 = ang2 * ang;
  const autodiff::real ang4 = ang3 * ang;
  const autodiff::real ang5 = ang4 * ang;
  const autodiff::real cang = cos(ang);
  const autodiff::real sang = sin(ang);
  const autodiff::real m2 = (ang - sang) / ang3;
  const autodiff::real m3 = (1.0 - 0.5 * ang2 - cang) / ang4;
  const autodiff::real m4 = 0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5);

  // Construct matrix terms
  autodiff::Matrix3real rx = so3::diff::hat(rho_ba);
  autodiff::Matrix3real px = so3::diff::hat(aaxis_ba);
  autodiff::Matrix3real pxrx = px * rx;
  autodiff::Matrix3real rxpx = rx * px;
  autodiff::Matrix3real pxrxpx = pxrx * px;

  // Construct Q matrix
  return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) -
         m3 * (px * pxrx + rxpx * px - 3 * pxrxpx) -
         m4 * (pxrxpx * px + px * pxrxpx);
}

autodiff::Matrix3real vec2Q(const autodiff::VectorXreal& xi_ba) {
  assert(xi_ba.size() == 6);
  return vec2Q(xi_ba.head<3>(), xi_ba.tail<3>());
}

autodiff::MatrixXreal vec2jac(const autodiff::Vector3real& rho_ba,
                              const autodiff::Vector3real& aaxis_ba) {
  // Init
  autodiff::MatrixXreal J_ab(6, 6);
  J_ab.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        autodiff::Matrix3real::Identity();
    J_ab.topRightCorner<3, 3>() = 0.5 * so3::diff::hat(rho_ba);
  } else {
    // General analytical scenario
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        so3::diff::vec2jac(aaxis_ba);
    J_ab.topRightCorner<3, 3>() = se3::diff::vec2Q(rho_ba, aaxis_ba);
  }
  return J_ab;
}

autodiff::MatrixXreal vec2jac(const autodiff::VectorXreal& xi_ba,
                              unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  if (numTerms == 0) {
    // Analytical solution
    return vec2jac(xi_ba.head<3>(), xi_ba.tail<3>());
  } else {
    // Numerical solution (good for testing the analytical solution)
    autodiff::MatrixXreal J_ab(6, 6);
    J_ab.setIdentity();

    // Incremental variables
    autodiff::MatrixXreal x_small = se3::diff::curlyhat(xi_ba);
    autodiff::MatrixXreal x_small_n(6, 6);
    x_small_n.setIdentity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
    return J_ab;
  }
}

autodiff::MatrixXreal vec2jacinv(const autodiff::Vector3real& rho_ba,
                                 const autodiff::Vector3real& aaxis_ba) {
  // Init
  autodiff::MatrixXreal J66_ab_inv(6, 6);
  J66_ab_inv.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        autodiff::Matrix3real::Identity();
    J66_ab_inv.topRightCorner<3, 3>() = -0.5 * so3::diff::hat(rho_ba);
  } else {
    // General analytical scenario
    autodiff::Matrix3real J33_ab_inv = so3::diff::vec2jacinv(aaxis_ba);
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        J33_ab_inv;
    J66_ab_inv.topRightCorner<3, 3>() =
        -J33_ab_inv * se3::diff::vec2Q(rho_ba, aaxis_ba) * J33_ab_inv;
  }
  return J66_ab_inv;
}

autodiff::MatrixXreal vec2jacinv(const autodiff::VectorXreal& xi_ba,
                                 unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  if (numTerms == 0) {
    // Analytical solution
    return vec2jacinv(xi_ba.head<3>(), xi_ba.tail<3>());
  } else {
    // Logic error
    if (numTerms > 20) {
      throw std::invalid_argument(
          "Numerical vec2jacinv does not support numTerms > 20");
    }

    // Numerical solution (good for testing the analytical solution)
    autodiff::MatrixXreal J_ab(6, 6); 
    J_ab.setIdentity();

    // Incremental variables
    autodiff::MatrixXreal x_small = se3::diff::curlyhat(xi_ba);
    autodiff::MatrixXreal x_small_n(6, 6); 
    x_small_n.setIdentity();
    
    // Boost has a bernoulli package... but we shouldn't need more than 20
    Eigen::Matrix<double, 21, 1> bernoulli;
    bernoulli << 1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
        -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
        -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0;

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n);
      J_ab += bernoulli(n) * x_small_n;
    }
    return J_ab;
  }
}

}  // namespace diff
}  // namespace se3
}  // namespace lgmath
