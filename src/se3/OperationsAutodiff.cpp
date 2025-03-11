#if USE_AUTODIFF
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

#ifdef AUTODIFF_USE_FORWARD 
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#else 
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#endif

namespace lgmath {
namespace se3 {
namespace diff {

Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> hat(const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho,
                                  const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis) {
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> mat = Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>::Zero();
  mat.topLeftCorner<3, 3>() = so3::diff::hat(aaxis);
  mat.topRightCorner<3, 1>() = rho;
  return mat;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> hat(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi) {
  assert(xi.size() == 6);
  return hat(xi.head<3>(), xi.tail<3>());
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> curlyhat(
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis) {
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> mat(6, 6);
  mat.setZero();
  mat.topLeftCorner<3, 3>() = mat.bottomRightCorner<3, 3>() =
      so3::diff::hat(aaxis);
  mat.topRightCorner<3, 3>() = so3::diff::hat(rho);
  return mat;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> curlyhat(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi) {
  assert(xi.size() == 6);
  return curlyhat(xi.head<3>(), xi.tail<3>());
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 6> point2fs(const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& p,
                                       double scale) {
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 6> mat = Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 6>::Zero();
  mat.topLeftCorner<3, 3>() = scale * Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>::Identity();
  mat.topRightCorner<3, 3>() = -so3::diff::hat(p);
  return mat;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 4> point2sf(const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& p,
                                       double scale) {
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 4> mat = Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 4>::Zero();
  mat.bottomLeftCorner<3, 3>() = -so3::diff::hat(p);
  mat.topRightCorner<3, 1>() = p;
  return mat;
}

void vec2tran_analytical(const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho_ba,
                         const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis_ba,
                         Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>* out_C_ab,
                         Eigen::Vector<AUTODIFF_VAR_TYPE, 3>* out_r_ba_ina) {
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
    *out_C_ab = Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>::Identity();
    *out_r_ba_ina = rho_ba;
  } else {
    // Normal analytical solution
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> J_ab;

    // Use rotation identity involving jacobian, as we need it to
    // convert rho_ba to the proper translation
    so3::diff::vec2rot(aaxis_ba, out_C_ab, &J_ab);

    // Convert rho_ba (twist-translation) to r_ba_ina
    *out_r_ba_ina = J_ab * rho_ba;
  }
}

void vec2tran_numerical(const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho_ba,
                        const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis_ba,
                        Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>* out_C_ab,
                        Eigen::Vector<AUTODIFF_VAR_TYPE, 3>* out_r_ba_ina,
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
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> T_ab =
      Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>::Identity();

  // Incremental variables
  Eigen::Vector<AUTODIFF_VAR_TYPE, 6> xi_ba(6);
  xi_ba << rho_ba, aaxis_ba;
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> x_small = se3::diff::hat(xi_ba);
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> x_small_n =
      Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>::Identity();

  // Loop over sum up to the specified numTerms
  for (unsigned int n = 1; n <= numTerms; n++) {
    x_small_n = x_small_n * x_small / double(n);
    T_ab += x_small_n;
  }

  // Fill output
  *out_C_ab = T_ab.topLeftCorner<3, 3>();
  *out_r_ba_ina = T_ab.topRightCorner<3, 1>();
}

void vec2tran(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi_ba,
              Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>* out_C_ab,
              Eigen::Vector<AUTODIFF_VAR_TYPE, 3>* out_r_ba_ina, unsigned int numTerms) {
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

Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> vec2tran(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi_ba,
                                       unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  // Get rotation and translation
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> C_ab;
  Eigen::Vector<AUTODIFF_VAR_TYPE, 3> r_ba_ina;
  vec2tran(xi_ba, &C_ab, &r_ba_ina, numTerms);

  // Fill output
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4> T_ab =
      Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>::Identity();
  T_ab.topLeftCorner<3, 3>() = C_ab;
  T_ab.topRightCorner<3, 1>() = r_ba_ina;
  return T_ab;
}

Eigen::Vector<AUTODIFF_VAR_TYPE, 6> tran2vec(
    const Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>& C_ab,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& r_ba_ina) {
  // Init
  Eigen::Vector<AUTODIFF_VAR_TYPE, 6> xi_ba(6);

  // Get axis angle from rotation matrix
  Eigen::Vector<AUTODIFF_VAR_TYPE, 3> aaxis_ba = so3::diff::rot2vec(C_ab);

  // Get twist-translation vector using Jacobian
  Eigen::Vector<AUTODIFF_VAR_TYPE, 3> rho_ba =
      so3::diff::vec2jacinv(aaxis_ba) * r_ba_ina;

  // Return se3 algebra vector
  xi_ba << rho_ba, aaxis_ba;
  return xi_ba;
}

Eigen::Vector<AUTODIFF_VAR_TYPE, 6> tran2vec(const Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>& T_ab) {
  return tran2vec(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> tranAd(
    const Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>& C_ab,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& r_ba_ina) {
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> adjoint_T_ab(6, 6);
  adjoint_T_ab.setZero();

  adjoint_T_ab.topLeftCorner<3, 3>() = adjoint_T_ab.bottomRightCorner<3, 3>() =
      C_ab;
  adjoint_T_ab.topRightCorner<3, 3>() = so3::diff::hat(r_ba_ina) * C_ab;
  return adjoint_T_ab;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> tranAd(
    const Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>& T_ab) {
  return tranAd(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> vec2Q(
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho_ba,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis_ba) {
  // Construct scalar terms
  const AUTODIFF_VAR_TYPE ang = aaxis_ba.norm();
  const AUTODIFF_VAR_TYPE ang2 = ang * ang;
  const AUTODIFF_VAR_TYPE ang3 = ang2 * ang;
  const AUTODIFF_VAR_TYPE ang4 = ang3 * ang;
  const AUTODIFF_VAR_TYPE ang5 = ang4 * ang;
  const AUTODIFF_VAR_TYPE cang = cos(ang);
  const AUTODIFF_VAR_TYPE sang = sin(ang);
  const AUTODIFF_VAR_TYPE m2 = (ang - sang) / ang3;
  const AUTODIFF_VAR_TYPE m3 = (1.0 - 0.5 * ang2 - cang) / ang4;
  const AUTODIFF_VAR_TYPE m4 = 0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5);

  // Construct matrix terms
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> rx = so3::diff::hat(rho_ba);
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> px = so3::diff::hat(aaxis_ba);
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> pxrx = px * rx;
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> rxpx = rx * px;
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> pxrxpx = pxrx * px;

  // Construct Q matrix
  return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) -
         m3 * (px * pxrx + rxpx * px - 3 * pxrxpx) -
         m4 * (pxrxpx * px + px * pxrxpx);
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> vec2Q(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi_ba) {
  assert(xi_ba.size() == 6);
  return vec2Q(xi_ba.head<3>(), xi_ba.tail<3>());
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> vec2jac(
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho_ba,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis_ba) {
  // Init
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> J_ab(6, 6);
  J_ab.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>::Identity();
    J_ab.topRightCorner<3, 3>() = 0.5 * so3::diff::hat(rho_ba);
  } else {
    // General analytical scenario
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        so3::diff::vec2jac(aaxis_ba);
    J_ab.topRightCorner<3, 3>() = se3::diff::vec2Q(rho_ba, aaxis_ba);
  }
  return J_ab;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> vec2jac(const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi_ba,
                                      unsigned int numTerms) {
  assert(xi_ba.size() == 6);
  if (numTerms == 0) {
    // Analytical solution
    return vec2jac(xi_ba.head<3>(), xi_ba.tail<3>());
  } else {
    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> J_ab(6, 6);
    J_ab.setIdentity();

    // Incremental variables
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> x_small = se3::diff::curlyhat(xi_ba);
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> x_small_n(6, 6);
    x_small_n.setIdentity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
    return J_ab;
  }
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> vec2jacinv(
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& rho_ba,
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 3>& aaxis_ba) {
  // Init
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> J66_ab_inv(6, 6);
  J66_ab_inv.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>::Identity();
    J66_ab_inv.topRightCorner<3, 3>() = -0.5 * so3::diff::hat(rho_ba);
  } else {
    // General analytical scenario
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> J33_ab_inv = so3::diff::vec2jacinv(aaxis_ba);
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        J33_ab_inv;
    J66_ab_inv.topRightCorner<3, 3>() =
        -J33_ab_inv * se3::diff::vec2Q(rho_ba, aaxis_ba) * J33_ab_inv;
  }
  return J66_ab_inv;
}

Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> vec2jacinv(
    const Eigen::Vector<AUTODIFF_VAR_TYPE, 6>& xi_ba, unsigned int numTerms) {
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
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> J_ab(6, 6);
    J_ab.setIdentity();

    // Incremental variables
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> x_small = se3::diff::curlyhat(xi_ba);
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6> x_small_n(6, 6);
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
#endif