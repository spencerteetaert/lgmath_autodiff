#if USE_AUTODIFF
/**
 * \file Operations.hpp
 * \brief Header file for the SE3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#pragma once

#include <lgmath/CommonMath.hpp>
#include <lgmath/so3/OperationsAutodiff.hpp>

#include <Eigen/Core>
#ifdef AUTODIFF_USE_BACKWARD
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#ifndef AUTODIFF_VAR_TYPE
#define AUTODIFF_VAR_TYPE autodiff::var
#endif
#else
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#ifndef AUTODIFF_VAR_TYPE
#define AUTODIFF_VAR_TYPE autodiff::real1st
#endif
#endif

/// Lie Group Math - Special Euclidean Group
namespace lgmath {
namespace se3 {
/**
 * \brief Builds the 4x4 "skew symmetric matrix"
 * \details
 * The hat (^) operator, builds the 4x4 skew symmetric matrix from the 3x1 axis
 * angle vector and 3x1 translation vector.
 *
 * hat(rho, aaxis) = [aaxis^ rho] = [0.0  -a3   a2  rho1]
 *                   [  0^T    0]   [ a3  0.0  -a1  rho2]
 *                                  [-a2   a1  0.0  rho3]
 *                                  [0.0  0.0  0.0   0.0]
 *
 * See eq. 4 in Barfoot-TRO-2014 for more information.
 */

template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>>
hat(const Eigen::MatrixBase<Derived>& rho,
    const Eigen::MatrixBase<Derived>& aaxis) {
  assert(rho.cols() == 1 && rho.rows() == 3);
  assert(aaxis.cols() == 1 && aaxis.rows() == 3);
  Eigen::Matrix<typename Derived::Scalar, 4, 4> mat =
      Eigen::Matrix<typename Derived::Scalar, 4, 4>::Zero();
  mat.template topLeftCorner<3, 3>() = so3::hat(aaxis);
  mat.template topRightCorner<3, 1>() = rho;
  return mat;
}

/**
 * \brief Builds the 4x4 "skew symmetric matrix"
 * \details
 * The hat (^) operator, builds the 4x4 skew symmetric matrix from
 * the 6x1 se3 algebra vector, xi:
 *
 * xi^ = [rho  ] = [aaxis^ rho] = [0.0  -a3   a2  rho1]
 *       [aaxis]   [  0^T    0]   [ a3  0.0  -a1  rho2]
 *                                [-a2   a1  0.0  rho3]
 *                                [0.0  0.0  0.0   0.0]
 *
 * See eq. 4 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>>
hat(const Eigen::MatrixBase<Derived>& xi) {
  assert(xi.rows() == 6 && xi.cols() == 1);
  return se3::hat(xi.template head<3>(), xi.template tail<3>());
}

/**
 * \brief Builds the 6x6 "curly hat" matrix (related to the skew symmetric
 * matrix)
 * \details
 * The curly hat operator builds the 6x6 skew symmetric matrix from the 3x1 axis
 * angle vector and 3x1 translation vector.
 *
 * curlyhat(rho, aaxis) = [aaxis^   rho^]
 *                        [     0 aaxis^]
 *
 * See eq. 12 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
curlyhat(const Eigen::MatrixBase<Derived>& rho,
         const Eigen::MatrixBase<Derived>& aaxis) {
  assert(rho.cols() == 1 && rho.rows() == 3);
  assert(aaxis.cols() == 1 && aaxis.rows() == 3);
  Eigen::Matrix<typename Derived::Scalar, 6, 6> mat(6, 6);
  mat.setZero();
  mat.template topLeftCorner<3, 3>() = mat.template bottomRightCorner<3, 3>() =
      so3::hat(aaxis);
  mat.template topRightCorner<3, 3>() = so3::hat(rho);
  return mat;
}

/**
 * \brief Builds the 6x6 "curly hat" matrix (related to the skew symmetric
 * matrix)
 * \details
 * The curly hat operator builds the 6x6 skew symmetric matrix
 * from the 6x1 se3 algebra vector, xi:
 *
 * curlyhat(xi) = curlyhat([rho  ]) = [aaxis^   rho^]
 *                        ([aaxis])   [     0 aaxis^]
 *
 * See eq. 12 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
curlyhat(const Eigen::MatrixBase<Derived>& xi) {
  assert(xi.cols() == 1 && xi.rows() == 6);
  return curlyhat(xi.template head<3>(), xi.template tail<3>());
}

/**
 * \brief Turns a homogeneous point into a special 4x6 matrix (circle-dot
 * operator)
 * \details
 * See eq. 72 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 6>>
point2fs(const Eigen::MatrixBase<Derived>& p,
         typename Derived::Scalar scale = typename Derived::Scalar(1)) {
  assert(p.rows() == 3 && p.cols() == 1);
  Eigen::Matrix<typename Derived::Scalar, 4, 6> mat =
      Eigen::Matrix<typename Derived::Scalar, 4, 6>::Zero();
  mat.template topLeftCorner<3, 3>() =
      scale * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
  mat.template topRightCorner<3, 3>() = -so3::hat(p);
  return mat;
}

/**
 * \brief Turns a homogeneous point into a special 6x4 matrix (double-circle
 * operator)
 *
 * See eq. 72 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 4>>
point2sf(const Eigen::MatrixBase<Derived>& p,
         typename Derived::Scalar scale = typename Derived::Scalar(1)) {
  assert(p.rows() == 3 && p.cols() == 1);
  Eigen::Matrix<typename Derived::Scalar, 6, 4> mat =
      Eigen::Matrix<typename Derived::Scalar, 6, 4>::Zero();
  mat.template bottomLeftCorner<3, 3>() = -so3::hat(p);
  mat.template topRightCorner<3, 1>() = p;
  return mat;
}

/**
 * \brief Builds a transformation matrix using the analytical exponential map
 * \details
 * This function builds a transformation matrix, T_ab, using the analytical
 * exponential map, from the se3 algebra vector, xi_ba,
 *
 *   T_ab = exp(xi_ba^) = [ C_ab r_ba_ina],   xi_ba = [  rho_ba]
 *                        [  0^T        1]            [aaxis_ba]
 *
 * where C_ab is a 3x3 rotation matrix from 'b' to 'a', r_ba_ina is the 3x1
 * translation vector from 'a' to 'b' expressed in frame 'a', aaxis_ba is a 3x1
 * axis-angle vector, the magnitude of the angle of rotation can be recovered by
 * finding the norm of the vector, and the axis of rotation is the unit-length
 * vector that arises from normalization. Note that the angle around the axis,
 * aaxis_ba, is a right-hand-rule (counter-clockwise positive) angle from 'a' to
 * 'b'.
 *
 * The parameter, rho_ba, is a special translation-like parameter related to
 * 'twist' theory. It is most inuitively described as being like a constant
 * linear velocity (expressed in the smoothly-moving frame) for a fixed
 * duration; for example, consider the curve of a car driving 'x' meters while
 * turning at a rate of 'y' rad/s.
 *
 * For more information see Barfoot-TRO-2014 Appendix B1.
 *
 * Alternatively, we that note that
 *
 *   T_ba = exp(-xi_ba^) = exp(xi_ab^).
 *
 * Both the analytical (numTerms = 0) or the numerical (numTerms > 0) may be
 * evaluated.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value, void>
vec2tran_analytical(const Eigen::MatrixBase<Derived>& rho_ba,
                    const Eigen::MatrixBase<Derived>& aaxis_ba,
                    Eigen::Matrix<typename Derived::Scalar, 3, 3>& out_C_ab,
                    Eigen::Vector<typename Derived::Scalar, 3>& out_r_ba_ina) {
  assert(rho_ba.rows() == 3 && rho_ba.cols() == 1);
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, rotation is Identity
    out_C_ab = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
               so3::hat(aaxis_ba);
    out_r_ba_ina = out_C_ab * rho_ba;
  } else {
    // Normal analytical solution
    Eigen::Matrix<typename Derived::Scalar, 3, 3> J_ab;

    // Use rotation identity involving jacobian, as we need it to
    // convert rho_ba to the proper translation
    so3::vec2rot(aaxis_ba, out_C_ab, J_ab);

    // Convert rho_ba (twist-translation) to r_ba_ina
    Eigen::Matrix<typename Derived::Scalar, 3, 1> out_r = J_ab * rho_ba;

    const typename Derived::Scalar phi_ba = aaxis_ba.norm();

    if (fabs(M_PI - phi_ba) < 1e-6) {
      std::cout
          << "[WARNING]: vec2tran is not differentiable at pi. "
             "lgmath_autodiff does not support gradients of angles greater "
             "than pi. Gradients through this function call are zero."
          << std::endl;
      // If angle is near pi, then the jacobian is not differentiable
      out_r_ba_ina = Eigen::Matrix<typename Derived::Scalar, 3, 1>(
          out_r.template cast<double>());
    } else {
      out_r_ba_ina = out_r;
    }
  }
}

/**
 * \brief Builds a transformation matrix using the first N terms of the
 * infinite series
 * \details
 * Builds a transformation matrix numerically using the infinite series
 * evalation of the exponential map.
 *
 * For more information see eq. 96 in Barfoot-TRO-2014
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value, void>
vec2tran_numerical(const Eigen::MatrixBase<Derived>& rho_ba,
                   const Eigen::MatrixBase<Derived>& aaxis_ba,
                   Eigen::Matrix<typename Derived::Scalar, 3, 3>& out_C_ab,
                   Eigen::Vector<typename Derived::Scalar, 3>& out_r_ba_ina,
                   unsigned int numTerms = 0) {
  assert(rho_ba.rows() == 3 && rho_ba.cols() == 1);
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Init 4x4 transformation
  Eigen::Matrix<typename Derived::Scalar, 4, 4> T_ab =
      Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();

  // Incremental variables
  Eigen::Vector<typename Derived::Scalar, 6> xi_ba(6);
  xi_ba << rho_ba, aaxis_ba;
  Eigen::Matrix<typename Derived::Scalar, 4, 4> x_small = se3::hat(xi_ba);
  Eigen::Matrix<typename Derived::Scalar, 4, 4> x_small_n =
      Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();

  // Loop over sum up to the specified numTerms
  for (unsigned int n = 1; n <= numTerms; n++) {
    x_small_n = x_small_n * x_small / double(n);
    T_ab += x_small_n;
  }

  // Fill output
  out_C_ab = T_ab.template topLeftCorner<3, 3>();
  out_r_ba_ina = T_ab.template topRightCorner<3, 1>();
}

/**
 * \brief Builds the 3x3 rotation and 3x1 translation using the exponential
 * map, the default parameters (numTerms = 0) use the analytical solution.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value, void>
vec2tran(const Eigen::MatrixBase<Derived>& xi_ba,
         Eigen::Matrix<typename Derived::Scalar, 3, 3>& out_C_ab,
         Eigen::Vector<typename Derived::Scalar, 3>& out_r_ba_ina,
         unsigned int numTerms = 0) {
  assert(xi_ba.rows() == 6 && xi_ba.cols() == 1);
  if (numTerms == 0) {
    // Analytical solution
    vec2tran_analytical(xi_ba.template head<3>(), xi_ba.template tail<3>(),
                        out_C_ab, out_r_ba_ina);
  } else {
    // Numerical solution (good for testing the analytical solution)
    vec2tran_numerical(xi_ba.template head<3>(), xi_ba.template tail<3>(),
                       out_C_ab, out_r_ba_ina, numTerms);
  }
}

/**
 * \brief Builds a 4x4 transformation matrix using the exponential map, the
 * default parameters (numTerms = 0) use the analytical solution.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 4, 4>>
vec2tran(const Eigen::MatrixBase<Derived>& xi_ba, unsigned int numTerms = 0) {
  assert(xi_ba.rows() == 6 && xi_ba.cols() == 1);
  // Get rotation and translation
  Eigen::Matrix<typename Derived::Scalar, 3, 3> C_ab;
  Eigen::Vector<typename Derived::Scalar, 3> r_ba_ina;
  vec2tran(xi_ba, C_ab, r_ba_ina, numTerms);

  // Fill output
  Eigen::Matrix<typename Derived::Scalar, 4, 4> T_ab =
      Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();
  T_ab.template topLeftCorner<3, 3>() = C_ab;
  T_ab.template topRightCorner<3, 1>() = r_ba_ina;
  return T_ab;
}

/**
 * \brief Compute the matrix log of a transformation matrix (from the rotation
 * and trans)
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a the 3x3 rotation and 3x1 translation vector back to a 6x1 se3
 * algebra vector (composed of a 3x1 axis-angle vector and 3x1 twist-translation
 * vector). In some cases, when the rotation in the transformation matrix is
 * 'numerically off', this involves some 'projection' back to SE(3).
 *
 *   xi_ba = ln(T_ab)
 *
 * where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
 *
 *   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
 *
 * See Barfoot-TRO-2014 Appendix B2 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Vector<AUTODIFF_VAR_TYPE, 6>>
tran2vec(const Eigen::MatrixBase<Derived>& C_ab,
         const Eigen::Vector<typename Derived::Scalar, 3>& r_ba_ina) {
  assert(C_ab.rows() == 3 && C_ab.cols() == 3);
  // Init
  Eigen::Vector<typename Derived::Scalar, 6> xi_ba(6);

  // Get axis angle from rotation matrix
  Eigen::Vector<typename Derived::Scalar, 3> aaxis_ba = so3::rot2vec(C_ab);

  // Get twist-translation vector using Jacobian
  Eigen::Vector<typename Derived::Scalar, 3> rho_ba =
      so3::vec2jacinv(aaxis_ba) * r_ba_ina;

  // Return se3 algebra vector
  xi_ba << rho_ba, aaxis_ba;
  return xi_ba;
}

/**
 * \brief Compute the matrix log of a transformation matrix
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a 4x4 transformation matrix back to a 6x1 se3 algebra vector
 * (composed of a 3x1 axis-angle vector and 3x1 twist-translation vector). In
 * some cases, when the rotation in the transformation matrix is 'numerically
 * off', this involves some 'projection' back to SE(3).
 *
 *   xi_ba = ln(T_ab)
 *
 * where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
 *
 *   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
 *
 * See Barfoot-TRO-2014 Appendix B2 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Vector<AUTODIFF_VAR_TYPE, 6>>
tran2vec(const Eigen::MatrixBase<Derived>& T_ab) {
  assert(T_ab.rows() == 4 && T_ab.cols() == 4);
  return tran2vec(T_ab.template topLeftCorner<3, 3>(),
                  T_ab.template topRightCorner<3, 1>());
}

/**
 * \brief Builds the 6x6 adjoint transformation matrix from the 3x3 rotation
 * matrix and 3x1 translation vector.
 * \details
 * Builds the 6x6 adjoint transformation matrix from the 3x3 rotation matrix and
 * 3x1 translation vector.
 *
 *  Adjoint(T_ab) = Adjoint([C_ab r_ba_ina]) = [C_ab r_ba_ina^*C_ab] =
 * exp(curlyhat(xi_ba))
 *                         ([ 0^T        1])   [   0           C_ab]
 *
 * See eq. 101 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
tranAd(const Eigen::MatrixBase<Derived>& C_ab,
       const Eigen::Vector<typename Derived::Scalar, 3>& r_ba_ina) {
  assert(C_ab.rows() == 3 && C_ab.cols() == 3);
  Eigen::Matrix<typename Derived::Scalar, 6, 6> adjoint_T_ab(6, 6);
  adjoint_T_ab.setZero();

  adjoint_T_ab.template topLeftCorner<3, 3>() =
      adjoint_T_ab.template bottomRightCorner<3, 3>() = C_ab;
  adjoint_T_ab.template topRightCorner<3, 3>() = so3::hat(r_ba_ina) * C_ab;
  return adjoint_T_ab;
}

/**
 * \brief Builds the 6x6 adjoint transformation matrix from a 4x4 one
 * \details
 * Builds the 6x6 adjoint transformation matrix from a 4x4 transformation matrix
 *
 *  Adjoint(T_ab) = Adjoint([C_ab r_ba_ina]) = [C_ab r_ba_ina^*C_ab] =
 * exp(curlyhat(xi_ba))
 *                         ([ 0^T        1])   [   0           C_ab]
 *
 * See eq. 101 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
tranAd(const Eigen::MatrixBase<Derived>& T_ab) {
  assert(T_ab.rows() == 4 && T_ab.cols() == 4);
  return tranAd(T_ab.template topLeftCorner<3, 3>(),
                T_ab.template topRightCorner<3, 1>());
}

/**
 * \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
 * \details
 * See eq. 102 in Barfoot-TRO-2014 for more information
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>>
vec2Q(const Eigen::MatrixBase<Derived>& rho_ba,
      const Eigen::MatrixBase<Derived>& aaxis_ba) {
  assert(rho_ba.rows() == 3 && rho_ba.cols() == 1);
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Construct scalar terms
  const typename Derived::Scalar ang = aaxis_ba.norm();
  const typename Derived::Scalar ang2 = ang * ang;
  const typename Derived::Scalar ang3 = ang2 * ang;
  const typename Derived::Scalar ang4 = ang3 * ang;
  const typename Derived::Scalar ang5 = ang4 * ang;
  const typename Derived::Scalar cang = cos(ang);
  const typename Derived::Scalar sang = sin(ang);
  const typename Derived::Scalar m2 = (ang - sang) / ang3;
  const typename Derived::Scalar m3 = (1.0 - 0.5 * ang2 - cang) / ang4;
  const typename Derived::Scalar m4 =
      0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5);

  // Construct matrix terms
  Eigen::Matrix<typename Derived::Scalar, 3, 3> rx = so3::hat(rho_ba);
  Eigen::Matrix<typename Derived::Scalar, 3, 3> px = so3::hat(aaxis_ba);
  Eigen::Matrix<typename Derived::Scalar, 3, 3> pxrx = px * rx;
  Eigen::Matrix<typename Derived::Scalar, 3, 3> rxpx = rx * px;
  Eigen::Matrix<typename Derived::Scalar, 3, 3> pxrxpx = pxrx * px;

  // Construct Q matrix
  return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) -
         m3 * (px * pxrx + rxpx * px - 3 * pxrxpx) -
         m4 * (pxrxpx * px + px * pxrxpx);
}

/**
 * \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
 * \details
 * See eq. 102 in Barfoot-TRO-2014 for more information
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>>
vec2Q(const Eigen::MatrixBase<Derived>& xi_ba) {
  assert(xi_ba.rows() == 6 && xi_ba.cols() == 1);
  return vec2Q(xi_ba.template head<3>(), xi_ba.template tail<3>());
}

/**
 * \brief Builds the 6x6 Jacobian matrix of SE(3) using the analytical
 * expression
 * \details
 * Build the 6x6 left Jacobian of SE(3).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * transformation,
 *
 *   J_ab = J(xi_ba)
 *
 * Where applicable, we also note that
 *
 *   J(xi_ba) = Adjoint(exp(xi_ba^)) * J(-xi_ba),
 *
 * and
 *
 *   Adjoint(exp(xi_ba^)) = identity + curlyhat(xi_ba) * J(xi_ba).
 *
 * For more information see eq. 100 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
vec2jac(const Eigen::MatrixBase<Derived>& rho_ba,
        const Eigen::MatrixBase<Derived>& aaxis_ba) {  // Init
  assert(rho_ba.rows() == 3 && rho_ba.cols() == 1);
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  Eigen::Matrix<typename Derived::Scalar, 6, 6> J_ab(6, 6);
  J_ab.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J_ab.template topLeftCorner<3, 3>() =
        J_ab.template bottomRightCorner<3, 3>() =
            Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
            0.5 * so3::hat(aaxis_ba);
    J_ab.template topRightCorner<3, 3>() = 0.5 * so3::hat(rho_ba);
  } else {
    // General analytical scenario
    J_ab.template topLeftCorner<3, 3>() =
        J_ab.template bottomRightCorner<3, 3>() = so3::vec2jac(aaxis_ba);
    J_ab.template topRightCorner<3, 3>() = se3::vec2Q(rho_ba, aaxis_ba);
  }
  return J_ab;
}

/**
 * \brief Builds the 6x6 Jacobian matrix of SE(3) from the se(3) algebra; note
 * that the default parameter (numTerms = 0) will call the analytical solution,
 * but the numerical solution can also be evaluating to some number of terms.
 * \details
 * For more information see eq. 100 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
vec2jac(const Eigen::MatrixBase<Derived>& xi_ba, unsigned int numTerms = 0) {
  assert(xi_ba.rows() == 6 && xi_ba.cols() == 1);
  if (numTerms == 0) {
    // Analytical solution
    return vec2jac(xi_ba.template head<3>(), xi_ba.template tail<3>());
  } else {
    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<typename Derived::Scalar, 6, 6> J_ab(6, 6);
    J_ab.setIdentity();

    // Incremental variables
    Eigen::Matrix<typename Derived::Scalar, 6, 6> x_small =
        se3::curlyhat(xi_ba);
    Eigen::Matrix<typename Derived::Scalar, 6, 6> x_small_n(6, 6);
    x_small_n.setIdentity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
    return J_ab;
  }
}

/**
 * \brief Builds the 6x6 inverse Jacobian matrix of SE(3) using the analytical
 * expression
 * \details
 * Build the 6x6 inverse left Jacobian of SE(3).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * transformation,
 *
 *   J_ab_inverse = J(xi_ba)^{-1},
 *
 * Please note that J_ab_inverse is not equivalent to J_ba:
 *
 *   J(xi_ba)^{-1} != J(-xi_ba)
 *
 * For more information see eq. 103 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
vec2jacinv(const Eigen::MatrixBase<Derived>& rho_ba,
           const Eigen::MatrixBase<Derived>& aaxis_ba) {
  assert(rho_ba.rows() == 3 && rho_ba.cols() == 1);
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Init
  Eigen::Matrix<typename Derived::Scalar, 6, 6> J66_ab_inv(6, 6);
  J66_ab_inv.setZero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J66_ab_inv.template topLeftCorner<3, 3>() =
        J66_ab_inv.template bottomRightCorner<3, 3>() =
            Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
            0.5 * so3::hat(aaxis_ba);
    J66_ab_inv.template topRightCorner<3, 3>() = -0.5 * so3::hat(rho_ba);
  } else {
    // General analytical scenario
    Eigen::Matrix<typename Derived::Scalar, 3, 3> J33_ab_inv =
        so3::vec2jacinv(aaxis_ba);
    J66_ab_inv.template topLeftCorner<3, 3>() =
        J66_ab_inv.template bottomRightCorner<3, 3>() = J33_ab_inv;
    J66_ab_inv.template topRightCorner<3, 3>() =
        -J33_ab_inv * se3::vec2Q(rho_ba, aaxis_ba) * J33_ab_inv;
  }
  return J66_ab_inv;
}

/**
 * \brief Builds the 6x6 inverse Jacobian matrix of SE(3) from the se(3)
 * algebra; note that the default parameter (numTerms = 0) will call the
 * analytical solution, but the numerical solution can also be evaluating to
 * some number of terms.
 * \details
 * For more information see eq. 103 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, AUTODIFF_VAR_TYPE>::value,
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 6, 6>>
vec2jacinv(const Eigen::MatrixBase<Derived>& xi_ba, unsigned int numTerms = 0) {
  assert(xi_ba.rows() == 6 && xi_ba.cols() == 1);
  if (numTerms == 0) {
    // Analytical solution
    return vec2jacinv(xi_ba.template head<3>(), xi_ba.template tail<3>());
  } else {
    // Logic error
    if (numTerms > 20) {
      throw std::invalid_argument(
          "Numerical vec2jacinv does not support numTerms > 20");
    }

    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<typename Derived::Scalar, 6, 6> J_ab(6, 6);
    J_ab.setIdentity();

    // Incremental variables
    Eigen::Matrix<typename Derived::Scalar, 6, 6> x_small =
        se3::curlyhat(xi_ba);
    Eigen::Matrix<typename Derived::Scalar, 6, 6> x_small_n(6, 6);
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

}  // namespace se3
}  // namespace lgmath
#endif