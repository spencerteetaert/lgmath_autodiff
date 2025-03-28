
/**
 * \file Operations.hpp
 * \brief Header file for the SO3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * orthogonal (SO) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#pragma once

#ifdef USE_AUTODIFF
#ifdef USE_AUTODIFF_BACKWARD

#include <Eigen/Core>
#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using autodiff::var; 

/// Lie Group Math - Special Orthogonal Group
namespace lgmath {
namespace so3 {

inline var clamp(const var& val,
                               const var& min,
                               const var& max) {
  return (val < min) ? min : (max < val) ? max : val;
}

inline var smooth_acos(const var& x) {
  if (1.0 - fabs(x) > 1e-6)
    return acos(x);
  else
    return M_PI_2 - atan(x / sqrt(1.0 - x * x + 1e-13));
}

/**
 * \brief Builds the 3x3 skew symmetric matrix
 * \details
 * The hat (^) operator, builds the 3x3 skew symmetric matrix from the 3x1
 * vector:
 *
 * v^ = [0.0  -v3   v2]
 *      [ v3  0.0  -v1]
 *      [-v2   v1  0.0]
 *
 * See eq. 5 in Barfoot-TRO-2014 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value,
    Eigen::Matrix<var, 3, 3>>
hat(const Eigen::MatrixBase<Derived>& vector) {
  assert(vector.rows() == 3 && vector.cols() == 1);
  Eigen::Matrix<typename Derived::Scalar, 3, 3> mat;
  mat << 0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1],
      vector[0], 0.0;
  return mat;
}

/**
 * \brief Builds a rotation matrix using the exponential map
 * \details
 * This function builds a rotation matrix, C_ab, using the exponential map (from
 * an axis- angle parameterization).
 *
 *   C_ab = exp(aaxis_ba^),
 *
 * where aaxis_ba is a 3x1 axis-angle vector, the magnitude of the angle of
 * rotation can be recovered by finding the norm of the vector, and the axis of
 * rotation is the unit- length vector that arises from normalization. Note that
 * the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise
 * positive) angle from 'a' to 'b'.
 *
 * Alternatively, we that note that
 *
 *   C_ba = exp(-aaxis_ba^) = exp(aaxis_ab^).
 *
 * Typical robotics convention has some oddity when it comes using this
 * exponential map in practice. For example, if we wish to integrate the
 * kinematics:
 *
 *   d/dt C = omega^ * C,
 *
 * where omega is the 3x1 angular velocity, we employ the convention:
 *
 *   C_20 = exp(deltaTime*-omega^) * C_10,
 *
 * Noting that omega is negative (left-hand-rule).
 * For more information see eq. 97 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value,
    Eigen::Matrix<var, 3, 3>>
vec2rot(const Eigen::MatrixBase<Derived>& aaxis_ba, unsigned int numTerms = 0) {
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Get angle
  const typename Derived::Scalar phi_ba = aaxis_ba.norm();

  // If angle is very small, return Identity
  if (phi_ba < 1e-12) {
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
           hat(aaxis_ba);
  }

  if (numTerms == 0) {
    // Analytical solution
    Eigen::Vector<typename Derived::Scalar, 3> axis = aaxis_ba / phi_ba;
    const typename Derived::Scalar sinphi_ba = sin(phi_ba);
    const typename Derived::Scalar cosphi_ba = cos(phi_ba);
    return cosphi_ba *
               Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
           (1.0 - cosphi_ba) * axis * axis.transpose() +
           sinphi_ba * so3::hat(axis);

  } else {
    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<typename Derived::Scalar, 3, 3> C_ab =
        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Incremental variables
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small = so3::hat(aaxis_ba);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small_n =
        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n);
      C_ab += x_small_n;
    }
    return C_ab;
  }
}

/**
 * \brief Compute the matrix log of a rotation matrix
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a 3x3 rotation matrix back to a 3x1 axis angle parameterization.
 * In some cases, when the rotation matrix is 'numerically off', this involves
 * some 'projection' back to SO(3).
 *
 *   aaxis_ba = ln(C_ab)
 *
 * where aaxis_ba is a 3x1 axis angle, where the axis is normalized and the
 * magnitude of the rotation can be recovered by finding the norm of the axis
 * angle. Note that the angle around the axis, aaxis_ba, is a right-hand-rule
 * (counter-clockwise positive) angle from 'a' to 'b'.
 *
 * Alternatively, we that note that
 *
 *   aaxis_ab = -aaxis_ba = ln(C_ba) = ln(C_ab^T)
 *
 * See Barfoot-TRO-2014 Appendix B2 for more information.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value,
    Eigen::Vector<var, 3>>
rot2vec(const Eigen::MatrixBase<Derived>& C_ab, const double eps = 1e-6) {
  assert(C_ab.rows() == 3 && C_ab.cols() == 3);
  // Get angle
  // Find the eigenvalues and eigenvectors
  const typename Derived::Scalar phi_ba =
      smooth_acos(clamp(0.5 * (C_ab.trace() - 1.0), -1.0, 1.0));
  const typename Derived::Scalar sinphi_ba = sin(phi_ba);

  if (fabs(sinphi_ba) > eps) {
    // General case, angle is NOT near 0, pi, or 2*pi
    Eigen::Vector<typename Derived::Scalar, 3> axis;
    axis << C_ab(2, 1) - C_ab(1, 2), C_ab(0, 2) - C_ab(2, 0),
        C_ab(1, 0) - C_ab(0, 1);
    return (0.5 * phi_ba / sinphi_ba) * axis;
  } else if (fabs(phi_ba) > eps) {
    // pi case
    std::cout << "[WARNING]: Rot2Vec is not differentiable at pi. "
                 "lgmath_autodiff does not support gradients of angles greater "
                 "than pi. Gradients through this function call are zero."
              << std::endl;
    Eigen::Matrix3d R = C_ab.template cast<double>();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(R);

    const auto eivalues = eigenSolver.eigenvalues();
    const auto eivalues_imag = eivalues.imag();
    const auto eivalues_real = eivalues.real();
    const auto eivec = eigenSolver.eigenvectors();
    const auto eivec_real = eivec.real();

    // Try each eigenvalue
    for (int i = 0; i < 3; i++) {
      // Check if eigen value is near +1.0
      if ((fabs(eivalues_imag(i)) < 1e-6) &&
          (fabs(eivalues_real(i) - 1.0) < 1e-6)) {
        // Get corresponding angle-axis
        Eigen::Vector3d aaxis_ba = double(phi_ba) * eivec_real.col(i);
        return static_cast<Eigen::Matrix<typename Derived::Scalar, 3, 1>>(
            aaxis_ba);
      }
    }

    // Runtime error
    throw std::runtime_error(
        "so3 logarithmic map failed to find an axis-angle, "
        "angle was near pi, or 2*pi, but no eigenvalues were near 1");
  } else {
    // Angle is near zero or 2*pi
    Eigen::Vector<typename Derived::Scalar, 3> aaxis_ba;
    aaxis_ba << (-1.0 * C_ab(1, 2) + C_ab(2, 1)) / 2.0,
        (C_ab(0, 2) - C_ab(2, 0)) / 2.0, (-1.0 * C_ab(0, 1) + C_ab(1, 0)) / 2.0;
    return aaxis_ba;
  }
}

/**
 * \brief Builds the 3x3 Jacobian matrix of SO(3)
 * \details
 * Build the 3x3 left Jacobian of SO(3).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * rotation,
 *
 *   J_ab = J(aaxis_ba),
 *
 * although we note to the SO(3) novice that this Jacobian is not a rotation
 * matrix, and should be used with care.
 *
 * For more information see eq. 98 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value,
    Eigen::Matrix<var, 3, 3>>
vec2jac(const Eigen::MatrixBase<Derived>& aaxis_ba, unsigned int numTerms = 0) {
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Get angle
  const typename Derived::Scalar phi_ba = aaxis_ba.norm();
  if (phi_ba < 1e-12) {
    // If angle is very small, return Identity
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
           0.5 * hat(aaxis_ba);
  }

  // Numerical solution (good for testing the analytical solution)
  Eigen::Matrix<typename Derived::Scalar, 3, 3> J_ab;

  if (numTerms == 0) {
    // Analytical solution
    Eigen::Vector<typename Derived::Scalar, 3> axis = aaxis_ba / phi_ba;
    const typename Derived::Scalar sinTerm = sin(phi_ba) / phi_ba;
    const typename Derived::Scalar cosTerm = (1.0 - cos(phi_ba)) / phi_ba;
    J_ab = sinTerm * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
           (1.0 - sinTerm) * axis * axis.transpose() + cosTerm * so3::hat(axis);
  } else {
    // Numerical solution (good for testing the analytical solution)
    J_ab = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Incremental variables
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small = so3::hat(aaxis_ba);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small_n =
        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
  }

  if (fabs(M_PI - phi_ba) < 1e-6) {
    std::cout << "[WARNING]: vec2jac is not differentiable at pi. "
                 "lgmath_autodiff does not support gradients of angles greater "
                 "than pi. Gradients through this function call are zero."
              << std::endl;
    // If angle is near pi, then the jacobian is not differentiable
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>(
        J_ab.template cast<double>());
  }
  return J_ab;
}

/**
 * \brief Builds and returns both the rotation matrix and SO(3) Jacobian
 * \details
 * Similar to the function 'vec2rot', this function builds a rotation matrix,
 * C_ab, using an equivalent expression to the exponential map, but allows us
 * to simultaneously extract the Jacobian of SO(3), which is also required in
 * some cases.
 *
 *   J_ab = jac(aaxis_ba)
 *   C_ab = exp(aaxis_ba^) = identity + aaxis_ba^ * J_ab
 *
 * For more information see eq. 97 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value, void>
vec2rot(const Eigen::MatrixBase<Derived>& aaxis_ba,
        Eigen::Matrix<typename Derived::Scalar, 3, 3>& out_C_ab,
        Eigen::Matrix<typename Derived::Scalar, 3, 3>& out_J_ab) {
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Set Jacobian term
  out_J_ab = so3::vec2jac(aaxis_ba);

  // Set rotation matrix
  Eigen::Matrix<typename Derived::Scalar, 3, 3> C_ab =
      Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
      so3::hat(aaxis_ba) * out_J_ab;

  const typename Derived::Scalar phi_ba = aaxis_ba.norm();

  if (fabs(M_PI - phi_ba) < 1e-6) {
    std::cout << "[WARNING]: vec2rot is not differentiable at pi. "
                 "lgmath_autodiff does not support gradients of angles greater "
                 "than pi. Gradients through this function call are zero."
              << std::endl;
    // If angle is near pi, then the jacobian is not differentiable
    out_C_ab = Eigen::Matrix<typename Derived::Scalar, 3, 3>(
        C_ab.template cast<double>());
  } else {
    out_C_ab = C_ab;
  }
}

/**
 * \brief Builds the 3x3 inverse Jacobian matrix of SO(3)
 * \details
 * Build the 3x3 inverse left Jacobian of SO(3).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * rotation,
 *
 *   J_ab_inverse = J(aaxis_ba)^{-1},
 *
 * although we note to the SO(3) novice that this Jacobian is not a rotation
 * matrix, and should be used with care. Also, please note that J_ab_inverse
 * is not equivalent to J_ba:
 *
 *   J(aaxis_ba)^{-1} != J(-aaxis_ba)
 *
 * For more information see eq. 99 in Barfoot-TRO-2014.
 */
template <typename Derived>
std::enable_if_t<
    std::is_same<typename Derived::Scalar, var>::value,
    Eigen::Matrix<var, 3, 3>>
vec2jacinv(const Eigen::MatrixBase<Derived>& aaxis_ba,
           unsigned int numTerms = 0) {
  assert(aaxis_ba.rows() == 3 && aaxis_ba.cols() == 1);
  // Get angle
  const typename Derived::Scalar phi_ba = aaxis_ba.norm();
  if (phi_ba < 1e-12) {
    // If angle is very small, return Identity
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
           0.5 * hat(aaxis_ba);
  }

  Eigen::Matrix<typename Derived::Scalar, 3, 3> J_ab_inverse;

  if (numTerms == 0) {
    // Analytical solution
    Eigen::Vector<typename Derived::Scalar, 3> axis = aaxis_ba / phi_ba;
    const typename Derived::Scalar halfphi = 0.5 * phi_ba;
    const typename Derived::Scalar cotanTerm = halfphi / tan(halfphi);
    J_ab_inverse =
        cotanTerm * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
        (1.0 - cotanTerm) * axis * axis.transpose() - halfphi * so3::hat(axis);
  } else {
    // Logic error
    if (numTerms > 20) {
      throw std::invalid_argument(
          "Numerical vec2jacinv does not support numTerms > 20");
    }

    // Numerical solution (good for testing the analytical solution)
    J_ab_inverse = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Incremental variables
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small = so3::hat(aaxis_ba);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> x_small_n =
        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();

    // Boost has a bernoulli package... but we shouldn't need more than 20
    Eigen::Matrix<double, 21, 1> bernoulli;
    bernoulli << 1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
        -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
        -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0;

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n);
      J_ab_inverse += bernoulli(n) * x_small_n;
    }
  }

  if (fabs(M_PI - phi_ba) < 1e-6) {
    std::cout << "[WARNING]: vec2jacinv is not differentiable at pi. "
                 "lgmath_autodiff does not support gradients of angles greater "
                 "than pi. Gradients through this function call are zero."
              << std::endl;
    // If angle is near pi, then the inverse jacobian is not differentiable
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>(
        J_ab_inverse.template cast<double>());
  }
  return J_ab_inverse;
}
}  // namespace so3
}  // namespace lgmath
#endif
#endif