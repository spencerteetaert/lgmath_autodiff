#if USE_AUTODIFF
/**
 * \file Operations.cpp
 * \brief Implementation file for the SO3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * orthogonal (SO) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#include <lgmath/CommonMath.hpp>
#include <lgmath/so3/OperationsAutodiff.hpp>

#include <stdio.h>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace lgmath {
namespace so3 {
namespace diff {
autodiff::Matrix3real hat(const autodiff::Vector3real& vector) {
  autodiff::Matrix3real mat;
  mat << 0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1],
      vector[0], 0.0;
  return mat;
}

autodiff::Matrix3real vec2rot(const autodiff::Vector3real& aaxis_ba,
                              unsigned int numTerms) {
  // Get angle
  const autodiff::real phi_ba = aaxis_ba.norm();

  // If angle is very small, return Identity
  if (phi_ba < 1e-12) {
    return autodiff::Matrix3real::Identity();
  }

  if (numTerms == 0) {
    // Analytical solution
    autodiff::Vector3real axis = aaxis_ba / phi_ba;
    const autodiff::real sinphi_ba = sin(phi_ba);
    const autodiff::real cosphi_ba = cos(phi_ba);
    return cosphi_ba * autodiff::Matrix3real::Identity() +
           (1.0 - cosphi_ba) * axis * axis.transpose() +
           sinphi_ba * so3::diff::hat(axis);

  } else {
    // Numerical solution (good for testing the analytical solution)
    autodiff::Matrix3real C_ab = autodiff::Matrix3real::Identity();

    // Incremental variables
    autodiff::Matrix3real x_small = so3::diff::hat(aaxis_ba);
    autodiff::Matrix3real x_small_n = autodiff::Matrix3real::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n);
      C_ab += x_small_n;
    }
    return C_ab;
  }
}

void vec2rot(const autodiff::Vector3real& aaxis_ba,
             autodiff::Matrix3real* out_C_ab, autodiff::Matrix3real* out_J_ab) {
  // Check pointers
  if (out_C_ab == NULL) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2rot");
  }
  if (out_J_ab == NULL) {
    throw std::invalid_argument("Null pointer out_J_ab in vec2rot");
  }

  // Set Jacobian term
  *out_J_ab = so3::diff::vec2jac(aaxis_ba);

  // Set rotation matrix
  *out_C_ab = autodiff::Matrix3real::Identity() +
              so3::diff::hat(aaxis_ba) * (*out_J_ab);
}

autodiff::real clamp(const autodiff::real& val, const autodiff::real& min,
                     const autodiff::real& max) {
  return val < min ? min : (val > max ? max : val);
}

autodiff::Vector3real rot2vec(const autodiff::Matrix3real& C_ab,
                              const double eps) {
  // Get angle
  const autodiff::real phi_ba = acos(
      clamp(0.5 * (C_ab.trace() - 1.0), -0.999999999999999, 0.999999999999999));
  const autodiff::real sinphi_ba = sin(phi_ba);

  if (fabs(sinphi_ba.val()) > eps) {
    // General case, angle is NOT near 0, pi, or 2*pi
    autodiff::Vector3real axis;
    axis << C_ab(2, 1) - C_ab(1, 2), C_ab(0, 2) - C_ab(2, 0),
        C_ab(1, 0) - C_ab(0, 1);
    return (0.5 * phi_ba / sinphi_ba) * axis;

  } else if (fabs(phi_ba.val()) > eps) {
    // Angle is near pi or 2*pi
    // ** Note with this method we do not know the sign of 'phi', however since
    // we know phi is
    //    close to pi or 2*pi, the sign is unimportant..

    // Find the eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<autodiff::Matrix3real> eigenSolver(C_ab);

    /*
    Note: This requires a small change in the autodiff library. This comes from the fact that the Eigen eigensolver 
    uses std functions that are not supported by autodiff. The type casting from real type to double is only supported 
    explicitly. This edit allows for casting to double to be done implicitly, enabling use of std functions. 
    
    https://github.com/autodiff/autodiff/blob/b0a4feff5b2a61262e94305452ac53369fe35e75/autodiff/forward/real/real.hpp#L189 

    The change is to swap out the conditional code block in real.hpp with: 
    ----------------------------------------------------------------------------------------------------------------------
    #if defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION_REAL) || defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION)
      AUTODIFF_DEVICE_FUNC constexpr operator T() const { return static_cast<T>(m_data[0]); }
    #endif
    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr explicit operator U() const { return static_cast<U>(m_data[0]); }
    ----------------------------------------------------------------------------------------------------------------------

    This is untested and may break other functionality. 
    */

    // Try each eigenvalue
    for (int i = 0; i < 3; i++) {
      // Check if eigen value is near +1.0
      if (fabs(eigenSolver.eigenvalues()[i].val() - 1.0) < 1e-6) {
        // Get corresponding angle-axis
        autodiff::Vector3real aaxis_ba =
            phi_ba * eigenSolver.eigenvectors().col(i);
        return aaxis_ba;
      }
    }

    // Runtime error
    throw std::runtime_error(
        "so3 logarithmic map failed to find an axis-angle, "
        "angle was near pi, or 2*pi, but no eigenvalues were near 1");
  } else {
    // Angle is near zero
    return autodiff::Vector3real::Zero();
  }
}

void func(const autodiff::Matrix3real& R) {
  Eigen::SelfAdjointEigenSolver<autodiff::Matrix3real> eigenSolver(R);
}

autodiff::Matrix3real vec2jac(const autodiff::Vector3real& aaxis_ba,
                              unsigned int numTerms) {
  // Get angle
  const autodiff::real phi_ba = aaxis_ba.norm();
  if (phi_ba < 1e-12) {
    // If angle is very small, return Identity
    return autodiff::Matrix3real::Identity();
  }

  if (numTerms == 0) {
    // Analytical solution
    autodiff::Vector3real axis = aaxis_ba / phi_ba;
    const autodiff::real sinTerm = sin(phi_ba) / phi_ba;
    const autodiff::real cosTerm = (1.0 - cos(phi_ba)) / phi_ba;
    return sinTerm * autodiff::Matrix3real::Identity() +
           (1.0 - sinTerm) * axis * axis.transpose() +
           cosTerm * so3::diff::hat(axis);
  } else {
    // Numerical solution (good for testing the analytical solution)
    autodiff::Matrix3real J_ab = autodiff::Matrix3real::Identity();

    // Incremental variables
    autodiff::Matrix3real x_small = so3::diff::hat(aaxis_ba);
    autodiff::Matrix3real x_small_n = autodiff::Matrix3real::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
    return J_ab;
  }
}

autodiff::Matrix3real vec2jacinv(const autodiff::Vector3real& aaxis_ba,
                                 unsigned int numTerms) {
  // Get angle
  const autodiff::real phi_ba = aaxis_ba.norm();
  if (phi_ba < 1e-12) {
    // If angle is very small, return Identity
    return autodiff::Matrix3real::Identity();
  }

  if (numTerms == 0) {
    // Analytical solution
    autodiff::Vector3real axis = aaxis_ba / phi_ba;
    const autodiff::real halfphi = 0.5 * phi_ba;
    const autodiff::real cotanTerm = halfphi / tan(halfphi);
    return cotanTerm * autodiff::Matrix3real::Identity() +
           (1.0 - cotanTerm) * axis * axis.transpose() -
           halfphi * so3::diff::hat(axis);
  } else {
    // Logic error
    if (numTerms > 20) {
      throw std::invalid_argument(
          "Numerical vec2jacinv does not support numTerms > 20");
    }

    // Numerical solution (good for testing the analytical solution)
    autodiff::Matrix3real J_ab_inverse = autodiff::Matrix3real::Identity();

    // Incremental variables
    autodiff::Matrix3real x_small = so3::diff::hat(aaxis_ba);
    autodiff::Matrix3real x_small_n = autodiff::Matrix3real::Identity();

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
    return J_ab_inverse;
  }
}

}  // namespace diff
}  // namespace so3
}  // namespace lgmath
#endif