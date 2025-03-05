/**
 * \file Operations.hpp
 * \brief Header file for the SE3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#pragma once

#include <Eigen/Core>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

/// Lie Group Math - Special Euclidean Group
namespace lgmath {
namespace se3 {
namespace diff {
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
autodiff::Matrix4real hat(const autodiff::Vector3real& rho, const autodiff::Vector3real& aaxis);

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
autodiff::Matrix4real hat(const autodiff::VectorXreal& xi);

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
autodiff::MatrixXreal curlyhat(const autodiff::Vector3real& rho,
                                     const autodiff::Vector3real& aaxis);

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
autodiff::MatrixXreal curlyhat(const autodiff::VectorXreal& xi);

// /**
//  * \brief Turns a homogeneous point into a special 4x6 matrix (circle-dot
//  * operator)
//  * \details
//  * See eq. 72 in Barfoot-TRO-2014 for more information.
//  */
// Eigen::Matrix<double, 4, 6> point2fs(const autodiff::Vector3real& p,
//                                      double scale = 1);

// /**
//  * \brief Turns a homogeneous point into a special 6x4 matrix (double-circle
//  * operator)
//  *
//  * See eq. 72 in Barfoot-TRO-2014 for more information.
//  */
// Eigen::Matrix<double, 6, 4> point2sf(const autodiff::Vector3real& p,
//                                      double scale = 1);

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
void vec2tran_analytical(const autodiff::Vector3real& rho_ba,
                         const autodiff::Vector3real& aaxis_ba,
                         autodiff::Matrix3real* out_C_ab,
                         autodiff::Vector3real* out_r_ba_ina);

/**
 * \brief Builds a transformation matrix using the first N terms of the
 * infinite series
 * \details
 * Builds a transformation matrix numerically using the infinite series
 * evalation of the exponential map.
 *
 * For more information see eq. 96 in Barfoot-TRO-2014
 */
void vec2tran_numerical(const autodiff::Vector3real& rho_ba,
                        const autodiff::Vector3real& aaxis_ba,
                        autodiff::Matrix3real* out_C_ab,
                        autodiff::Vector3real* out_r_ba_ina,
                        unsigned int numTerms = 0);

/**
 * \brief Builds the 3x3 rotation and 3x1 translation using the exponential
 * map, the default parameters (numTerms = 0) use the analytical solution.
 */
void vec2tran(const autodiff::VectorXreal& xi_ba,
              autodiff::Matrix3real* out_C_ab, autodiff::Vector3real* out_r_ba_ina,
              unsigned int numTerms = 0);

/**
 * \brief Builds a 4x4 transformation matrix using the exponential map, the
 * default parameters (numTerms = 0) use the analytical solution.
 */
autodiff::Matrix4real vec2tran(const autodiff::VectorXreal& xi_ba,
                         unsigned int numTerms = 0);

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
autodiff::VectorXreal tran2vec(const autodiff::Matrix3real& C_ab,
                                     const autodiff::Vector3real& r_ba_ina);

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
autodiff::VectorXreal tran2vec(const autodiff::Matrix4real& T_ab);

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
autodiff::MatrixXreal tranAd(const autodiff::Matrix3real& C_ab,
                                   const autodiff::Vector3real& r_ba_ina);

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
autodiff::MatrixXreal tranAd(const autodiff::Matrix4real& T_ab);

/**
 * \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
 * \details
 * See eq. 102 in Barfoot-TRO-2014 for more information
 */
autodiff::Matrix3real vec2Q(const autodiff::Vector3real& rho_ba,
                      const autodiff::Vector3real& aaxis_ba);

/**
 * \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
 * \details
 * See eq. 102 in Barfoot-TRO-2014 for more information
 */
autodiff::Matrix3real vec2Q(const autodiff::VectorXreal& xi_ba);

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
autodiff::MatrixXreal vec2jac(const autodiff::Vector3real& rho_ba,
                                    const autodiff::Vector3real& aaxis_ba);

/**
 * \brief Builds the 6x6 Jacobian matrix of SE(3) from the se(3) algebra; note
 * that the default parameter (numTerms = 0) will call the analytical solution,
 * but the numerical solution can also be evaluating to some number of terms.
 * \details
 * For more information see eq. 100 in Barfoot-TRO-2014.
 */
autodiff::MatrixXreal vec2jac(const autodiff::VectorXreal& xi_ba,
                                    unsigned int numTerms = 0);

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
autodiff::MatrixXreal vec2jacinv(const autodiff::Vector3real& rho_ba,
                                       const autodiff::Vector3real& aaxis_ba);

/**
 * \brief Builds the 6x6 inverse Jacobian matrix of SE(3) from the se(3)
 * algebra; note that the default parameter (numTerms = 0) will call the
 * analytical solution, but the numerical solution can also be evaluating to
 * some number of terms.
 * \details
 * For more information see eq. 103 in Barfoot-TRO-2014.
 */
autodiff::MatrixXreal vec2jacinv(const autodiff::VectorXreal& xi_ba,
                                       unsigned int numTerms = 0);

}  // namespace diff
}  // namespace se3
}  // namespace lgmath
