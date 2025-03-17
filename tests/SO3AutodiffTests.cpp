//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SO3AutodiffTests.cpp
/// \brief Unit tests for the autodiff implementation of the SO3 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both
/// special cases,
///          and randomly generated cases.
///
/// \author Spencer Teetaert
//////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include <math.h>
#include <iomanip>
#include <ios>
#include <iostream>

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/OperationsAutodiff.hpp>

#ifdef AUTODIFF_USE_FORWARD
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#else
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SO(3) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SO(3) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, Test3x3HatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 3, 1>> trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 3, 3>> trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 3, 3> mat = lgmath::so3::hat(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> vec =
        trueVecs.at(i).cast<AUTODIFF_VAR_TYPE>();

    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> testMatAutodiff(3, 3);
    testMatAutodiff = lgmath::so3::diff::hat(vec);

    Eigen::Matrix<double, 3, 3> testMat = testMatAutodiff.cast<double>();

    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2rot and rot2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareAnalyticalAndNumericVec2Rot) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<AUTODIFF_VAR_TYPE, 3>> trueVecs;
  Eigen::Vector<AUTODIFF_VAR_TYPE, 3> temp(3);
  temp << 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << -lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, -lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.5 * lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.5 * lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> rand(3);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc matrices
  std::vector<Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>> analyticRots;
  for (unsigned i = 0; i < numTests; i++) {
    analyticRots.push_back(lgmath::so3::diff::vec2rot(trueVecs.at(i)));
  }

  // Compare analytical and numeric result
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> numericTran =
          lgmath::so3::diff::vec2rot(trueVecs.at(i), 20);
      std::cout << "ana: " << analyticRots.at(i) << std::endl;
      std::cout << "num: " << numericTran << std::endl;
      EXPECT_TRUE(lgmath::common::diff::nearEqual(analyticRots.at(i),
                                                  numericTran, 1e-6));
    }
  }

  // Test rot2vec
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Vector<AUTODIFF_VAR_TYPE, 3> testVec =
          lgmath::so3::diff::rot2vec(analyticRots.at(i));
      std::cout << "true: " << trueVecs.at(i) << std::endl;
      std::cout << "func: " << testVec << std::endl;
      EXPECT_TRUE(
          lgmath::common::diff::nearEqualAxisAngle(trueVecs.at(i), testVec, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareAnalyticalJacobInvAndNumericCounterpartsInSO3) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<AUTODIFF_VAR_TYPE, 3>> trueVecs;
  Eigen::Vector<AUTODIFF_VAR_TYPE, 3> temp(3);
  temp << 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << -lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, -lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.5 * lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.5 * lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> rand(3);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc analytical matrices
  std::vector<Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>> analyticJacs;
  std::vector<Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>> analyticJacInvs;
  for (unsigned i = 0; i < numTests; i++) {
    analyticJacs.push_back(lgmath::so3::diff::vec2jac(trueVecs.at(i)));
    analyticJacInvs.push_back(lgmath::so3::diff::vec2jacinv(trueVecs.at(i)));
  }

  // Compare inversed analytical and analytical inverse
  for (unsigned i = 0; i < numTests; i++) {
    std::cout << "ana: " << analyticJacs.at(i) << std::endl;
    std::cout << "num: " << analyticJacInvs.at(i) << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(analyticJacs.at(i).inverse(),
                                                analyticJacInvs.at(i), 1e-6));
  }

  // Compare analytical and 'numerical' jacobian
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> numericJac(3, 3);
    numericJac = lgmath::so3::vec2jac(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(
        lgmath::common::diff::nearEqual(analyticJacs.at(i), numericJac, 1e-6));
  }

  // Compare analytical and 'numerical' jacobian inverses
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> numericJac(3, 3);
    numericJac = lgmath::so3::vec2jacinv(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacInvs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(analyticJacInvs.at(i),
                                                numericJac, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability. f = J(xi) varpi, tests df/dvarpi 
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative1) {
  const unsigned numTests = 20;

  std::vector<Eigen::Vector<AUTODIFF_VAR_TYPE, 3>> xis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> rand(3);
    rand.setRandom();
    xis.push_back(rand);
  }

  std::vector<Eigen::Vector<AUTODIFF_VAR_TYPE, 3>> varpis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> rand(3);
    rand.setRandom();
    varpis.push_back(rand);
  }

  autodiff::Matrix2real test;

  auto func = [](const Eigen::Vector<AUTODIFF_VAR_TYPE, 3> &xi,
                 const Eigen::Vector<AUTODIFF_VAR_TYPE, 3> &varpi)
      -> Eigen::Vector<AUTODIFF_VAR_TYPE, 3> {
    return lgmath::so3::diff::vec2jac(xi) * varpi;
  };

  std::cout << varpis.at(0) << std::endl;

  for (unsigned i = 0; i < numTests; i++) {
#ifdef AUTODIFF_USE_FORWARD
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> F;

    auto func_jacobian =
        autodiff::jacobian(func, autodiff::wrt(varpis.at(i)),
                           autodiff::at(xis.at(i), varpis.at(i)), F);
#else
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> u = func(xis.at(i), varpis.at(i));
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> func_jacobian;
    for (int n = 0; n < 3; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), varpis.at(i));
    }
#endif
    auto expected = lgmath::so3::diff::vec2jac(xis.at(i));

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions. f = rot2vec(vec2rot(xi)), tests df/dxi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative2) {
  std::vector<Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 1> > xis;
  Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI;
  xis.push_back(temp);
  temp << -lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, -lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI;
  xis.push_back(temp);
  temp << lgmath::constants::PI_DIV_TWO, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, lgmath::constants::PI_DIV_TWO, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI_DIV_TWO;
  xis.push_back(temp);
  temp << lgmath::constants::TWO_PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, lgmath::constants::TWO_PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::TWO_PI;
  xis.push_back(temp);
  temp << -lgmath::constants::TWO_PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, -lgmath::constants::TWO_PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::TWO_PI;
  xis.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    xis.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  const unsigned numTests = xis.size();

  auto func = [](const Eigen::Vector<AUTODIFF_VAR_TYPE, 3> &xi)
      -> Eigen::Vector<AUTODIFF_VAR_TYPE, 3> {
    return lgmath::so3::diff::rot2vec(lgmath::so3::diff::vec2rot(xi));
  };

  for (unsigned i = 0; i < numTests; i++) {
#ifdef AUTODIFF_USE_FORWARD
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> F;
    auto func_jacobian = autodiff::jacobian(func, autodiff::wrt(xis.at(i)),
                                            autodiff::at(xis.at(i)), F);
#else
    Eigen::Vector<AUTODIFF_VAR_TYPE, 3> u = func(xis.at(i));
    Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3> func_jacobian;
    for (int n = 0; n < 3; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), xis.at(i));
    }
#endif
    auto expected = Eigen::Matrix<AUTODIFF_VAR_TYPE, 3, 3>::Identity(3, 3);

    std::cout << "xis.at(i): " << xis.at(i) << std::endl;

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
