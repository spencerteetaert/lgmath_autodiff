//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SO3AutodiffTests.cpp
/// \brief Unit tests for the autodiff implementation of the SO3 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both
/// special cases,
///          and randomly generated cases.
///
/// \author Spencer Teetaert
//////////////////////////////////////////////////////////////////////////////////////////////

#if USE_AUTODIFF
#if USE_AUTODIFF_BACKWARD
#include <gtest/gtest.h>

#include <math.h>
#include <iomanip>
#include <ios>
#include <iostream>

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/OperationsAutodiffBackward.hpp>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using autodiff::var;

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
    Eigen::Vector<var, 3> vec = trueVecs.at(i).cast<var>();

    Eigen::Matrix<var, 3, 3> testMatAutodiff(3, 3);
    testMatAutodiff = lgmath::so3::hat(vec);

    Eigen::Matrix<double, 3, 3> testMat = testMatAutodiff.cast<double>();

    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2rot and rot2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareSO3Vec2RotAndRot2Vec) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<double, 3>> trueVecs;
  std::vector<Eigen::Matrix<double, 3, 3>> trueMats;
  Eigen::Vector<double, 3> temp(3);
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
    Eigen::Vector<double, 3> rand(3);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 3, 3> mat =
        lgmath::so3::vec2rot(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Compare analytical vec2rot
  {
    std::cout << "\n\nCompare analytical vec2rot\n";
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> numericRot =
          lgmath::so3::vec2rot(trueVecs.at(i));
      Eigen::Matrix<var, 3, 3> numericRotDiff =
          lgmath::so3::vec2rot(trueVecs.at(i).cast<var>());
      std::cout << "non-diff: " << numericRot << std::endl;
      std::cout << "diff: " << numericRotDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual<var>(
          numericRot.cast<var>(), numericRotDiff, 1e-6));
    }
  }

  // Compare alternate vec2rot
  {
    std::cout << "\n\nCompare alternate vec2rot\n";
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> rot, jac;
      lgmath::so3::vec2rot(trueVecs.at(i), &rot, &jac);

      Eigen::Matrix<var, 3, 3> rotDiff, jacDiff;
      lgmath::so3::vec2rot(trueVecs.at(i).cast<var>(), rotDiff, jacDiff);
      std::cout << "non-diff rot: " << rot << std::endl;
      std::cout << "diff rot: " << rotDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(rot.cast<var>(), rotDiff, 1e-6));
      std::cout << "non-diff jac: " << jac << std::endl;
      std::cout << "diff jac: " << jacDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(jac.cast<var>(), jacDiff, 1e-6));
    }
  }

  // Compare numeric vec2rot
  {
    std::cout << "\n\nCompare numeric vec2rot\n";
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> numericRot =
          lgmath::so3::vec2rot(trueVecs.at(i), 20);
      Eigen::Matrix<var, 3, 3> numericRotDiff =
          lgmath::so3::vec2rot(trueVecs.at(i).cast<var>(), 20);
      std::cout << "non-diff numeric: " << numericRot << std::endl;
      std::cout << "diff numeric: " << numericRotDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericRot.cast<var>(), numericRotDiff, 1e-6));
    }
  }

  // Compare rot2vec
  {
    std::cout << "\n\nCompare rot2vec\n";
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 1> numericVec =
          lgmath::so3::rot2vec(trueMats.at(i));
      Eigen::Matrix<var, 3, 1> numericVecDiff =
          lgmath::so3::rot2vec(Eigen::Matrix<var, 3, 3>(trueMats.at(i)));
      std::cout << "non-diff numeric: " << numericVec << std::endl;
      std::cout << "diff numeric: " << numericVecDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericVec.cast<var>(), numericVecDiff, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareSO3Vec2JacAndVec2JacInv) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<double, 3>> trueVecs;
  Eigen::Vector<double, 3> temp(3);
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
    Eigen::Vector<double, 3> rand(3);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Compare analytical vec2jac
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> analyticJac =
          lgmath::so3::vec2jac(trueVecs.at(i));
      Eigen::Matrix<var, 3, 3> analyticJacDiff =
          lgmath::so3::vec2jac(trueVecs.at(i).cast<var>());
      std::cout << "non-diff: " << analyticJac << std::endl;
      std::cout << "diff: " << analyticJacDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          analyticJac.cast<var>(), analyticJacDiff, 1e-6));
    }
  }

  // Compare numeric vec2jac
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> numericJac =
          lgmath::so3::vec2jac(trueVecs.at(i), 20);
      Eigen::Matrix<var, 3, 3> numericJacDiff =
          lgmath::so3::vec2jac(trueVecs.at(i).cast<var>(), 20);
      std::cout << "non-diff: " << numericJac << std::endl;
      std::cout << "diff: " << numericJacDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericJac.cast<var>(), numericJacDiff, 1e-6));
    }
  }

  // Compare analytic vec2jacinv
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> analyticJacInv =
          lgmath::so3::vec2jacinv(trueVecs.at(i));
      Eigen::Matrix<var, 3, 3> analyticJacInvDiff =
          lgmath::so3::vec2jacinv(trueVecs.at(i).cast<var>());
      std::cout << "non-diff: " << analyticJacInv << std::endl;
      std::cout << "diff: " << analyticJacInvDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          analyticJacInv.cast<var>(), analyticJacInvDiff, 1e-6));
    }
  }

  // Compare numeric vec2jacinv
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 3> numericJacInv =
          lgmath::so3::vec2jacinv(trueVecs.at(i), 20);
      Eigen::Matrix<var, 3, 3> numericJacInvDiff =
          lgmath::so3::vec2jacinv(trueVecs.at(i).cast<var>(), 20);
      std::cout << "non-diff: " << numericJacInv << std::endl;
      std::cout << "diff: " << numericJacInvDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericJacInv.cast<var>(), numericJacInvDiff, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability. f = J(xi) varpi, tests df/dvarpi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestSO3Derivative1) {
  const unsigned numTests = 20;

  std::vector<Eigen::Vector<double, 3>> xis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<double, 3> rand(3);
    rand.setRandom();
    xis.push_back(rand);
  }

  std::vector<Eigen::Vector<double, 3>> varpis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<double, 3> rand(3);
    rand.setRandom();
    varpis.push_back(rand);
  }

  auto func = [](const Eigen::Vector<var, 3> &xi,
                 const Eigen::Vector<var, 3> &varpi) -> Eigen::Vector<var, 3> {
    return lgmath::so3::vec2jac(xi) * varpi;
  };

  std::cout << varpis.at(0) << std::endl;

  for (unsigned i = 0; i < numTests; i++) {
    auto varpi = varpis.at(i).cast<var>(); 
    Eigen::Vector<var, 3> u = func(xis.at(i).cast<var>(), varpi);
    Eigen::Matrix<double, 3, 3> func_jacobian;
    for (int n = 0; n < 3; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), varpi);
    }
    auto expected = lgmath::so3::vec2jac(xis.at(i));

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(
        lgmath::common::nearEqual(expected, func_jacobian.cast<var>(), 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions. f =
/// rot2vec(vec2rot(xi)), tests df/dxi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestSO3Derivative2) {
  std::vector<Eigen::Matrix<double, 3, 1>> xis;
  Eigen::Matrix<double, 3, 1> temp;
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
  temp << 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << lgmath::constants::PI_DIV_TWO, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, lgmath::constants::PI_DIV_TWO, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI_DIV_TWO;
  xis.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    xis.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  const unsigned numTests = xis.size();

  auto func = [](const Eigen::Vector<var, 3> &xi) -> Eigen::Vector<var, 3> {
    return lgmath::so3::rot2vec(lgmath::so3::vec2rot(xi));
  };

  for (unsigned i = 0; i < numTests; i++) {
    auto xi = xis.at(i).cast<var>();
    Eigen::Vector<var, 3> u = func(xi);
    Eigen::Matrix<double, 3, 3> func_jacobian;
    for (int n = 0; n < 3; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), xi);
    }
    auto identityMat = Eigen::Matrix<double, 3, 3>::Identity(3, 3);
    auto zeroMat = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

    std::cout << "xis.at(i): " << xis.at(i) << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    if (i >= 6) {
      std::cout << "expected: " << identityMat << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(identityMat, func_jacobian, 1e-6));
    } else {
      std::cout << "expected: " << zeroMat << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(zeroMat, func_jacobian, 1e-6));
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
#endif