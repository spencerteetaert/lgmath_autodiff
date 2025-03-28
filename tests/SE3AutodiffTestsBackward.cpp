//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SE3AutodiffTests.cpp
/// \brief Unit tests for the autodiff implementation of the SE3 Lie Group math.
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
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/OperationsAutodiffBackward.hpp>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using autodiff::var; 
/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(3) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, Test4x4HatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 6, 1>> trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 4>> trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 4> mat = lgmath::se3::hat(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> vec =
        trueVecs.at(i).cast<var>();

    Eigen::Matrix<var, 4, 4> testMatAutodiff(4, 4);
    testMatAutodiff = lgmath::se3::hat(vec);

    Eigen::Matrix<double, 4, 4> testMat = testMatAutodiff.cast<double>();

    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) curlyhat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestCurlyHatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 6, 1>> trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 6>> trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> mat = lgmath::se3::curlyhat(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> vec =
        trueVecs.at(i).cast<var>();

    Eigen::Matrix<var, 6, 6> testMatAutodiff(6, 6);
    testMatAutodiff = lgmath::se3::curlyhat(vec);

    Eigen::Matrix<double, 6, 6> testMat = testMatAutodiff.cast<double>();
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 4x6 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestPointTo4x6MatrixFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 4, 1>> trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 6>> trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 6> mat =
        lgmath::se3::point2fs(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 4> vec =
        trueVecs.at(i).cast<var>();
    Eigen::Matrix<var, 4, 6> testMat =
        lgmath::se3::point2fs(vec.head<3>(), vec[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i),
                                          testMat.cast<double>(), 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 6x4 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestPointTo6x4MatrixFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 4, 1>> trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 4>> trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 4> mat =
        lgmath::se3::point2sf(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 4> vec =
        trueVecs.at(i).cast<var>();
    Eigen::Matrix<var, 6, 4> testMat =
        lgmath::se3::point2sf(vec.head<3>(), vec[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i),
                                          testMat.cast<double>(), 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2tran and tran2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareSE3Vec2TranAndTran2Vec) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<var, 6>> trueVecs;
  std::vector<Eigen::Matrix<double, 4, 4>> trueMats;
  Eigen::Vector<var, 6> temp(6);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 4> mat =
        lgmath::se3::vec2tran(trueVecs.at(i).cast<double>());
    trueMats.push_back(mat);
  }

  // Compare analytical vec2tran
  {
    std::cout << "Analytic results: " << std::endl;
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 4, 4> numericTran =
          lgmath::se3::vec2tran(trueVecs.at(i).cast<double>());
      Eigen::Matrix<var, 4, 4> numericTranDiff =
          lgmath::se3::vec2tran(trueVecs.at(i));
      std::cout << "vec: \n" << trueVecs.at(i) << std::endl;
      std::cout << "non-diff: \n" << numericTran << std::endl;
      std::cout << "diff: \n" << numericTranDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericTran, numericTranDiff.cast<double>(), 1e-6));
    }
  }

  // Compare numeric vec2tran
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 4, 4> numericTran =
          lgmath::se3::vec2tran(trueVecs.at(i).cast<double>(), 20);
      Eigen::Matrix<var, 4, 4> numericTranDiff =
          lgmath::se3::vec2tran(trueVecs.at(i), 20);
      std::cout << "non-diff numeric: " << numericTran << std::endl;
      std::cout << "diff numeric: " << numericTranDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericTran, numericTranDiff.cast<double>(), 1e-6));
    }
  }

  // Compare tran2vec
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 1> numericVec =
          lgmath::se3::tran2vec(trueMats.at(i));
      Eigen::Matrix<var, 6, 1> numericVecDiff =
          lgmath::se3::tran2vec(
              Eigen::Matrix<var, 4, 4>(trueMats.at(i)));
      std::cout << "non-diff numeric: " << numericVec << std::endl;
      std::cout << "diff numeric: " << numericVecDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericVec, numericVecDiff.cast<double>(), 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareSE3Vec2JacAndVec2JacInv) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<var, 6>> trueVecs;
  Eigen::Vector<var, 6> temp(6);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Compare analytical vec2jac
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 6> analyticJac =
          lgmath::se3::vec2jac(trueVecs.at(i).cast<double>());
      Eigen::Matrix<var, 6, 6> analyticJacDiff =
          lgmath::se3::vec2jac(trueVecs.at(i));
      std::cout << "non-diff: " << analyticJac << std::endl;
      std::cout << "diff: " << analyticJacDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          analyticJac, analyticJacDiff.cast<double>(), 1e-6));
    }
  }

  // Compare numeric vec2jac
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 6> numericJac =
          lgmath::se3::vec2jac(trueVecs.at(i).cast<double>(), 20);
      Eigen::Matrix<var, 6, 6> numericJacDiff =
          lgmath::se3::vec2jac(trueVecs.at(i), 20);
      std::cout << "non-diff: " << numericJac << std::endl;
      std::cout << "diff: " << numericJacDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericJac, numericJacDiff.cast<double>(), 1e-6));
    }
  }

  // Compare analytic vec2jacinv
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 6> analyticJacInv =
          lgmath::se3::vec2jacinv(trueVecs.at(i).cast<double>());
      Eigen::Matrix<var, 6, 6> analyticJacInvDiff =
          lgmath::se3::vec2jacinv(trueVecs.at(i));
      std::cout << "non-diff: " << analyticJacInv << std::endl;
      std::cout << "diff: " << analyticJacInvDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          analyticJacInv, analyticJacInvDiff.cast<double>(), 1e-6));
    }
  }

  // Compare numeric vec2jacinv
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 6> numericJacInv =
          lgmath::se3::vec2jacinv(trueVecs.at(i).cast<double>(), 20);
      Eigen::Matrix<var, 6, 6> numericJacInvDiff =
          lgmath::se3::vec2jacinv(trueVecs.at(i), 20);
      std::cout << "non-diff: " << numericJacInv << std::endl;
      std::cout << "diff: " << numericJacInvDiff << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          numericJacInv, numericJacInvDiff.cast<double>(), 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of adjoint tranformation identity, Ad(T(v)) = I +
/// curlyhat(v)*J(v)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestIdentityAdTvEqualIPlusCurlyHatvTimesJv) {
  // Add vectors to be tested
  std::vector<Eigen::Vector<var, 6>> trueVecs;
  Eigen::Vector<var, 6> temp(6);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Test Identity
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<var, 6, 6> lhs =
        lgmath::se3::tranAd(lgmath::se3::vec2tran(trueVecs.at(i)));
    Eigen::Matrix<var, 6, 6> rhs =
        Eigen::Matrix<var, 6, 6>::Identity(6, 6) +
        lgmath::se3::curlyhat(trueVecs.at(i)) *
            lgmath::se3::vec2jac(trueVecs.at(i));
    std::cout << "lhs: " << lhs << std::endl;
    std::cout << "rhs: " << rhs << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(lhs, rhs, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability. f = J(xi) varpi, tests df/dvarpi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestSE3Derivative1) {
  const unsigned numTests = 20;

  std::vector<Eigen::Vector<var, 6>> xis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    xis.push_back(rand);
  }

  std::vector<Eigen::Vector<var, 6>> varpis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    varpis.push_back(rand);
  }

  auto func = [](const Eigen::Vector<var, 6> &xi,
                 const Eigen::Vector<var, 6> &varpi)
      -> Eigen::Vector<var, 6> {
    return lgmath::se3::vec2jac(xi) * varpi;
  };

  std::cout << varpis.at(0) << std::endl;

  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> u = func(xis.at(i), varpis.at(i));
    Eigen::Matrix<var, 6, 6> func_jacobian;
    for (int n = 0; n < 6; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), varpis.at(i));
    }
    auto expected = lgmath::se3::vec2jac(xis.at(i));

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(expected.cast<double>(),
                                          func_jacobian, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test of derivative of vec2tran
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TESTVec2TranDerivative) {
  std::vector<Eigen::Matrix<var, 6, 1>> xis;
  std::vector<Eigen::Matrix<var, 16, 6>> expectedMats;
  Eigen::Matrix<var, 6, 1> temp;
  Eigen::Matrix<var, 6, 16> tempMat;
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI;
  xis.push_back(temp);
  tempMat.setZero();
  expectedMats.push_back(tempMat.transpose());
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  xis.push_back(temp);
  tempMat << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  expectedMats.push_back(tempMat.transpose());

  const unsigned numTests = xis.size();

  auto func = [](const Eigen::Vector<var, 6> &xi)
      -> Eigen::Vector<var, 16> {
    auto result = lgmath::se3::vec2tran(xi);
    Eigen::Map<const Eigen::Matrix<var, 16, 1>> resultMap(
        result.data(), result.size());

    return resultMap;
  };

  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 16> F;
    auto func_jacobian = autodiff::jacobian(func, autodiff::wrt(xis.at(i)),
                                            autodiff::at(xis.at(i)), F);

    std::cout << "xis.at(i):\n" << xis.at(i) << std::endl;
    std::cout << "F:\n" << F << std::endl;
    std::cout << "returned grad:\n" << func_jacobian << std::endl;
    std::cout << "expected:\n" << expectedMats.at(i) << std::endl;

    EXPECT_TRUE(lgmath::common::nearEqual(expectedMats.at(i).cast<double>(),
                                          func_jacobian, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions. f =
/// tran2vec(vec2tran(xi)), tests df/dxi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestSE3Derivative2) {
  std::vector<Eigen::Matrix<var, 6, 1>> xis;
  Eigen::Matrix<var, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI, 0.0;
  xis.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * lgmath::constants::PI;
  xis.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    xis.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  const unsigned numTests = xis.size();

  auto func = [](const Eigen::Vector<var, 6> &xi)
      -> Eigen::Vector<var, 6> {
    return lgmath::se3::tran2vec(lgmath::se3::vec2tran(xi));
  };

  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> u = func(xis.at(i));
    Eigen::Matrix<var, 6, 6> func_jacobian;
    for (int n = 0; n < 6; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), xis.at(i));
    }
    Eigen::Matrix<double, 6, 6> identityMat =
        Eigen::Matrix<double, 6, 6>::Identity(6, 6);
    Eigen::Matrix<double, 6, 6> zeroMat =
        Eigen::Matrix<double, 6, 6>::Zero(6, 6);

    std::cout << "xis.at(i): " << xis.at(i) << std::endl;
    std::cout << "F: " << F << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqualLieAlg(xis.at(i), F, 1e-6));

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

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions. f = T(xi) * a,
/// tests df/dxi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestSE3Derivative3) {
  const unsigned numTests = 20;

  std::vector<Eigen::Vector<var, 6>> xis;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> rand(6);
    rand.setRandom();
    xis.push_back(rand);
  }

  Eigen::Vector<var, 4> a;
  a << 1.0, 1.0, 1.0, 1.0;
  auto func = [a](const Eigen::Vector<var, 6> &xi)
      -> Eigen::Vector<var, 4> {
    return lgmath::se3::vec2tran(xi) * a;
  };

  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector<var, 6> u = func(xis.at(i));
    Eigen::Matrix<var, 4, 6> func_jacobian;
    for (int n = 0; n < 4; ++n) {
      func_jacobian.row(n) = autodiff::gradient(u(n), xis.at(i));
    }

    Eigen::Matrix<double, 4, 4> T =
        lgmath::se3::vec2tran(xis.at(i).cast<double>());
    Eigen::Vector<double, 4> temp_vec = T * a.cast<double>();
    Eigen::Matrix<double, 4, 6> p2fs =
        lgmath::se3::point2fs(temp_vec.topRows(3), double(temp_vec(3)));
    Eigen::Matrix<double, 6, 6> jac =
        lgmath::se3::vec2jac(xis.at(i).cast<double>());
    Eigen::Matrix<double, 4, 6> expected = p2fs * jac;

    std::cout << "temp_vec: " << temp_vec << std::endl;
    std::cout << "p2fs: " << p2fs << std::endl;
    std::cout << "jac: " << jac << std::endl;

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;

    std::cout << "output: "
              << lgmath::common::nearEqual(expected, func_jacobian, 1e-6);

    EXPECT_TRUE(lgmath::common::nearEqual(expected, func_jacobian, 1e-6));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
#endif