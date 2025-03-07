//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SE3AutodiffTests.cpp
/// \brief Unit tests for the autodiff implementation of the SE3 Lie Group math.
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
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/OperationsAutodiff.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/OperationsAutodiff.hpp>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

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
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 4> mat = lgmath::se3::hat(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal vec = trueVecs.at(i).cast<autodiff::real>();

    autodiff::MatrixXreal testMatAutodiff(4, 4);
    testMatAutodiff = lgmath::se3::diff::hat(vec);

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
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> mat = lgmath::se3::curlyhat(trueVecs.at(i));
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal vec = trueVecs.at(i).cast<autodiff::real>();

    autodiff::MatrixXreal testMatAutodiff(6, 6);
    testMatAutodiff = lgmath::se3::diff::curlyhat(vec);

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
  std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 6> mat =
        lgmath::se3::point2fs(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::Vector4real vec = trueVecs.at(i).cast<autodiff::real>();
    Eigen::Matrix<double, 4, 6> testMat =
        lgmath::se3::diff::point2fs(vec.head<3>(), vec[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 6x4 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestPointTo6x4MatrixFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 4> mat =
        lgmath::se3::point2sf(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::Vector4real vec = trueVecs.at(i).cast<autodiff::real>();
    Eigen::Matrix<double, 6, 4> testMat =
        lgmath::se3::point2sf(vec.head<3>(), vec[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2tran and tran2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareAnalyticalAndNumericVec2Tran) {
  // Add vectors to be tested
  std::vector<autodiff::VectorXreal> trueVecs;
  autodiff::VectorXreal temp(6);
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
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc matrices
  std::vector<autodiff::Matrix4real> analyticTrans;
  for (unsigned i = 0; i < numTests; i++) {
    analyticTrans.push_back(lgmath::se3::diff::vec2tran(trueVecs.at(i)));
  }

  // Compare analytical and numeric result
  {
    for (unsigned i = 0; i < numTests; i++) {
      autodiff::Matrix4real numericTran =
          lgmath::se3::diff::vec2tran(trueVecs.at(i), 20);
      std::cout << "ana: " << analyticTrans.at(i) << std::endl;
      std::cout << "num: " << numericTran << std::endl;
      EXPECT_TRUE(lgmath::common::diff::nearEqual(analyticTrans.at(i),
                                                  numericTran, 1e-6));
    }
  }

  // Test rot2vec
  {
    for (unsigned i = 0; i < numTests; i++) {
      autodiff::VectorXreal testVec =
          lgmath::se3::diff::tran2vec(analyticTrans.at(i));
      std::cout << "true: " << trueVecs.at(i) << std::endl;
      std::cout << "func: " << testVec << std::endl;
      EXPECT_TRUE(
          lgmath::common::diff::nearEqualLieAlg(trueVecs.at(i), testVec, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, CompareAnalyticalJacobInvAndNumericCounterpartsInSE3) {
  // Add vectors to be tested
  std::vector<autodiff::VectorXreal> trueVecs;
  autodiff::VectorXreal temp(6);
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
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc analytical matrices
  std::vector<autodiff::MatrixXreal> analyticJacs;
  std::vector<autodiff::MatrixXreal> analyticJacInvs;
  for (unsigned i = 0; i < numTests; i++) {
    analyticJacs.push_back(lgmath::se3::diff::vec2jac(trueVecs.at(i)));
    analyticJacInvs.push_back(lgmath::se3::diff::vec2jacinv(trueVecs.at(i)));
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
    autodiff::MatrixXreal numericJac(6, 6);
    numericJac = lgmath::se3::vec2jac(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(
        lgmath::common::diff::nearEqual(analyticJacs.at(i), numericJac, 1e-6));
  }

  // Compare analytical and 'numerical' jacobian inverses
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::MatrixXreal numericJac(6, 6);
    numericJac = lgmath::se3::vec2jacinv(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacInvs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(analyticJacInvs.at(i),
                                                numericJac, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of adjoint tranformation identity, Ad(T(v)) = I +
/// curlyhat(v)*J(v)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestIdentityAdTvEqualIPlusCurlyHatvTimesJv) {
  // Add vectors to be tested
  std::vector<autodiff::VectorXreal> trueVecs;
  autodiff::VectorXreal temp(6);
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
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    trueVecs.push_back(rand);
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Test Identity
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::MatrixXreal lhs =
        lgmath::se3::diff::tranAd(lgmath::se3::diff::vec2tran(trueVecs.at(i)));
    autodiff::MatrixXreal rhs = autodiff::MatrixXreal::Identity(6, 6) +
                                lgmath::se3::diff::curlyhat(trueVecs.at(i)) *
                                    lgmath::se3::diff::vec2jac(trueVecs.at(i));
    std::cout << "lhs: " << lhs << std::endl;
    std::cout << "rhs: " << rhs << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(lhs, rhs, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative1) {
  const unsigned numTests = 20;

  std::vector<autodiff::VectorXreal> xis;
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    xis.push_back(rand);
  }
  std::vector<autodiff::VectorXreal> varpis;
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    varpis.push_back(rand);
  }

  auto func = [](const autodiff::VectorXreal &xi,
                 const autodiff::VectorXreal &varpi) -> autodiff::VectorXreal {
    return lgmath::se3::diff::vec2jac(xi) * varpi;
  };

  std::cout << varpis.at(0) << std::endl;

  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal F;
    auto func_jacobian =
        autodiff::jacobian(func, autodiff::wrt(varpis.at(i)),
                           autodiff::at(xis.at(i), varpis.at(i)), F);

    auto expected = lgmath::se3::diff::vec2jac(xis.at(i));

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative2) {
  const unsigned numTests = 20;

  std::vector<autodiff::VectorXreal> xis;
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    xis.push_back(rand);
  }

  auto func = [](const autodiff::VectorXreal &xi) -> autodiff::VectorXreal {
    return lgmath::se3::diff::tran2vec(lgmath::se3::diff::vec2tran(xi));
  };

  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal F;
    auto func_jacobian = autodiff::jacobian(func, autodiff::wrt(xis.at(i)),
                                            autodiff::at(xis.at(i)), F);

    auto expected = autodiff::MatrixXreal::Identity(6, 6);

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative3) {
  const unsigned numTests = 20;

  std::vector<autodiff::VectorXreal> xis;
  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal rand(6);
    rand.setRandom();
    xis.push_back(rand);
  }

  autodiff::Vector4real a;
  a << 1.0, 1.0, 1.0, 1.0;
  auto func = [a](const autodiff::VectorXreal &xi) -> autodiff::VectorXreal {
    return lgmath::se3::diff::vec2tran(xi) * a;
  };

  for (unsigned i = 0; i < numTests; i++) {
    autodiff::VectorXreal F;
    auto func_jacobian = autodiff::jacobian(func, autodiff::wrt(xis.at(i)),
                                            autodiff::at(xis.at(i)), F);

    autodiff::Matrix4real T = lgmath::se3::diff::vec2tran(xis.at(i));
    autodiff::Vector4real temp_vec = T * a;
    Eigen::Matrix<autodiff::real, 4, 6> p2fs =
        lgmath::se3::diff::point2fs(temp_vec.topRows(3), temp_vec(3));
    Eigen::Matrix<autodiff::real, 6, 6> jac =
        lgmath::se3::diff::vec2jac(xis.at(i));
    Eigen::Matrix<autodiff::real, 4, 6> expected = p2fs * jac;

    std::cout << "temp_vec: " << temp_vec << std::endl;
    std::cout << "p2fs: " << p2fs << std::endl;
    std::cout << "jac: " << lgmath::se3::diff::vec2jac(xis.at(i)) << std::endl;

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;

    std::cout << "output: "
              << lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6);

    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
