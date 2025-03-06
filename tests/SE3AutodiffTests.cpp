//////////////////////////////////////////////////////////////////////////////////////////////
/// \file NaiveSE3Tests.cpp
/// \brief Unit tests for the naive implementation of the SE3 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both
/// special cases,
///          and randomly generated cases.
///
/// \author Sean Anderson
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

// /////////////////////////////////////////////////////////////////////////////////////////////
// /// \brief General test of homogeneous point to 4x6 matrix function
// /////////////////////////////////////////////////////////////////////////////////////////////
// TEST(LGMathAutodiff, TestPointTo4x6MatrixFunction) {
//   // Number of random tests
//   const unsigned numTests = 20;

//   // Add vectors to be tested - random
//   std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
//   for (unsigned i = 0; i < numTests; i++) {
//     trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
//   }

//   // Setup truth matrices
//   std::vector<Eigen::Matrix<double, 4, 6> > trueMats;
//   for (unsigned i = 0; i < numTests; i++) {
//     Eigen::Matrix<double, 4, 6> mat;
//     mat << trueVecs.at(i)[3], 0.0, 0.0, 0.0, trueVecs.at(i)[2],
//         -trueVecs.at(i)[1], 0.0, trueVecs.at(i)[3], 0.0, -trueVecs.at(i)[2],
//         0.0, trueVecs.at(i)[0], 0.0, 0.0, trueVecs.at(i)[3],
//         trueVecs.at(i)[1], -trueVecs.at(i)[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//         0.0;
//     trueMats.push_back(mat);
//   }

//   // Test the 3x1 function with scaling param
//   for (unsigned i = 0; i < numTests; i++) {
//     Eigen::Matrix<double, 4, 6> testMat =
//         lgmath::se3::point2fs(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
//     std::cout << "true: " << trueMats.at(i) << std::endl;
//     std::cout << "func: " << testMat << std::endl;
//     EXPECT_TRUE(lgmath::common::diff::nearEqual(trueMats.at(i), testMat,
//     1e-6));
//   }
// }

// /////////////////////////////////////////////////////////////////////////////////////////////
// /// \brief General test of homogeneous point to 6x4 matrix function
// /////////////////////////////////////////////////////////////////////////////////////////////
// TEST(LGMathAutodiff, TestPointTo6x4MatrixFunction) {
//   // Number of random tests
//   const unsigned numTests = 20;

//   // Add vectors to be tested - random
//   std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
//   for (unsigned i = 0; i < numTests; i++) {
//     trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
//   }

//   // Setup truth matrices
//   std::vector<Eigen::Matrix<double, 6, 4> > trueMats;
//   for (unsigned i = 0; i < numTests; i++) {
//     Eigen::Matrix<double, 6, 4> mat;
//     mat << 0.0, 0.0, 0.0, trueVecs.at(i)[0], 0.0, 0.0, 0.0,
//     trueVecs.at(i)[1],
//         0.0, 0.0, 0.0, trueVecs.at(i)[2], 0.0, trueVecs.at(i)[2],
//         -trueVecs.at(i)[1], 0.0, -trueVecs.at(i)[2], 0.0, trueVecs.at(i)[0],
//         0.0, trueVecs.at(i)[1], -trueVecs.at(i)[0], 0.0, 0.0;
//     trueMats.push_back(mat);
//   }

//   // Test the 3x1 function with scaling param
//   for (unsigned i = 0; i < numTests; i++) {
//     Eigen::Matrix<double, 6, 4> testMat =
//         lgmath::se3::point2sf(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
//     std::cout << "true: " << trueMats.at(i) << std::endl;
//     std::cout << "func: " << testMat << std::endl;
//     EXPECT_TRUE(lgmath::common::diff::nearEqual(trueMats.at(i), testMat,
//     1e-6));
//   }
// }

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
autodiff::VectorXreal func(const autodiff::VectorXreal &xi, const autodiff::VectorXreal &varpi)
{    
    auto J = lgmath::se3::diff::vec2jac(xi);
    return J * varpi;
}
/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of differentiability of simple functions
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMathAutodiff, TestDerivative) {
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

  for (unsigned i = 0; i < numTests; i++) {
    // auto func = [](const autodiff::VectorXreal& xi, const autodiff::VectorXreal& varpi) {
    //   auto J = lgmath::se3::diff::vec2jac(xi);
    //   return J * varpi;
    // };

    autodiff::VectorXreal F; 
    auto func_jacobian = autodiff::jacobian(func, autodiff::wrt(varpis.at(i)), autodiff::at(xis.at(i), varpis.at(i)), F);

    auto expected = lgmath::se3::diff::vec2jac(xis.at(i)); 

    std::cout << "expected: " << expected << std::endl;
    std::cout << "returned grad: " << func_jacobian << std::endl;
    EXPECT_TRUE(lgmath::common::diff::nearEqual(expected, func_jacobian, 1e-6));
  }
}

// TODO: Add more derivative tests. Ideally with larger expansions than single term taylor series.

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
