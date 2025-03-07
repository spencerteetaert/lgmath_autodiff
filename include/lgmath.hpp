/**
 * \file lgmath.hpp
 * \brief Convenience Header
 *
 * \author Sean Anderson
 */
#pragma once

// SO2
// todo

// SE2
// todo

// SO3
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/Rotation.hpp>
#include <lgmath/so3/Types.hpp>

// SE3
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se3/Types.hpp>

// R3
#include <lgmath/r3/Operations.hpp>
#include <lgmath/r3/Types.hpp>

// Autodiff 
#if USE_AUTODIFF
#include <lgmath/se3/OperationsAutodiff.hpp>
#include <lgmath/so3/OperationsAutodiff.hpp>
#endif