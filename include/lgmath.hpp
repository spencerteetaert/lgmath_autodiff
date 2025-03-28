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
#if AUTODIFF_USE_BACKWARD
#include <lgmath/se3/OperationsAutodiffBackward.hpp>
#include <lgmath/so3/OperationsAutodiffBackward.hpp>
#else
#include <lgmath/se3/OperationsAutodiffForward.hpp>
#include <lgmath/so3/OperationsAutodiffForward.hpp>
#endif
#endif