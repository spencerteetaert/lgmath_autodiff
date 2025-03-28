/**
 * \file lgmath.hpp
 * \brief Convenience Header
 *
 * \author Sean Anderson (autodiff support by Spencer Teetaert) 
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
#ifdef USE_AUTODIFF
#ifdef USE_AUTODIFF_BACKWARD
#include <lgmath/se3/OperationsAutodiffBackward.hpp>
#include <lgmath/so3/OperationsAutodiffBackward.hpp>
#else
#include <lgmath/se3/OperationsAutodiffForward.hpp>
#include <lgmath/so3/OperationsAutodiffForward.hpp>
#endif
#endif