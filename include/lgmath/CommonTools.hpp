/**
 * \file CommonTools.hpp
 * \brief A header-only helper file with a few common tools.
 * \details Implements a basic timer tool.
 *
 * \author Sean Anderson, ASRL (autodiff support by Spencer Teetaert) 
 */
#pragma once

// system timer
#include <sys/time.h>
#include <iostream>

namespace lgmath {
namespace common {

/** \brief Simple wall timer class to get approximate timings of functions */
class Timer {
 public:
  /** \brief Default constructor */
  Timer() { reset(); }

  /** \brief Reset timer */
  void reset() { beg_ = this->get_wall_time(); }

  /** \brief Get seconds since last reset */
  double seconds() const { return this->get_wall_time() - beg_; }

  /** \brief Get milliseconds since last reset */
  double milliseconds() const { return 1e3 * (this->get_wall_time() - beg_); }

  /** \brief Get microseconds since last reset */
  double microseconds() const { return 1e6 * (this->get_wall_time() - beg_); }

  /** \brief Get nanoseconds since last reset */
  double nanoseconds() const { return 1e9 * (this->get_wall_time() - beg_); }

 private:
  /** \brief Get current wall time */
  double get_wall_time() const {
    struct timeval time;
    //  Handle error
    if (gettimeofday(&time, NULL)) {
      return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
  }

  /** \brief Wall time at reset */
  double beg_;
};

}  // namespace common
}  // namespace lgmath
