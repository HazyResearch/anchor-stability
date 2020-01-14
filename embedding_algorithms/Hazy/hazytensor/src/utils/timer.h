//  Utility file to time methods.
//  
//  Example usage to time 'method1':
//    const auto tic = timer::start_clock();
//    method1();
//    timer::stop_clock("method1", tic);

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

namespace timer {

inline std::chrono::time_point<std::chrono::system_clock> start_clock() {
  return std::chrono::system_clock::now();
}

inline double stop_clock(
    const std::chrono::time_point<std::chrono::system_clock> t_in) {
  std::chrono::duration<double> elapsed_seconds =
      std::chrono::system_clock::now() - t_in;
  return elapsed_seconds.count();
}

inline double stop_clock(
    const std::string in,
    const std::chrono::time_point<std::chrono::system_clock> t_in) {
  double t2 = stop_clock(t_in);
  printf("Time[%s]: %fs\n", in.c_str(), t2);
  return t2;
}

}  // namespace timer

#endif