#ifndef LEARNCPP_TIMER_H
#define LEARNCPP_TIMER_H

#include <chrono> // for std::chrono functions

class Timer
{
private:
    // Type aliases to make accessing nested type easier
    using clock_t = std::chrono::steady_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<clock_t> begin;

public:
    Timer() : begin(clock_t::now())
    {
    }

    void reset()
    {
      begin = clock_t::now();
    }

    double elapsed() const
    {
      return std::chrono::duration_cast<second_t>(clock_t::now() - begin).count();
    }
};

#endif //LEARNCPP_TIMER_H
