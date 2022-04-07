
#ifndef TIMER_H
#define TIMER_H


#include "defines.h"


enum class TimerType {
	MICROSECONDS
};


template<TimerType T>
class Timer {
public:

	void start() {
		startTimepoint = std::chrono::high_resolution_clock::now();
	}

	u64 stop(b8 printValue = true) {
		stopTimepoint = std::chrono::high_resolution_clock::now();
		if constexpr (T == TimerType::MICROSECONDS) {
			const auto duration =
				std::chrono::duration_cast<std::chrono::microseconds>(stopTimepoint - startTimepoint);
			const u64 microseconds{ (u64)duration.count() };
			if (printValue) {
				std::cerr << "Timer was on for " << microseconds << " microseconds\n";
			}
			return microseconds;
		}
		std::cerr << "Given wrong TimerType! Could not calculate time!\n";
		return 0;
	}

private:

	std::chrono::steady_clock::time_point startTimepoint;
	std::chrono::steady_clock::time_point stopTimepoint;

};


#endif
