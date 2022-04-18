
#ifndef TIMER_H
#define TIMER_H


#include "defines.h"


enum class TimerType {
	MICROSECONDS, MILISECONDS
};


template <typename T>
struct TimeInfo {
	T duration;
	std::string str;

	TimeInfo(T _dur, std::string _str) :
		duration(_dur),
		str(_str)
	{ }

};


template<TimerType T>
static auto getProperTimeType(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop) {
	if constexpr (T == TimerType::MICROSECONDS) {
		return TimeInfo<std::chrono::microseconds>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start), "microseconds");
	}
	else if constexpr (T == TimerType::MILISECONDS) {
		return TimeInfo<std::chrono::milliseconds>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start), "miliseconds");
	}
}


template<TimerType T>
class Timer {
public:

	void start() {
		startTimepoint = std::chrono::high_resolution_clock::now();
	}

	u64 stop(b8 printValue = true) {
		stopTimepoint = std::chrono::high_resolution_clock::now();
		auto timerInfo = getProperTimeType<T>(startTimepoint, stopTimepoint);
		const u64 count{ (u64)timerInfo.duration.count() };
		if (printValue) {
			std::cerr << "Timer was on for " << count << " " << timerInfo.str << "\n";
		}
		return count;
	}

private:

	std::chrono::steady_clock::time_point startTimepoint;
	std::chrono::steady_clock::time_point stopTimepoint;

};


#endif
