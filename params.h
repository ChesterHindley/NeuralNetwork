#pragma once
namespace params {
	inline constexpr int testMax = 10000;
	inline constexpr int learnMax = 60000;


	inline constexpr int epochs = 50;
	inline constexpr auto alpha = 0.01;
	inline constexpr int learnSeriesNum = 60000;
	inline constexpr int testSeriesNum = 10000;



	static_assert(testSeriesNum <= testMax);
	static_assert(learnSeriesNum <= learnMax);


};