#include "Matrix.h"
#include <cassert>
#include <ranges>
#include "Model.h"
#include <fstream>
#include <iomanip>
#include "params.h"
#include <chrono>


void ch3t1() 
{
	Matrix input(
		{
			0.5,0.1,0.2,0.8,
			0.75,0.3,0.1,0.9,
			0.1,0.7,0.6,0.2
		}, 3, 4
	);

	Matrix Wh(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3);
	Matrix Wy(
		{
			0.7,0.9,-0.4,0.8,0.1,
			0.8,0.5,0.3,0.1,0.0,
			-0.3,0.9,0.3,0.1,-0.2
		}, 3, 5);
	Model m;
	Matrix expected(
		{
			0.376,0.082,0.053,0.49,
			0.3765,0.133,0.067,0.465,
			0.305,0.123,0.073,0.402
		}, 3, 4);

	m.addLayer(Layer(std::move(Wh), activationFunctions::ReLU,activationFunctions::dReLU)).addLayer(Layer(std::move(Wy),nullptr,nullptr));
	std::cout << m.predict(input);

}
void ch3t2()
{
	Matrix input(
		{
			0.5,0.1,0.2,0.8,
			0.75,0.3,0.1,0.9,
			0.1,0.7,0.6,0.2
		}, 3, 4
	);
	Matrix expected(
		{
			0.1,0.5,0.1,0.7,
			1.0,0.2,0.3,0.6,
			0.1,-0.5,0.2,0.2
		}, 3, 4);

	auto alpha = 0.01;

	Matrix Wh(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3);

	Matrix Wy(
		{
			0.7,0.9,-0.4,0.8,0.1,
			0.8,0.5,0.3,0.1,0.0,
			-0.3,0.9,0.3,0.1,-0.2
		}, 3, 5);
	Model m;
	m.addLayer(Layer(std::move(Wh),activationFunctions::ReLU, activationFunctions::dReLU)).addLayer(Layer(std::move(Wy), nullptr, nullptr));
	m.learn(50, input, expected, alpha);

}
void ch3t3()
{
	// msvc intrinsic function
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(40, -0.1, 0.1, activationFunctions::ReLU, activationFunctions::dReLU).addLayer(10,-0.1,0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*) &magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);
	
	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num<<'\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28*28*num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	//Matrix debugImg(28, 28);
	//for (int i = 0; i < 28; i++)
	//	for (int j = 0; j < 28; j++)
	//	{
	//		debugImg(i, j) = std::ceil(input(28 * i + j, 1));
	//	}
 //   std::cout << std::setprecision(1) << debugImg<<"\n";
	//std::cout << int(labels[1]);
	
	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i],i) = 1;
	}
	std::cout << labels[0];

	m.learn(params::epochs, input, expected,params::alpha);




	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);
	
	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predict(testinput);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num*100 << "%";
	
}
void ch3t4()
{
	std::vector<double> inputRaw{ 0.91, 0.82, 0.05, 0.91, 0.09, 0.26, 0.89, 0.90, 0.18, 0.82, 0.04, 0.08, 0.19, 0.91, 0.00, 0.20, 0.17, 0.90, 0.10, 0.97, 0.00, 0.22, 0.97, 0.03, 0.20, 0.20, 0.89, 0.03, 0.97, 0.21, 0.92, 0.28, 0.26, 0.00, 0.10, 0.86, 0.12, 0.17, 0.91, 0.94, 0.15, 0.06, 0.81, 0.25, 0.09, 0.00, 0.04, 0.88, 0.83, 0.95, 0.08, 0.84, 0.93, 0.06, 0.19, 0.17, 0.88, 0.03, 0.21, 0.93, 0.42, 0.89, 0.24, 0.15, 0.25, 0.90, 0.86, 0.99, 0.19, 0.29, 0.86, 0.10, 0.89, 0.84, 0.28, 0.82, 0.08, 0.13, 0.87, 0.16, 0.17, 0.25, 0.37, 0.85, 0.90, 0.10, 0.01, 0.08, 0.03, 0.87, 0.99, 0.92, 0.00, 0.88, 0.89, 0.28, 0.91, 0.00, 0.08, 0.01, 0.24, 0.95, 0.86, 0.00, 0.18, 0.89, 0.12, 0.21, 0.90, 0.88, 0.36, 0.22, 0.25, 0.90, 0.30, 0.95, 0.16, 0.19, 0.00, 0.85, 0.09, 0.02, 0.95, 0.88, 0.82, 0.08, 0.91, 0.15, 0.04, 0.95, 0.85, 0.07, 0.34, 0.88, 0.22, 0.88, 0.89, 0.28, 0.87, 0.18, 0.22, 0.83, 0.92, 0.11, 0.86, 0.34, 0.00, 0.00, 0.25, 0.92, 0.81, 0.24, 0.20, 0.27, 0.89, 0.07, 0.98, 0.16, 0.27, 0.90, 0.28, 0.05, 0.23, 0.28, 0.90, 0.17, 0.08, 0.93, 0.88, 0.90, 0.18, 0.95, 0.93, 0.00, 0.12, 0.10, 0.82, 0.03, 0.86, 0.19, 0.93, 0.26, 0.29, 0.99, 0.91, 0.01, 0.16, 0.14, 0.89, 0.93, 0.24, 0.24, 0.92, 0.91, 0.00, 0.91, 0.22, 0.26, 0.11, 0.79, 0.12, 0.91, 0.14, 0.02, 0.37, 0.98, 0.25, 0.14, 0.86, 0.16, 0.91, 0.18, 0.30, 0.93, 0.20, 0.08, 0.99, 0.87, 0.09, 0.13, 0.90, 0.20, 0.90, 0.29, 0.03, 0.77, 0.00, 0.25, 0.15, 0.17, 0.89, 0.85, 0.18, 0.10, 0.14, 0.85, 0.17, 0.21, 0.21, 0.90, 0.11, 0.90, 0.00, 0.89, 0.84, 0.18, 0.01, 0.30, 0.85, 0.21, 0.94, 0.13, 0.14, 0.94, 0.08, 0.86, 0.89, 0.21, 0.06, 0.20, 0.95, 0.86, 0.78, 0.26, 0.19, 1.00, 0.21, 0.21, 0.09, 0.89, 0.86, 0.93, 0.22, 0.08, 0.83, 0.21, 0.26, 0.87, 0.12, 0.84, 0.13, 0.17, 0.90, 0.28, 0.16, 0.82, 0.97, 0.12, 0.11, 0.95, 0.13, 0.94, 0.94, 0.16, 0.17, 0.86, 0.11, 0.91, 0.89, 0.01, 0.94, 0.90, 0.18, 0.00, 0.83, 0.21, 0.18, 0.87, 0.27, 0.90, 0.26, 0.09, 0.14, 0.90, 0.04, 0.86, 0.20, 0.00, 0.09, 0.12, 0.87, 0.97, 0.02, 0.13, 0.91, 0.19, 0.14 };
	Matrix input(3, 109);
	for (int c = 0; c < 109; c++)
		for (int r = 0; r < 3; r++)
			input(r, c) = inputRaw[r + c * 3];

	Model m;
	m.addLayer(16,-3,3,activationFunctions::ReLU,activationFunctions::dReLU).addLayer(4,-3,3);


	//change colors expected into neuron matrix form
	std::vector<double> expectedRaw(109 * 4);
	constexpr static std::array expectedColors = { 4 ,1 ,4 ,1 ,2 ,3 ,2 ,2 ,3 ,2 ,1 ,3 ,3 ,1 ,1 ,3 ,4 ,4 ,3 ,3 ,2 ,3 ,4 ,2 ,4 ,1 ,1 ,3 ,1 ,3 ,4 ,4 ,1 ,3 ,1 ,1 ,4 ,3 ,2 ,3 ,3 ,4 ,1 ,4 ,2 ,4 ,1 ,4 ,1 ,3 ,1 ,2 ,1 ,1 ,3 ,3 ,4 ,4 ,3 ,2 ,1 ,4 ,3 ,1 ,4 ,1 ,2 ,1 ,2 ,2 ,1 ,1 ,4 ,2 ,1 ,1 ,3 ,1 ,2 ,3 ,2 ,4 ,3 ,2 ,2 ,4 ,3 ,4 ,2 ,3 ,4 ,2 ,2 ,1 ,1 ,4 ,2 ,4 ,2 ,4 ,4 ,2 ,2 ,1 ,2 ,1 ,3 ,1 ,1 };
	for (int i = 0; i < 109; i++)
	{
		expectedRaw[i * 4 + expectedColors[i] - 1] = 1;
	}

	Matrix expected( 4, 109);
	for (int c = 0; c < 109; c++)
		for (int r = 0; r < 4; r++)
			expected(r, c) = expectedRaw[r + c * 4];

	
	static constexpr int iterations = 100;
	static constexpr double alpha = 0.001;
	m.learn(iterations, input, expected, alpha);
	std::cout << "Iterations: " << iterations << "\talpha: " << alpha;
	std::cout <<"\nMSE: " << m.getError() << '\n';


	// TODO some nice way to read data from files, also somehow enable column major initialization
	constexpr std::array expectedColorsTest = { 4, 2, 3, 2, 1, 3, 1, 3, 4, 3, 3, 3, 4, 3, 1, 4, 3, 3, 1, 4, 2, 3, 4, 1, 1, 4, 3, 3, 4, 1, 4, 3, 1, 2, 4, 1, 4, 3, 1, 4, 4, 4, 3, 2, 1, 4, 4, 1, 2, 2, 4, 2, 2, 2, 3, 1, 1, 1, 2, 2, 2, 4, 4, 2, 4, 4, 1, 1, 2, 1, 1, 4, 2, 3, 4, 4, 3, 3, 4, 3, 2, 4, 3, 1, 4, 1, 2, 4, 3, 3, 1, 2, 3, 1, 2, 2, 2, 4, 2, 3, 1, 3, 2, 1, 3, 2, 1, 1, 3, 3, 4, 3, 2, 1, 2, 3, 4, 3, 3, 1, 4, 2, 1, 1, 3, 1, 3, 1, 1, 4 };
	Matrix testData(3,130 );
	constexpr static std::array testDataRaw{ 0.84, 0.93, 0.28, 0.14, 0.88, 0.12, 0.16, 0.03, 0.88, 0.08, 0.89, 0.30, 0.85, 0.39, 0.09, 0.26, 0.19, 0.88, 0.91, 0.18, 0.05, 0.14, 0.07, 0.94, 0.93, 0.88, 0.06, 0.19, 0.14, 0.92, 0.13, 0.05, 0.90, 0.17, 0.15, 0.80, 0.87, 0.84, 0.09, 0.01, 0.05, 0.86, 0.81, 0.13, 0.27, 0.87, 0.92, 0.28, 0.04, 0.31, 0.83, 0.42, 0.21, 0.88, 0.86, 0.20, 0.46, 0.84, 0.88, 0.37, 0.00, 0.95, 0.32, 0.12, 0.00, 0.93, 0.85, 0.86, 0.25, 0.87, 0.14, 0.05, 0.83, 0.21, 0.26, 0.87, 0.85, 0.19, 0.08, 0.00, 0.86, 0.11, 0.06, 0.89, 1.01, 0.97, 0.15, 0.89, 0.20, 0.02, 0.90, 0.97, 0.24, 0.18, 0.07, 0.82, 0.87, 0.25, 0.03, 0.22, 0.95, 0.20, 0.90, 0.90, 0.08, 0.94, 0.20, 0.23, 0.88, 0.99, 0.07, 0.09, 0.19, 0.95, 0.97, 0.04, 0.04, 0.84, 0.91, 0.26, 0.83, 0.87, 0.12, 0.97, 0.95, 0.04, 0.09, 0.03, 0.92, 0.12, 0.85, 0.18, 0.97, 0.08, 0.16, 0.91, 0.91, 0.14, 0.84, 0.95, 0.05, 0.90, 0.22, 0.27, 0.00, 0.99, 0.25, 0.21, 0.96, 0.19, 0.95, 0.89, 0.16, 0.19, 0.92, 0.17, 0.19, 0.88, 0.04, 0.10, 0.93, 0.00, 0.27, 0.11, 0.94, 0.85, 0.14, 0.24, 0.92, 0.09, 0.24, 0.90, 0.09, 0.35, 0.11, 0.96, 0.13, 0.27, 0.76, 0.12, 0.15, 0.81, 0.26, 0.89, 0.94, 0.18, 0.96, 0.88, 0.11, 0.25, 0.90, 0.03, 0.95, 0.82, 0.27, 0.88, 0.96, 0.12, 0.88, 0.22, 0.17, 1.02, 0.08, 0.24, 0.09, 0.92, 0.07, 0.95, 0.19, 0.37, 0.94, 0.28, 0.15, 1.03, 0.96, 0.23, 0.01, 0.90, 0.14, 0.00, 0.00, 0.91, 0.88, 0.94, 0.27, 0.90, 0.86, 0.17, 0.28, 0.18, 0.86, 0.00, 0.27, 0.89, 0.91, 0.90, 0.04, 0.18, 0.11, 0.95, 0.24, 0.88, 0.18, 0.96, 0.90, 0.00, 0.03, 0.00, 0.90, 0.92, 0.15, 0.33, 0.90, 0.92, 0.29, 0.94, 0.04, 0.16, 0.18, 0.88, 0.16, 0.88, 0.90, 0.11, 0.12, 0.17, 0.86, 0.17, 0.19, 1.04, 0.87, 0.19, 0.40, 0.16, 0.90, 0.26, 0.31, 0.20, 0.95, 0.90, 0.22, 0.19, 0.18, 1.01, 0.18, 0.00, 0.93, 0.15, 0.26, 0.88, 0.26, 0.95, 0.88, 0.19, 0.29, 0.91, 0.17, 0.32, 0.23, 0.91, 0.92, 0.17, 0.07, 0.11, 0.00, 0.91, 0.26, 0.83, 0.11, 0.87, 0.02, 0.19, 0.31, 0.24, 0.90, 0.03, 0.92, 0.26, 0.96, 0.04, 0.25, 0.87, 0.01, 0.00, 0.20, 0.02, 0.93, 0.10, 0.16, 0.95, 0.91, 0.89, 0.11, 0.12, 0.27, 0.92, 0.15, 0.88, 0.20, 0.88, 0.32, 0.15, 0.07, 0.81, 0.00, 0.10, 0.17, 0.92, 0.88, 0.93, 0.19, 0.07, 0.24, 0.88, 0.24, 0.10, 0.82, 0.90, 0.14, 0.35, 0.97, 0.89, 0.26, 0.29, 0.93, 0.14, 0.89, 0.05, 0.00, 0.84, 0.30, 0.14, 0.02, 0.12, 0.92, 0.96, 0.23, 0.23, 0.23, 0.36, 0.87, 0.98, 0.49, 0.18, 0.94, 0.00, 0.08, 0.90, 0.78, 0.37 };

	for (int c = 0; c < 130; c++)
		for (int r = 0; r < 3; r++)
			testData(r, c) = testDataRaw[r + c * 3];

	auto values = m.predict(std::move(testData));

	int correctGuessCount = 0;
	for (int j = 0; j < values.getCols(); j++){
		double max = -10;
		ptrdiff_t max_index = -1;
		for (const auto& seria : std::ranges::views::enumerate(values.getCol(j))) {
			auto&& [index, value] = seria;
			if (value > max) {
				max = value;
				max_index = index;
			}
		}
		if (max_index + 1 == expectedColorsTest[j])
			correctGuessCount += 1;

	}
	auto successRate = static_cast<double>(correctGuessCount) / expectedColorsTest.size() * 100;
	std::cout << "Success rate: " << successRate <<"%\n";
}


void ch4t1()
{
	// msvc intrinsic function
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(40, -0.1, 0.1, activationFunctions::ReLU, activationFunctions::dReLU).addLayer(10, -0.1, 0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*)&magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num << '\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28 * 28 * num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	//Matrix debugImg(28, 28);
	//for (int i = 0; i < 28; i++)
	//	for (int j = 0; j < 28; j++)
	//	{
	//		debugImg(i, j) = std::ceil(input(28 * i + j, 1));
	//	}
 //   std::cout << std::setprecision(1) << debugImg<<"\n";
	//std::cout << int(labels[1]);

	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i], i) = 1;
	}
	std::cout << labels[0];

	m.learnDropout(params::epochs, input, expected, params::alpha);


	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);

	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predict(testinput);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num * 100 << "%";

}


void ch4t2() 	{
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(40, -0.1, 0.1, activationFunctions::ReLU, activationFunctions::dReLU).addLayer(10, -0.1, 0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*)&magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num << '\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28 * 28 * num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i], i) = 1;
	}
	std::cout << labels[0];

	m.learnDropoutWithBatching(params::epochs, input, expected, params::alpha,100);


	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);

	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predict(testinput);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num * 100 << "%";

}

void ch4t3(){
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(100, -0.01, 0.01, activationFunctions::tanh, activationFunctions::dTanh).addLayer(10, -0.1, 0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*)&magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num << '\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28 * 28 * num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i], i) = 1;
	}
	std::cout << labels[0];

	m.learnDropoutWithBatchingAndSoftMax(params::epochs, input, expected, params::alpha, 100);


	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);

	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predict(testinput);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num * 100 << "%";

}

void ch5t1(){
	Matrix input({
		1,1,1,0,0,
		0,1,1,1,0,
		0,0,1,1,1,
		0,0,1,1,0,
		0,1,1,0,0
		}, 5, 5);

	Matrix filter(
		{
			1,0,1,
			0,1,0,
			1,0,1
		}, 3, 3);

	Model m;
	std::cout << m.convolution(input, filter, 1, 0);
	auto m1 = m.makeImageSections(input, 3, 1);
	auto m2 = m.createKernels({ filter });
	std::cout << m1;
	std::cout << "\n\n" << m2;
	std::cout << "\n\n" << m.makeImageSections(input, 3, 1) * m.createKernels({filter}).transpose();
}


void ch5t2() {
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(10, -0.1, 0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*)&magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num << '\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28 * 28 * num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	std::vector<Matrix> filters(16,Matrix(3,3));
	for (auto& e : filters)
		e.fillRandom(-0.01, 0.01);
	
	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i], i) = 1;
	}
	std::cout << labels[0];
	m.learnWithConvolution(params::epochs, input, expected, params::alpha,filters,1);


	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);

	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predictConv(testinput,1);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num * 100 << "%";

}


void ch5t3() {
	unsigned long _byteswap_ulong(unsigned long value);

	Model m;
	m.addLayer(10, -0.1, 0.1);

	std::ifstream tdataLabels("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tdataLabels.is_open())
		return;
	int magic, num;
	tdataLabels.read((char*)&magic, 4);
	tdataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	num = params::learnSeriesNum;

	std::cout << "number of series for training: " << num << '\n';
	std::vector<char> labels(num);

	tdataLabels.read(labels.data(), num);

	std::ifstream tdata("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);
	tdata.read((char*)&magic, 4);


	std::vector<std::array<unsigned char, 28 * 28>> imagedata(num);


	for (int i = 0; i < num; ++i)
	{
		tdata.read((char*)imagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> normalizedImageData(28 * 28 * num);
	int pixelCounter = 0;

	for (auto& img : imagedata)
	{
		for (auto& pixel : img)
		{
			normalizedImageData[pixelCounter++] = pixel / 255.;
		}
	}

	Matrix input(std::move(normalizedImageData), num, 784); // again careful for column majority 
	input = input.transpose();

	std::vector<Matrix> filters(16, Matrix(3, 3));
	for (auto& e : filters)
		e.fillRandom(-0.01, 0.01);

	std::vector<double> labelsdouble;
	Matrix expected(10, num);
	labelsdouble.reserve(num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		expected(labels[i], i) = 1;
	}
	std::cout << labels[0];
	m.learnWithConvolutionAndPooling(params::epochs, input, expected, params::alpha, filters, 1);


	std::ifstream dataLabels("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!dataLabels.is_open())
		return;
	dataLabels.read((char*)&magic, 4);
	dataLabels.read((char*)&num, 4);

	magic = _byteswap_ulong(magic);
	num = _byteswap_ulong(num);

	std::vector<char> testlabels(num);

	dataLabels.read(testlabels.data(), num);

	std::ifstream data("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	// just skip ...
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);
	data.read((char*)&magic, 4);


	num = params::testSeriesNum; // test input series
	std::cout << "number of series for test: " << num << '\n';

	std::vector<std::array<unsigned char, 28 * 28>> testimagedata(num);


	for (int i = 0; i < num; ++i)
	{
		data.read((char*)testimagedata[i].data(), 28 * 28);
	}

	// normalize
	std::vector<double> testnormalizedImageData(28 * 28 * num);

	pixelCounter = 0;


	for (auto& img : testimagedata)
	{
		for (auto& pixel : img)
		{
			testnormalizedImageData[pixelCounter++] = pixel / 255.f;
		}
	}

	Matrix testinput(std::move(testnormalizedImageData), num, 784); // careful, column major needed
	testinput = testinput.transpose();

	Matrix testexpected(10, num);

#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		testexpected(testlabels[i], i) = 1;
	}
	//std::cout << testexpected;

	auto out = m.predictWithConvolutionAndPooling(testinput, 1);

	//std::cout << std::setprecision(2)<<'\n';
	out = Matrix::oneHot(out);
	//std::cout << out;
	double correct = 0;
	for (int col = 0; col < out.getCols(); col++)
	{
		if (out.getCol(col) == testexpected.getCol(col))
			correct++;
	}

	std::cout << correct / num * 100 << "%";

}

int main()
{
	auto start_time = std::chrono::high_resolution_clock::now();
	//ch3t2();
	//ch4t1();
	//ch4t2();
	//ch4t3();
	//ch5t1();
	//ch5t2();
	ch5t3();
	auto current_time = std::chrono::high_resolution_clock::now();
	std::cout << "task has been running for " << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << " seconds" << std::endl;
	start_time = std::chrono::high_resolution_clock::now();
	//ch3t4();
	current_time = std::chrono::high_resolution_clock::now();
	std::cout << "task has been running for " << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << " seconds" << std::endl;
	


	
}