#include "Matrix.h"
#include <cassert>
#include <ranges>
#include "Model.h"
#include <fstream>
#include <iomanip>

auto neuron(Matrix weights, Matrix input, int bias)
{
	return (weights * input)(0,0) + bias;
}
auto neural_network(Matrix weights, Matrix input)
{
	return weights * input;
}

// first element in vector of weights should be the layer closest to input (the hidden layer)
auto deep_neural_network(Matrix input, std::vector<Matrix> weights)
{
	auto ret = weights[0] * input;
	for (auto& weight : weights | std::ranges::views::drop(1))
		ret = weight * ret;
	return ret;
}

void testNeural()
{
	Matrix m1(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3);
	Matrix m2({
		0.5,
		0.75,
		0.1
		}, 3, 1);
	assert(neural_network(m1, m2) == Matrix({ 0.095,0.2,0.535,0.4,0.235 }, 5, 1));

}
void testDeep()
{
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
			0.7, 0.9,-0.4, 0.8, 0.1,
			0.8, 0.5, 0.3, 0.1, 0.0,
			-0.3, 0.9, 0.3, 0.1, -0.2
		}, 3, 5);
	
	Matrix input({
		0.5,
		0.75,
		0.1
		}, 3, 1);
	std::vector weights{ std::move(Wh),std::move(Wy) };
	assert(deep_neural_network(input, weights) == Matrix({
		0.376,
		0.3765,
		0.305
		}, 3, 1));
}
void testModel()
{
	std::ofstream("model.log").close(); 
	std::ofstream("log.txt").close();
	Model m;
	m.addLayer(5)
		.addLayer(3);
	Matrix res = m.predict(Matrix({1,2,3,2,2,1,2},7,1));

	std::cout << res;
	res.serialize("log.txt");
	res = Matrix::deserialize("log.txt");
	std::cout << res;
	m.serialize("model.log");
	m = Model::deserialize("model.log");
	m.serialize("modelcopy.log");

}


void chapter2()
{
	Matrix m1(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3);
	Matrix input(
		{
			0.5,
			0.75,
			0.1,
		}, 3, 1);
	Matrix expected(
		{
			0.1,
			1.0,
			0.1,
			0.0,
			-0.1
		}, 5, 1);

	auto output = m1 * input;
	auto delta = 2.f / output.getRows() * (output - expected) * input.transpose();
	static constexpr double alpha = 0.01;
	m1 = m1 - alpha * delta;
	std::cout << m1;

}
void ch2task1()
{
	Model m;
	m.addLayer(1,0.5,0.5);
	Matrix expected({ 0.8 }, 1, 1);
	Matrix input({ 2 }, 1, 1);
	static constexpr double alpha = 0.1;

	for (int i = 0; i < 4; i++) {
		auto output = m.predict(input);
		auto delta = 2. / output.getRows() * (output - expected) * input.transpose();
		auto updatedWeights = m.getLayer(0) - alpha * delta;
		m.updateLayer(0, updatedWeights);
	}
	std::cout << std::setprecision(15);
	std::cout << "Expected error: 0.0000001024 got: " << m.getError();
	std::cout <<"\nExpected: 0.80032, got: " << m.predict(input);
	

	for (int i = 0; i < 19; i++) {
		auto output = m.predict(input);
		auto delta = 2. / output.getRows() * (output - expected) * input.transpose();
		auto updatedWeights = m.getLayer(0) - alpha * delta;
		m.updateLayer(0, updatedWeights);
	}
	std::cout << "Expected:  0.800000000000010, got: " << m.predict(input);
	std::cout << "Expected error: 0 got: " << m.getError() << "\n";

}
void ch2t2()
{
	//3 wejscia 5 neuronow
	Model m;

	m.addLayer(Matrix(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3));

	Matrix input(
		{
			0.5,0.1,0.2,0.8,
			0.75,0.3,0.1,0.9,
			0.1,0.7,0.6,0.2
		}, 3, 4);
	Matrix expected({
		0.1,0.5,0.1,0.7,
		1.0,0.2,0.3,0.6,
		0.1,-0.5,0.2,0.2,
		0.0,0.3,0.9,-0.1,
		-0.1,0.7,0.1,0.8
		},5, 4);

	m.learn(1000,input, std::move(expected), 0.01);
	std::cout << m.predict(input);
	std::cout << "Error: " << m.getError() << "\n";

}
void ch2t3()
{
	std::vector<double> inputRaw{ 0.91, 0.82, 0.05, 0.91, 0.09, 0.26, 0.89, 0.90, 0.18, 0.82, 0.04, 0.08, 0.19, 0.91, 0.00, 0.20, 0.17, 0.90, 0.10, 0.97, 0.00, 0.22, 0.97, 0.03, 0.20, 0.20, 0.89, 0.03, 0.97, 0.21, 0.92, 0.28, 0.26, 0.00, 0.10, 0.86, 0.12, 0.17, 0.91, 0.94, 0.15, 0.06, 0.81, 0.25, 0.09, 0.00, 0.04, 0.88, 0.83, 0.95, 0.08, 0.84, 0.93, 0.06, 0.19, 0.17, 0.88, 0.03, 0.21, 0.93, 0.42, 0.89, 0.24, 0.15, 0.25, 0.90, 0.86, 0.99, 0.19, 0.29, 0.86, 0.10, 0.89, 0.84, 0.28, 0.82, 0.08, 0.13, 0.87, 0.16, 0.17, 0.25, 0.37, 0.85, 0.90, 0.10, 0.01, 0.08, 0.03, 0.87, 0.99, 0.92, 0.00, 0.88, 0.89, 0.28, 0.91, 0.00, 0.08, 0.01, 0.24, 0.95, 0.86, 0.00, 0.18, 0.89, 0.12, 0.21, 0.90, 0.88, 0.36, 0.22, 0.25, 0.90, 0.30, 0.95, 0.16, 0.19, 0.00, 0.85, 0.09, 0.02, 0.95, 0.88, 0.82, 0.08, 0.91, 0.15, 0.04, 0.95, 0.85, 0.07, 0.34, 0.88, 0.22, 0.88, 0.89, 0.28, 0.87, 0.18, 0.22, 0.83, 0.92, 0.11, 0.86, 0.34, 0.00, 0.00, 0.25, 0.92, 0.81, 0.24, 0.20, 0.27, 0.89, 0.07, 0.98, 0.16, 0.27, 0.90, 0.28, 0.05, 0.23, 0.28, 0.90, 0.17, 0.08, 0.93, 0.88, 0.90, 0.18, 0.95, 0.93, 0.00, 0.12, 0.10, 0.82, 0.03, 0.86, 0.19, 0.93, 0.26, 0.29, 0.99, 0.91, 0.01, 0.16, 0.14, 0.89, 0.93, 0.24, 0.24, 0.92, 0.91, 0.00, 0.91, 0.22, 0.26, 0.11, 0.79, 0.12, 0.91, 0.14, 0.02, 0.37, 0.98, 0.25, 0.14, 0.86, 0.16, 0.91, 0.18, 0.30, 0.93, 0.20, 0.08, 0.99, 0.87, 0.09, 0.13, 0.90, 0.20, 0.90, 0.29, 0.03, 0.77, 0.00, 0.25, 0.15, 0.17, 0.89, 0.85, 0.18, 0.10, 0.14, 0.85, 0.17, 0.21, 0.21, 0.90, 0.11, 0.90, 0.00, 0.89, 0.84, 0.18, 0.01, 0.30, 0.85, 0.21, 0.94, 0.13, 0.14, 0.94, 0.08, 0.86, 0.89, 0.21, 0.06, 0.20, 0.95, 0.86, 0.78, 0.26, 0.19, 1.00, 0.21, 0.21, 0.09, 0.89, 0.86, 0.93, 0.22, 0.08, 0.83, 0.21, 0.26, 0.87, 0.12, 0.84, 0.13, 0.17, 0.90, 0.28, 0.16, 0.82, 0.97, 0.12, 0.11, 0.95, 0.13, 0.94, 0.94, 0.16, 0.17, 0.86, 0.11, 0.91, 0.89, 0.01, 0.94, 0.90, 0.18, 0.00, 0.83, 0.21, 0.18, 0.87, 0.27, 0.90, 0.26, 0.09, 0.14, 0.90, 0.04, 0.86, 0.20, 0.00, 0.09, 0.12, 0.87, 0.97, 0.02, 0.13, 0.91, 0.19, 0.14 };
	Matrix input(3, 109);
	for (int c = 0; c < 109; c++)
		for (int r = 0; r < 3; r++)
			input(r, c) = inputRaw[r + c * 3];

	Model m;
	m.addLayer(4,-3,3);


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

	
	static constexpr int iterations = 1000;
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
	if (successRate > 99)
		m.serialize("GOODMODEL.txt");
}

void ch3()
{
	// outer product - two vectors, y o x = xy^T
	Model m;
	m.addLayer(3).addLayer(2);
	Matrix input({ 0.5,0.2 }, 2, 1);
	double alpha = 0.01;
	Matrix expected({ 0.1,1. }, 2, 1);

	auto output =  m.predict(input);
	auto delta = 2. / output.getRows() * (output - expected);
	auto hiddenLayerDelta = m.getLayer(1).transpose() * delta;

	// for multiple series of input it might be necessary to transpose input for correct sizes of matrices
	//auto temp = input.transpose();

	const auto view = std::ranges::views::cartesian_product(hiddenLayerDelta.data, input.data);
	for (const auto& [i1,i2] : view)
	{
		std::cout << "(" << i1 << ", " << i2 << ")\n";
	}
	delta = delta.transpose() * m.getLayer(0);
	hiddenLayerDelta = hiddenLayerDelta * input;
	auto updatedWeights = m.getLayer(1) - alpha * delta;
	m.updateLayer(1, updatedWeights);
	updatedWeights = m.getLayer(0) - alpha * hiddenLayerDelta;
	m.updateLayer(0, updatedWeights);
	std::cout << m.getLayer(0) <<"\n" << m.getLayer(1);
	
}
void ch3attemp2()
{
	Model m;
	Matrix Wh(
		{
			0.1,0.1,-0.3,
			0.1,0.2,0.0,
			0.0,0.7,0.1,
			0.2,0.4,0.0,
			-0.3,0.5,0.1
		}, 5, 3);
	m.addLayer(std::move(Wh));
	Matrix Wy(
		{
			0.7,0.9,-0.4,0.8,0.1,
			0.8,0.5,0.3,0.1,0.0,
			-0.3,0.9,0.3,0.1,-0.2,
		}, 3, 5);

	m.addLayer(std::move(Wy));
	
	Matrix input(
		{
			0.5,0.1,0.2,0.8,
			0.75,0.3,0.1,0.9,
			0.1,0.7,0.6,0.2
		}, 3, 4);

	Matrix expected(
		{
			0.1,0.5,0.1,0.7,
			1.0,0.2,0.3,0.6,
			0.1,-0.5,0.2,0.2,
		}, 3, 4);

	constexpr static double alpha = 0.01;
	auto output = m.predict(input);
	auto delta = 2. / m.getLayer(1).getRows() * (output - expected);


	auto dWh = m.getLayer(0) * delta;


	dWh = dWh.hadamardProduct( dReLU(m.getLayer(0)));

}
int main()
{
	//testNeural();
	//testDeep();
	//testModel();
	//chapter2();
	//ch2task1();
	//ch2t2();
	//ch2t3();
	ch3attemp2();


	
}