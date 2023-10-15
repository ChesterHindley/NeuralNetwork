#include "Model.h"
#include <random>
#include <ranges>
#include <fstream>
#include <numeric>
#include <algorithm>

double ReLU(double value)
{
	return std::max(0.0, value);
}

double dReLU(double value)
{
	return value > 0 ? 1. : 0.;
}



Model& Model::addLayer(int N, double min, double max)
{
	auto static constexpr numberOfInputs = 1;
	Matrix mat(N, !layers.empty() ? layers.back().getRows() :  numberOfInputs);
	mat.fillRandom(min, max);
	layers.push_back(std::move(mat));
	return *this;
}

Model& Model::updateLayer(int N,Matrix m)
{
	layers.at(N) = std::move(m);
	return *this;
}

Matrix Model::getLayer(int N) const
{
	return layers.at(N);
}

Model& Model::addLayer(Matrix mat)
{
	layers.push_back(std::move(mat));
	return *this;
}

double Model::getError() const
{
	return error;
}

Matrix Model::predict(Matrix input) 
{
	// TODO LAYER OUTPUT CACHING

	if (input.getRows() != layers[0].getCols())
	layers[0].resize(layers[0].getRows(), input.getRows())
		.fillRandom(-1.,1.);  // TODO
	
	Matrix ret = layers[0] * input;

	for (auto& layer : layers | std::ranges::views::drop(1))
		ret = layer * ret;
	return ret;
}

void Model::learn(int N, Matrix input, Matrix expected, double learningRate)
{
	
	for (int i = 0; i < N; i++) {
		auto output = predict(input);
		auto delta = 2. / output.getRows() * (output - expected) * input.transpose();
		auto updatedWeights = getLayer(0) - learningRate * delta;
		updateLayer(0, updatedWeights);
	}
		error = 0.;
		for (const auto& elem : (predict(input) - expected).data)
		{
			error += elem * elem;
		}
		error /= expected.rows;
}

void Model::serialize(std::string_view s)
{
	for (const auto& layer : layers)
		layer.serialize(s);

}

Model Model::deserialize(std::string_view s)
{
	Model ret;
	std::ifstream f(s.data());
	while (f.good())
	{
		ret.addLayer(Matrix::deserialize(f));
		f.get(); // new line
		f.peek(); // trigger eof

	}
	return ret;
}
