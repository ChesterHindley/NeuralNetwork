#pragma once
#include "Matrix.h"
#include <vector>
class Model
{
	// first matrix is layer closest to input

	std::vector<Matrix> layers;
	// need only last hidden layer output for updating output layer
	Matrix layerOutputCache;
	double error;

public:
	Model& addLayer(int N, double min = -1.0, double max = 1.0);
	Model& updateLayer(int N,Matrix m);
	Matrix getLayer(int N) const;
	Matrix predict(Matrix input);
	void learn(int N,Matrix input, Matrix expected, double learningRate);
	void serialize(std::string_view);
	static Model deserialize(std::string_view);

	Model& addLayer(Matrix mat);
	double getError() const;
private:
};

double ReLU(double);
double dReLU(double);
Matrix ReLU(Matrix);
Matrix dReLU(Matrix);