#pragma once
#include "Matrix.h"
#include <vector>
#include "Layer.h"
class Model
{
	// first matrix is layer closest to input

	std::vector<Layer> layers;
	// need only last hidden layer output for updating output layer
	std::vector<Matrix> layersoutputCache;
	Matrix mask;
	std::vector<Matrix> filters;

	double error;
	int numberOfOutputNeurons;

public:
	Model& addLayer(int N, double min = -1.0, double max = 1.0, Layer::ActivationFunction fun=nullptr, Layer::ActivationFunction dfun = nullptr);
	Model& updateLayer(int N,Matrix m);
	Matrix getLayer(int N) const;
	Matrix getLastLayer() const;
	Matrix predict(Matrix input);
	Matrix predictDropout(Matrix input,int batch);
	Matrix predictConv(Matrix input, int stride);
	Matrix predictWithConvolutionAndPooling(Matrix input, int stride);
	void learn(int N,Matrix input, Matrix expected, double learningRate);
	void learnDropout(int N, Matrix input, Matrix expected, double learningRate);
	void learnDropoutWithBatching(int N, Matrix input, Matrix expected, double learningRate,int batchSize);
	void learnDropoutWithBatchingAndSoftMax(int N, Matrix input, Matrix expected, double learningRate, int batchSize);
	void learnWithConvolution(int N, Matrix input, Matrix expected, double learningRate, std::vector<Matrix> filters, int stride);
	void learnWithConvolutionAndPooling(int N, Matrix input, Matrix expected, double learningRate, std::vector<Matrix> filters_input, int stride);
	void serialize(std::string_view);

	Matrix makeImageSections(Matrix image, int filterSize, int stride);

	Matrix createKernels(std::vector<Matrix> filters);



	Matrix convolution(Matrix input, Matrix filter, int stride, int padding);
	//static Model deserialize(std::string_view);

	Model& addLayer(Layer mat);
	double getError() const;
	int getNumberOfOutputNeurons() const;
	Matrix generateMask(int batch);
private:
};

namespace activationFunctions {
	double ReLU(double);
	double dReLU(double);
	Matrix ReLU(Matrix);
	Matrix dReLU(Matrix);

	double sigmoid(double);
	double dSigmoid(double);
	Matrix sigmoid(Matrix);
	Matrix dSigmoid(Matrix);

	double tanh(double);
	double dTanh(double);
	Matrix tanh(Matrix);
	Matrix dTanh(Matrix);
}