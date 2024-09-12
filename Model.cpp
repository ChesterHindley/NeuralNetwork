#include "Model.h"
#include <random>
#include <ranges>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <iomanip>

namespace activationFunctions {
	double ReLU(double value)
	{
		return std::max(0.0, value);
	}

	double dReLU(double value)
	{
		return value > 0 ? 1. : 0.;
	}
	Matrix ReLU(Matrix m)
	{
		for (auto& elem : m.data)
			elem = ReLU(elem);
		return m;
	}
	Matrix dReLU(Matrix m)
	{
		for (auto& elem : m.data)
			elem = dReLU(elem);
		return m;
	}
	double sigmoid(double x)
	{
		return 1. / (1 + exp(-x));
	}
	double dSigmoid(double x)
	{
		return x*(1-x);
	}
	Matrix sigmoid(Matrix m)
	{
		for (auto& elem : m.data)
			elem = sigmoid(elem);
		return m;
	}
	Matrix dSigmoid(Matrix m)
	{
		for (auto& elem : m.data)
			elem = dSigmoid(elem);
		return m;
	}

	double tanh(double x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	Matrix tanh(Matrix m) {
		for (auto& elem : m.data)
			elem = tanh(elem);
		return m;
	}

	double dTanh(double x) {
		return 	1 - x * x;
	}

	Matrix dTanh(Matrix m) {
		for (auto& elem : m.data)
			elem = dTanh(elem);
		return m;
	}

	Matrix softMax(Matrix m) {
		Matrix ret(m.rows, m.cols);
		std::vector<double> col_sum(m.cols, 0.0);  // To store sum of exponentials for each column

		// First, calculate the exponentials and column-wise sums
		for (int j = 0; j < m.cols; j++)
		{
			double sum = 0.0;
			for (int i = 0; i < m.rows; i++)
			{
				double exp_val = exp(m(i, j));  // Calculate exponential of each element
				ret(i, j) = exp_val;            // Store it in the result matrix
				sum += exp_val;                 // Update the sum for the current column
			}
			col_sum[j] = sum;  // Store the sum of exponentials for this column
		}

		// Normalize each element by the corresponding column sum
		for (int j = 0; j < m.cols; j++)
		{
			for (int i = 0; i < m.rows; i++)
			{
				ret(i, j) /= col_sum[j];  // Divide by the sum of exponentials for the column
			}
		}

		return ret;  // Return the matrix containing softmax probabilities
	}

}



Model& Model::addLayer(int N, double min, double max,Layer::ActivationFunction fun, Layer::ActivationFunction dfun)
{

	numberOfOutputNeurons = N;
	Matrix mat(N, !layers.empty() ? layers.back().m.getRows() :  1);
	mat.fillRandom(min, max);
	layers.emplace_back(std::move(mat),fun,dfun);
	return *this;
}

Model& Model::updateLayer(int N,Matrix m)
{
	layers.at(N).update(std::move(m));
	return *this;
}

Matrix Model::getLayer(int N) const
{
	return layers.at(N).m;
}

Matrix Model::getLastLayer() const
{
	return layers.back().m;
}

Model& Model::addLayer(Layer l)
{
	numberOfOutputNeurons = l.m.getRows();
	layers.push_back(std::move(l));
	return *this;
}

double Model::getError() const
{
	return error;
}

int Model::getNumberOfOutputNeurons() const
{
	return numberOfOutputNeurons;
}

Matrix Model::predict(Matrix input) 
{
	// TODO LAYER OUTPUT CACHING

	if (input.getRows() != layers[0].m.getCols())
	layers[0].m.resize(layers[0].m.getRows(), input.getRows())
		.fillRandom(-1.,1.);
	
	Matrix ret = layers[0].calculate(input);
		
	for (auto& layer : layers | std::ranges::views::drop(1))
		ret = layer.calculate(ret);
	return ret;
}

Matrix Model::predictDropout(Matrix input,int batch) {
	if (input.getRows() != layers[0].m.getCols())
	{
		layers[0].m.resize(layers[0].m.getRows(), input.getRows()).fillRandom(-1.0, 1.0);
	}

	// Apply dropout to the activations of the first hidden layer, not the weights
	Matrix ret = layers[0].calculate(input);

	// Generate the dropout mask and apply it to the activations
	mask = generateMask(batch);  // Create mask with the same shape as activations
	ret = ret.hadamardProduct(mask) * 2;  // Scale the activations after dropout

	// Pass through the rest of the layers
	for (auto& layer : layers | std::ranges::views::drop(1))
	{
		ret = layer.calculate(ret);
	}

	return ret;
}

Matrix Model::predictConv(Matrix input, int stride) {
	// Assuming `input` is a matrix where each column represents one image.
	// Initialize an output matrix with the same number of columns as input.
	Matrix predictions(10, input.getCols()); // Adjusted to 10 rows and input.getCols() columns

	// Iterate over each image in the input
	for (int i = 0; i < input.getCols(); i++)
	{
		// Step 1: Extract the current input image (one series)
		Matrix input_img = Matrix(input.getCol(i), 28, 28);

		// Step 2: Convolution
		Matrix imgSections = makeImageSections(input_img, filters[0].getRows(), stride);
		Matrix kernels = createKernels(filters);
		Matrix conv = imgSections * kernels.transpose();

		// Step 3: Apply ReLU activation
		Matrix activatedConv = activationFunctions::ReLU(conv);

		// Step 4: Flatten the activated convolution output
		Matrix flattenedConv = activatedConv.flattenRowMajor();

		// Step 5: Forward pass through the fully connected layers
		Matrix output = predict(flattenedConv);

		// Store the prediction
		// Ensure correct row-major access: output (10x1), predictions (10xinput.getCols())
		for (int row = 0; row < output.getRows(); row++)
		{
			predictions(row, i) = output(row, 0); // Assigning correctly to the predictions matrix
		}
	}
	return predictions;
}

Matrix Model::predictWithConvolutionAndPooling(Matrix input, int stride) {
	// Assuming input is a matrix where each column represents one image.
	// Initialize an output matrix with the same number of columns as input.
	Matrix predictions(10, input.getCols()); // Adjust the row size to match your output layer size.

	// Iterate over each image in the input
	for (int i = 0; i < input.getCols(); i++)
	{
		// Step 1: Extract the current input image (one column of the input matrix)
		Matrix input_img = Matrix(input.getCol(i), 28, 28);

		// Step 2: Convolution
		Matrix imgSections = makeImageSections(input_img, filters[0].getRows(), stride);
		Matrix kernels = createKernels(filters);
		Matrix conv = imgSections * kernels.transpose();  // Perform convolution

		// Step 3: Apply ReLU activation
		Matrix activatedConv = activationFunctions::ReLU(conv);

		// Step 4: Pooling
		Matrix pooledConv(13 * 13, activatedConv.getCols()); // Pooling to reduce size from 26x26 to 13x13
		for (int filter_index = 0; filter_index < activatedConv.getCols(); filter_index++)
		{
			auto img = Matrix(activatedConv.getCol(filter_index), 26, 26); // 26x26 img, each for one filter
			auto imgSectionsForPooling = makeImageSections(img, 2, 2); // Pooling with 2x2 sections

			for (int row_index = 0; row_index < imgSectionsForPooling.getRows(); row_index++)
			{
				auto tempVector = imgSectionsForPooling.getRow(row_index); // Get the 2x2 pooling window
				auto max = std::ranges::max_element(tempVector); // Find the max element in the 2x2 region

				int pooledRow = row_index / 13; // 13 because 26 -> 13 after pooling
				int pooledCol = row_index % 13;

				// Store the max value directly into pooledConv
				pooledConv(pooledRow * 13 + pooledCol, filter_index) = *max;
			}
		}

		// Step 5: Flatten the pooled output
		Matrix flattenedConv = pooledConv.flattenRowMajor();

		// Step 6: Forward pass through the fully connected layers
		Matrix output = predict(flattenedConv);

		// Store the prediction
		for (int row = 0; row < output.getRows(); row++)
		{
			predictions(row, i) = output(row, 0); // Assigning correctly to the predictions matrix
		}
	}
	return predictions;
}



Matrix Model::generateMask(int batch) {
	// Get the dimensions of the hidden layer
	int rows = getLayer(0).rows;  // Number of neurons in the hidden layer
	int cols = batch;  // Number of columns (can be 1, depending on your setup)

	// Initialize the mask matrix with the same dimensions as the hidden layer
	Matrix mask(rows, cols);

	// Random engine for generating random values
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	// 50% dropout, so keep 1 with probability 0.5 and 0 otherwise
	for (int i = 0; i < rows; i++)
	{
		int res = (dis(gen) < 0.5) ? 0 : 1;
		for (int j = 0; j < cols; j++)
		{
			mask(i, j) = res;
		}
	}

	return mask;
}
void Model::learn(int N, Matrix input, Matrix expected, double learningRate)
{
	for (int j = 0; j < N; j++) // epoch
	{
		std::cout << "Epoch " << j << "/"<< N << "\n";
		Matrix output;
		Matrix accuracy;
		double correct = 0;
		for (int i = 0; i < input.getCols(); i++) // series of input
		{
			auto input_col = Matrix(input.getCol(i), input.getRows(), 1);
			output = predict(input_col);


				accuracy = output;
			

			auto layer_output_delta = 2 * 1. / numberOfOutputNeurons * (output - Matrix(expected.getCol(i), output.getRows(), output.getCols()));
			auto layer_output_weight_delta = layer_output_delta * layers[0].f(getLayer(0) * input_col).transpose();
			auto layer_hidden_delta = getLastLayer().transpose() * layer_output_delta;
			layer_hidden_delta = layer_hidden_delta.hadamardProduct(layers[0].df(getLayer(0) * input_col));
			auto layer_hidden_weight_delta = layer_hidden_delta * input_col.transpose();
			layers[0].m = layers[0].m - learningRate * layer_hidden_weight_delta;
			layers[1].m = layers[1].m - learningRate * layer_output_weight_delta;

			accuracy = Matrix::oneHot(accuracy);

			if (accuracy.getCol(0) == expected.getCol(i))
				correct++;

		}
			std::cout << correct / expected.getCols() * 100 << "%\n";
	}
}

void Model::learnDropout(int N, Matrix input, Matrix expected, double learningRate) {
	for (int epoch = 0; epoch < N; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << "/" << N << "\n";
		double correct = 0;

		//std::cout <<std::fixed <<std::setprecision(1) <<std::setw(1) << Matrix(input.getCol(0), 28, 28) * 10;

		for (int i = 0; i < input.getCols(); i++)
		{
			auto input_col = Matrix(input.getCol(i), input.getRows(), 1);
			auto output = predictDropout(input_col,1);

			Matrix hidden_output = layers[0].calculate(input_col).hadamardProduct(mask)*2;
			// Debug: Print input and output for each sample
			//std::cout << "Sample " << i + 1 << " Input: \n" << input_col << "\n";
			//std::cout << "Predicted Output: \n" << output << "\n";

			// Backpropagation
			auto output_error = output - Matrix(expected.getCol(i), output.getRows(), output.getCols());
			auto output_delta = output_error * (2.0 / numberOfOutputNeurons);
			auto output_weight_delta = output_delta * hidden_output.transpose();

			auto hidden_delta = layers.back().m.transpose() * output_delta;
			hidden_delta = hidden_delta.hadamardProduct(layers[0].df(hidden_output));
			auto hidden_weight_delta = hidden_delta * input_col.transpose();


			layers[0].m -= learningRate * hidden_weight_delta;
			layers[1].m -= learningRate * output_weight_delta;

			// Debug: Print output error and weight deltas
			//std::cout << "Output Error: \n" << output_error << "\n";
			//std::cout << "Output Delta: \n" << output_delta << "\n";
			//std::cout << "Hidden Weight Delta: \n" << hidden_weight_delta << "\n";

			// Accuracy Calculation
			output = Matrix::oneHot(output);
			if (output.getCol(0) == expected.getCol(i))
			{
				correct++;
				//std::cout << "correct";
			}

		}

		double accuracy = (correct / input.getCols()) * 100.0;
		std::cout << "Accuracy: " << accuracy << "%\n";
		//std::cout << layers[0].m << '\n';
		//std::cout << layers[1].m;
	}
}

void Model::learnDropoutWithBatching(int N, Matrix input, Matrix expected, double learningRate,int batchSize) {
	int numBatches = input.getCols() / batchSize;

	for (int epoch = 0; epoch < N; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << "/" << N << "\n";
		double correct = 0;
		double totalSeries = 0;

		for (int batch = 0; batch < numBatches; ++batch)
		{

			Matrix input_batch(input.getRows(), batchSize);
			Matrix expected_batch(expected.getRows(), batchSize);

			for (int i = 0; i < batchSize; i++)
			{
				int data_index = batch * batchSize + i;
				for (int j = 0; j < input.getRows(); j++)
				{
					input_batch(j, i) = input(j, data_index);
				}

				for (int k = 0; k < expected.getRows(); k++)
				{
					expected_batch(k, i) = expected(k, data_index);
				}

			}


			auto output = predictDropout(input_batch,batchSize);

			Matrix hidden_output = layers[0].calculate(input_batch).hadamardProduct(mask) * 2;

			// Backpropagation
			auto output_error = output - expected_batch;
			auto output_delta = output_error * (2.0 / (numberOfOutputNeurons * batchSize));	
			auto output_weight_delta = output_delta * hidden_output.transpose();

			auto hidden_delta = layers.back().m.transpose() * output_delta;
			hidden_delta = hidden_delta.hadamardProduct(layers[0].df(hidden_output));
			auto hidden_weight_delta = hidden_delta * input_batch.transpose();


			layers[0].m -= learningRate * hidden_weight_delta;
			layers[1].m -= learningRate * output_weight_delta;


			// Accuracy Calculation
			output = Matrix::oneHot(output);
			for (int i = 0; i < batchSize; i++)
			{
				if (output.getCol(i) == expected_batch.getCol(i))
				{
					correct++;
				}
			}
			
			totalSeries += batchSize;
		}

		double accuracy = (correct / totalSeries) * 100.0;
		std::cout << "Accuracy: " << accuracy << "%\n";

	}
}


void Model::learnDropoutWithBatchingAndSoftMax(int N, Matrix input, Matrix expected, double learningRate, int batchSize) {
	int numBatches = input.getCols() / batchSize;

	for (int epoch = 0; epoch < N; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << "/" << N << "\n";
		double correct = 0;
		double totalSeries = 0;

		for (int batch = 0; batch < numBatches; ++batch)
		{

			Matrix input_batch(input.getRows(), batchSize);
			Matrix expected_batch(expected.getRows(), batchSize);

			for (int i = 0; i < batchSize; i++)
			{
				int data_index = batch * batchSize + i;
				for (int j = 0; j < input.getRows(); j++)
				{
					input_batch(j, i) = input(j, data_index);
				}

				for (int k = 0; k < expected.getRows(); k++)
				{
					expected_batch(k, i) = expected(k, data_index);
				}

			}

			auto output = predictDropout(input_batch, batchSize);
			
			output = activationFunctions::softMax(output);


			
			Matrix hidden_output = layers[0].calculate(input_batch).hadamardProduct(mask) * 2;

			// Backpropagation
			auto output_error = output - expected_batch;


			auto output_delta = output_error * (2.0 / (numberOfOutputNeurons * batchSize));
			auto output_weight_delta = output_delta * hidden_output.transpose();

			auto hidden_delta = layers.back().m.transpose() * output_delta;
			hidden_delta = hidden_delta.hadamardProduct(layers[0].df(hidden_output));
			auto hidden_weight_delta = hidden_delta * input_batch.transpose();


			layers[0].m -= learningRate * hidden_weight_delta;
			layers[1].m -= learningRate * output_weight_delta;


			// Accuracy Calculation
			output = Matrix::oneHot(output);

			for (int i = 0; i < batchSize; i++)
			{
				if (output.getCol(i) == expected_batch.getCol(i))
				{
					correct++;
				}
			}

			totalSeries += batchSize;
		}

		double accuracy = (correct / totalSeries) * 100.0;
		std::cout << "Accuracy: " << accuracy << "%\n";

	}
}

void Model::learnWithConvolution(int N, Matrix input, Matrix expected, double learningRate, std::vector<Matrix> filters_input, int stride) {
	filters = std::move(filters_input);
	for (int epoch = 0; epoch < N; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << "/" << N << "\n";
		double correct = 0;

		for (int i = 0; i < input.getCols(); i++)
		{
			// Step 1: Extract the current input image (one series)
			Matrix input_img = Matrix(input.getCol(i), 28, 28);

			// Step 2: Convolution
			Matrix imgSections = makeImageSections(input_img, filters[0].getRows(), stride);
			Matrix kernels = createKernels(filters);
			Matrix conv = imgSections * kernels.transpose();  // 16 imgs 26x26px

			// Step 3: Apply ReLU activation
			Matrix activatedConv = activationFunctions::ReLU(conv);

			// Step 4: Flatten the activated convolution output
			Matrix flattenedConv = activatedConv.flattenRowMajor();

			// Step 5: Forward pass through the fully connected layers
			Matrix output = predict(flattenedConv);

			// Step 6: Calculate loss (output error)
			Matrix output_delta = (output - Matrix(expected.getCol(i), output.getRows(), 1)) * 0.2;

			// Weight update for fully connected layers
			Matrix kernel_layer_delta = layers[0].m.transpose() * output_delta;
			kernel_layer_delta = kernel_layer_delta.hadamardProduct(flattenedConv);

			auto kernel_layer_delta_reshaped = Matrix(std::vector<double>(kernel_layer_delta.data), conv.rows, conv.cols);
			Matrix layer_output_weight_delta = output_delta * flattenedConv.transpose();

			auto kernel_layer_weight_delta = kernel_layer_delta_reshaped.transpose() * imgSections;

			// Step 9: Update convolutional filters
			for (int k = 0; k < filters.size(); k++)
			{
				filters[k] -= learningRate * Matrix(kernel_layer_weight_delta.getRow(k),filters[k].getRows(), filters[k].getCols());
			}
			layers[0].m -= layer_output_weight_delta * learningRate;
		

			// Convert output to one-hot
			Matrix output_one_hot = Matrix::oneHot(output);

			if (output_one_hot.getCol(0) == Matrix(expected.getCol(i), output.getRows(), output.getCols()).getCol(0))
			{
				correct++;
			}
		}
		double accuracy = (correct / static_cast<double>(input.getCols())) * 100.0;
		std::cout << "Epoch " << epoch + 1 << " Accuracy: " << accuracy << "%\n";


	}
}
void Model::learnWithConvolutionAndPooling(int N, Matrix input, Matrix expected, double learningRate, std::vector<Matrix> filters_input, int stride) {
	filters = std::move(filters_input);
	for (int epoch = 0; epoch < N; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << "/" << N << "\n";
		double correct = 0;

		for (int i = 0; i < input.getCols(); i++)
		{
			// Step 1: Extract the current input image (one series)
			Matrix input_img = Matrix(input.getCol(i), 28, 28);

			// Step 2: Convolution
			Matrix imgSections = makeImageSections(input_img, filters[0].getRows(), stride);
			Matrix kernels = createKernels(filters);
			Matrix conv = imgSections * kernels.transpose();  // 16 imgs 26x26px

			// Step 3: Apply ReLU activation
			Matrix activatedConv = activationFunctions::ReLU(conv);

			// Step 4: Pooling with indices
			std::vector<std::vector<std::pair<int, int>>> maxIndicesPerFilter(activatedConv.getCols());
			Matrix pooledConv(13 * 13, activatedConv.getCols()); // Each column corresponds to one filter, and each filter's pooled output is 13 * 13

			for (int filter_index = 0; filter_index < activatedConv.getCols(); filter_index++)
			{
				auto img = Matrix(activatedConv.getCol(filter_index), 26, 26); // 26x26 img, each for one filter
				auto imgSectionsForPooling = makeImageSections(img, 2, 2);

				for (int row_index = 0; row_index < imgSectionsForPooling.getRows(); row_index++)
				{
					auto tempVector = imgSectionsForPooling.getRow(row_index); // Get the 2x2 pooling window
					auto max = std::ranges::max_element(tempVector); // Find the max element in the 2x2 region
					int idx = std::distance(tempVector.begin(), max); // Get the index of the max element

					int pooledRow = row_index / 13; // 13 because 26 -> 13 after pooling
					int pooledCol = row_index % 13;

					// Store the max value directly into pooledConv
					pooledConv(pooledRow * 13 + pooledCol, filter_index) = *max;

					int originalRow = pooledRow * 2 + idx / 2;
					int originalCol = pooledCol * 2 + idx % 2;

					maxIndicesPerFilter[filter_index].emplace_back(originalRow, originalCol); // Store max indices
				}
			}

			// Step 5: Flatten the pooled output
			Matrix flattenedConv = pooledConv.flattenRowMajor();

			// Step 6: Forward pass through the fully connected layers
			Matrix output = predict(flattenedConv);

			// Step 7: Calculate loss (output error)
			Matrix output_delta = (output - Matrix(expected.getCol(i), output.getRows(), 1)) * 0.2;

			// Step 8: Backpropagation through the fully connected layers

			// Reshape output_delta to match the fully connected layer output dimensions
			Matrix fc_delta = layers[0].m.transpose() * output_delta;

			// Reshape `fc_delta` back to the size of the pooled output
			Matrix kernel_layer_delta_reshaped = Matrix(std::vector<double>(fc_delta.data), pooledConv.getRows(), pooledConv.getCols());

			// Initialize gradient matrix for before pooling (26x26 per filter)
			Matrix pooling_grad(26 * 26,activatedConv.getCols());

			// Step 9: Populate pooling_grad with indices data
			for (int filter_index = 0; filter_index < activatedConv.getCols(); filter_index++) // for each filter
			{
				const auto& maxIndices = maxIndicesPerFilter[filter_index];
				for (int pooledIdx = 0; pooledIdx < maxIndices.size(); pooledIdx++) // for each pooled img section
				{
					int originalRow = maxIndices[pooledIdx].first;
					int originalCol = maxIndices[pooledIdx].second;

					// Propagate the delta back to the position of the max element
					pooling_grad(originalRow * 26 + originalCol, filter_index) +=
						kernel_layer_delta_reshaped(pooledIdx,filter_index);
				}
			}

			// Step 10: Apply ReLU derivative to the pooling_grad
			Matrix relu_derivative = activationFunctions::dReLU(conv);
			pooling_grad = pooling_grad.hadamardProduct(relu_derivative);


			// Continue with the convolutional backpropagation using pooling_grad instead of conv gradients
			Matrix kernel_layer_weight_delta = imgSections.transpose() * pooling_grad;

			// use derivative of relu somehow

			// Update convolutional filters
			for (int k = 0; k < filters.size(); k++)
			{
				filters[k] -= learningRate * Matrix(kernel_layer_weight_delta.getCol(k), filters[k].getRows(), filters[k].getCols());
			}
			layers[0].m -= learningRate * output_delta * flattenedConv.transpose();

			// Convert output to one-hot
			Matrix output_one_hot = Matrix::oneHot(output);

			if (output_one_hot.getCol(0) == Matrix(expected.getCol(i), output.getRows(), output.getCols()).getCol(0))
			{
				correct++;
			}

		}

		double accuracy = (correct / static_cast<double>(input.getCols())) * 100.0;
		std::cout << "Epoch " << epoch + 1 << " Accuracy: " << accuracy << "%\n";
	}
}

void Model::serialize(std::string_view s)
{
	for (const auto& layer : layers)
		layer.m.serialize(s);

}

Matrix Model::makeImageSections(Matrix image, int filterSize, int stride) {
	int input_rows = image.getRows();
	int input_cols = image.getCols();
	int output_rows = (input_rows - filterSize) / stride + 1;
	int output_cols = (input_cols - filterSize) / stride + 1;

	Matrix sections(output_rows * output_cols, filterSize * filterSize);

	int section_idx = 0;

	// Slide the filter over the image
	for (int i = 0; i <= input_rows - filterSize; i += stride)
	{
		for (int j = 0; j <= input_cols - filterSize; j += stride)
		{
			// Create a row for each section of the image
			int col_idx = 0;
			for (int l = 0; l < filterSize; l++)  // Traverse columns first
			{
				for (int k = 0; k < filterSize; k++)  // Then traverse rows
				{
					sections(section_idx, col_idx++) = image(i + k, j + l);
				}
			}
			section_idx++;
		}
	}

	return sections;
}

Matrix Model::createKernels(std::vector<Matrix> filters) {
	// Assume all filters have the same dimensions
	int filter_size = filters[0].getRows() * filters[0].getCols();
	int num_filters = filters.size();

	// Initialize the kernel matrix with dimensions num_filters x filter_size
	Matrix kernels(num_filters, filter_size);

	// Fill the kernel matrix by flattening each filter into a row
	for (int i = 0; i < num_filters; ++i)
	{
		int idx = 0;
		for (int r = 0; r < filters[i].getRows(); ++r)
		{
			for (int c = 0; c < filters[i].getCols(); ++c)
			{
				kernels(i, idx++) = filters[i](r, c);
			}
		}
	}

	return kernels;
}




Matrix Model::convolution(Matrix input, Matrix filter, int stride, int padding) {
	// Get input and filter dimensions
	int inputRows = input.getRows();
	int inputCols = input.getCols();
	int filterRows = filter.getRows();
	int filterCols = filter.getCols();

	// Calculate the dimensions of the output matrix
	int outputRows = ((inputRows - filterRows + 2 * padding) / stride) + 1;
	int outputCols = ((inputCols - filterCols + 2 * padding) / stride) + 1;

	// Initialize the output matrix
	Matrix output(outputRows, outputCols);

	// Apply padding to the input matrix (if padding > 0)
	Matrix paddedInput = input;
	if (padding > 0)
	{
		int paddedRows = inputRows + 2 * padding;
		int paddedCols = inputCols + 2 * padding;
		paddedInput = Matrix(paddedRows, paddedCols);
		// Fill the paddedInput with zeroes or some other padding value
		for (int i = 0; i < inputRows; ++i)
		{
			for (int j = 0; j < inputCols; ++j)
			{
				paddedInput(i + padding, j + padding) = input(i, j);
			}
		}
	}

	// Perform convolution
	for (int i = 0; i < outputRows; ++i)
	{
		for (int j = 0; j < outputCols; ++j)
		{
			double sum = 0.0;

			// Apply the filter to the current window of the input matrix
			for (int fi = 0; fi < filterRows; ++fi)
			{
				for (int fj = 0; fj < filterCols; ++fj)
				{
					int rowIndex = i * stride + fi;
					int colIndex = j * stride + fj;
					sum += paddedInput(rowIndex, colIndex) * filter(fi, fj);
				}
			}

			// Assign the computed sum to the output matrix
			output(i, j) = sum;
		}
	}

	return output;
}

//Model Model::deserialize(std::string_view s)
//{
//	Model ret;
//	std::ifstream f(s.data());
//	while (f.good())
//	{
//		ret.addLayer(Matrix::deserialize(f));
//		f.get(); // new line
//		f.peek(); // trigger eof
//
//	}
//	return ret;
//}
