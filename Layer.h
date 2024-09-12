#pragma once
#include "Matrix.h"

class Layer
{
	using ActivationFunction = Matrix(*)(Matrix);
	friend class Model;

	Matrix m;
	ActivationFunction f;
	ActivationFunction df;

public:
	Layer(Matrix, ActivationFunction function, ActivationFunction derivative);
	Matrix calculate(Matrix input);
	void update(Matrix m);


};

