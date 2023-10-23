#pragma once
#include "Matrix.h"

class Layer
{
	using ActivationFunction = Matrix(*)(Matrix);
	friend class Model;

	Matrix m;
	ActivationFunction f;

public:
	Layer(Matrix, ActivationFunction);
	Matrix calculate(Matrix input);
	void update(Matrix m);

};

