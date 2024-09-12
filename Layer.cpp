#include "Layer.h"

Layer::Layer(Matrix m_, ActivationFunction fun_, ActivationFunction dfun_) : m{ m_ }, f{fun_}, df{dfun_}
{

}

Matrix Layer::calculate(Matrix input)
{
	return f ? f(m*input) : m*input;
}

void Layer::update(Matrix m_)
{
	m = m_;
}
