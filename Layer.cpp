#include "Layer.h"

Layer::Layer(Matrix m_, ActivationFunction fun_) : m{ m }, f{fun_}
{

}

Matrix Layer::calculate(Matrix input)
{
	return f(m);
}

void Layer::update(Matrix m_)
{
	m = m_;
}
