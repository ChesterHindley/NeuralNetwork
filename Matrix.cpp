#include "Matrix.h"
#include <cassert>
#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <ranges>

// REMEMBER THAT SUCH INITIALIZATION WORKS ONLY FOR ROW MAJOR MATRICES
Matrix::Matrix(std::initializer_list<dataType> l, int rows_, int cols_) : data{ l }, rows{ rows_ }, cols{ cols_ }
{
	assert(rows * cols == l.size());
}

Matrix::Matrix(std::vector<dataType>&& v, int rows_, int cols_) : data{ std::move(v) }, rows{ rows_ }, cols{cols_}
{
	assert(data.size() == rows * cols);
}

Matrix::Matrix(int rows_, int cols_) : rows{ rows_ }, cols{ cols_ }, data(rows_*cols_,dataType())
{
}

Matrix::Matrix() = default;

Matrix::dataType& Matrix::operator()(int row, int col)
{
	assert(row < rows && col < cols);
	return data[row * cols + col];
}

Matrix::dataType Matrix::operator()(int row, int col) const
{
	assert(row < rows && col < cols);
	return data[row * cols + col];
}

bool Matrix::operator==(const Matrix& m) const
{
	static constexpr double epsilon = 0.000005;
	auto & otherData = m.data;
	return rows == m.rows && cols == m.cols && std::ranges::all_of(data, [i = 0, &otherData](const auto& elem) mutable {return fabs(elem - otherData[i++]) < epsilon; });
}

Matrix Matrix::transpose() const
{
	auto ret = *this;
	ret.rows = cols;
	ret.cols = rows;
	for (int i = 0; i < getRows(); i++)
		for (int j = 0; j < getCols(); j++)
			ret(j, i) = this->operator()(i,j);

	return ret;
}

int Matrix::getRows() const
{
	return rows;
}
int Matrix::getCols() const
{
	return cols;
}

std::vector<Matrix::dataType> Matrix::getCol(int N) const
{
	std::vector<dataType> ret;
	ret.reserve(rows);
	for (int i = 0; i < rows; i++)
			ret.push_back(this->operator()(i, N));
	return ret;
}

Matrix Matrix::vectorize() const
{
	std::vector<dataType> ret;
	ret.reserve(rows * cols);
	for (int i = 0; i < cols; i++)
	{
		ret.append_range(getCol(i));
	}
	return Matrix(std::move(ret),rows*cols,1);

}

Matrix Matrix::ewSquare() const
{
	Matrix ret = *this;

	for (auto& elem : ret.data)
		elem *= elem;

	return ret;
}

Matrix Matrix::hadamardProduct(Matrix m) const
{
	assert(cols == m.cols && rows == m.rows);
	 Matrix ret(*this);
	 for (int i = 0; i < cols; i++)
		 for (int j = 0; j < rows; j++)
			 ret(i, j) *= m(i, j);
	 return ret;
	 

	 
}


Matrix& Matrix::fillRandom(dataType min, dataType max)
{
	auto mt = std::mt19937(std::random_device{}());
	std::ranges::generate(data, [&] {return std::uniform_real_distribution(min, max)(mt); });
	return *this;
}

Matrix& Matrix::resize(int rows_, int cols_)
{
	rows = rows_;
	cols = cols_;
	data.resize(rows * cols);
	return *this;
}

void Matrix::serialize(std::string_view fileName) const
{
	std::ofstream f(fileName.data(),std::ios::app);
	f << rows << ' ' << cols << '\n';
	for (const auto& elem : data | std::ranges::views::take(data.size()-1))
		f << elem << ' ';
	f << data.back();
	f << '\n';

}

Matrix operator+(const Matrix& m1, const Matrix& m2)
{
	assert(m1.cols == m2.cols && m1.rows == m2.rows);
	Matrix ret = m1;

	for (int i = 0; i < m1.data.size(); ++i)
		ret.data[i] += m2.data[i];

	return ret;
}

Matrix operator-(const Matrix& m1, const Matrix& m2)
{
	auto neg = m2;
	for (auto& elem : neg.data)
		elem *= -1;
	return m1 + neg;
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	assert(m1.cols == m2.rows);
	Matrix ret(m1.rows,m2.cols);
	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m2.cols; j++)
			for (int k = 0; k < m1.cols; k++)
				ret(i, j) += m1(i, k) * m2(k, j);
	return ret;
}

Matrix operator*(Matrix::dataType scalar, const Matrix& m)
{
	auto ret = m;
	for (auto& elem : ret.data)
		elem *= scalar;
	return ret;
}

Matrix operator*( const Matrix& m, Matrix::dataType scalar)
{
	return operator*(scalar,m);
}

std::ostream& operator<<(std::ostream& out, const Matrix& m)
{
	for (int i = 0; i < m.rows; i++) {
		if (i) out << '\n';
		for (int j = 0; j < m.cols; j++)
			out << m(i, j) << ' ';
	}
	out << "\n";
	return out;
}

Matrix Matrix::deserialize(std::string_view fileName)
{
	std::ifstream in(fileName.data());
	Matrix m;
	int rows, cols;
	in >> rows >> cols;
	m.resize(rows, cols);
	for (auto& elem : m.data)
		in >> elem;
	return m;
}

Matrix Matrix::deserialize(std::ifstream& in )
{
	Matrix m;
	int rows, cols;
	in >> rows >> cols;
	assert(rows > 0 && cols > 0);

	m.resize(rows, cols);
	for (auto& elem : m.data)
		in >> elem;
	return m;
}
