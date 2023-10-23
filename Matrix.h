#pragma once
#include <vector>
#include <initializer_list>
#include <iostream>
#include <string_view>
class Matrix
{
public:
	using dataType = double;

	friend Matrix operator+(const Matrix& m1, const Matrix& m2);
	friend Matrix operator-(const Matrix& m1, const Matrix& m2);
	friend Matrix operator*(const Matrix& m1, const Matrix& m2);
	friend Matrix operator*(dataType scalar, const Matrix& m);
	friend Matrix operator*(const Matrix& m, dataType scalar);

	friend std::ostream& operator<<(std::ostream& out, const Matrix& m);

	friend class Model;


public:
	std::vector<dataType> data;
	int rows;
	int cols;

public:
	Matrix(std::initializer_list<dataType> l, int rows_, int cols_);
	Matrix(std::vector<dataType>&& v, int rows_,int cols_);
	Matrix(int rows_, int cols_);
	Matrix();  // default

	dataType& operator()(int row, int col);
	dataType operator()(int row, int col) const;
	bool operator==(const Matrix&) const;
	Matrix transpose() const;
	Matrix& fillRandom(dataType min, dataType max);
	Matrix& resize(int rows_, int cols_);
	void serialize(std::string_view) const;
	static Matrix deserialize(std::string_view);
	static Matrix deserialize(std::ifstream&);
	int getRows() const;
	int getCols() const;
	std::vector<dataType> getCol(int N) const;
	Matrix vectorize() const;

	// ew - element wise
	Matrix ewSquare() const;
	Matrix hadamardProduct(Matrix) const;


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
}
