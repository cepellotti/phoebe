#ifndef VECTOR_EPA_H
#define VECTOR_EPA_H

#include "Matrix.h"
#include "PMatrix.h"
#include "active_bandstructure.h"
#include "context.h"
#include "eigen.h"

/** Class used to store a "matrix" of data.
 * The class member "data" has size data(numCalculations,numStates), where numCalculations
 * is the number of temperatures/chemical potentials, and numStates is
 * specified in input. Used for the EPA electron transport calculation.
 * It is subclassed to a VectorBTE case when numStates is aligned with the
 * Bloch states of the band structure class.
 */
class VectorEPA {
 public:
  /** Constructor method, initializes raw buffer data and saves helper
   * variables.
   * @param statisticsSweep: saves the info on how many temperatures/chemical
   * potentials we are evaluating.
   * @param numStates: saves the number of states on which we compute the
   * vector.
   * @param dimensionality: determines the size of the vector on cartesian
   * indices. 1 for scalar quantities like line-widths Gamma(BlochIndices), 3
   * for vector quantities like phonon populations f(blochIndices,cartesian).
   */
   VectorEPA(StatisticsSweep &statisticsSweep_, const int &numStates_,
                const int &dimensionality_ = 3);

  /** Copy constructor
   */
   VectorEPA(const VectorEPA &that);

  /** Copy assignment operator
   */
   VectorEPA &operator=(const VectorEPA &that);

  /** Computes the scalar product between two BaseVectorBTE objects.
   * The scalar product of x and y, is defined such as
   * z(iCalc) = sum_i x(iCalc,i) y(iCalc,i), where i is an index over Bloch
   * states, and iCalc is an index over temperatures and chemical potentials.
   * @param that: the second vector used in the scalar product
   */
  virtual Eigen::MatrixXd dot(const VectorEPA &that);

  /** element wise product between two BaseVectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x*y.
   * If y has dimensionality 1, we compute x(every dim)*y(0), and the result
   * has the dimensionality of x.
   * @param that: the second BaseVectorBTE object y, such that result = *this*y
   */
  VectorEPA operator*(VectorEPA &that);

  /** Computes the product of a BaseVectorBTE with a scalar, i.e. all elements
   * of vectorBTE x -> x * scalar.
   * @param scalar: a double with the constant factor to be used in the
   * element-wise multiplication.
   */
  VectorEPA operator*(const double &scalar);

  /** Computes the product of a BaseVectorBTE with a vector. The vector has
   * size equal to the number of calculations (i.e. number of temperatures
   * times the number of chemical potentials) used in the run. Given a
   * calculation index iCalc, the result is an element-wise x(it)*vector(it).
   * @param vector: a double vector to be used in the product, of size
   * equal to numCalculations.
   */
  VectorEPA operator*(const Eigen::VectorXd &vector);

  /** Computes the product of a BaseVectorBTE with a parallel matrix. Only works
   * if the number of temperatures/chemical potentials (numCalculations) is equal
   * to one. At fixed calculation index iCalc, the result is an matrix-vector
   * multiplication x(it,i)*pMatrix(i,j).
   * @param pMatrix: a parallel distributed double matrix to be used in the
   * product, of size equal to numStates x numStates.
   */
  VectorEPA operator*(ParallelMatrix<double> &matrix);

  /** element wise sum between two BaseVectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x+y.
   * If y has dimensionality 1, we compute x(every dim)+y(0), and the result
   * has the dimensionality of x.
   * @param that: the second BaseVectorBTE object y, such that result = *this+y
   */
  VectorEPA operator+(VectorEPA &that);

  /** element wise difference between two BaseVectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x-y.
   * If y has dimensionality 1, we compute x(every dim)-y(0), and the result
   * has the dimensionality of x.
   * @param that: the second BaseVectorBTE object y, such that result = *this-y
   */
  VectorEPA operator-(VectorEPA &that);

  /** Invert the sign of the BaseVectorBTE content i.e. x -> -x
   */
  VectorEPA operator-();

  /** element wise division between two BaseVectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x/y.
   * If y has dimensionality 1, we compute x(every dim)/y(0), and the result
   * has the dimensionality of x.
   * @param that: the second BaseVectorBTE object y, such that result = *this/y
   */
  VectorEPA operator/(VectorEPA &that);

  /** Replace the content of BaseVectorBTE with its square root
   * (element-wise x -> sqrt(x) ).
   */
  VectorEPA sqrt();
  /** Replace the content of BaseVectorBTE with its reciprocal
   * (element-wise x -> 1/x).
   */
  VectorEPA reciprocal();

  /** Set the whole content (raw buffer) of BaseVectorBTE to a scalar value.
   * @param constant: the value to be used in the set.
   */
  void setConst(const double &constant);

  /** Get and set operator
   */
  double &operator()(const int &iCalc, const int &iDim, const int &iState);

  /** Const get and set operator
   */
  const double &operator()(const int &iCalc, const int &iDim,
                           const int &iState) const;

  /** raw buffer containing the values of the vector
   *  The matrix has size (numCalculations, numStates), where numCalculations is the number
   *  of pairs of temperature and chemical potentials, and numStates is the
   *  number of Bloch states used in the Boltzmann equation.
   */
  Eigen::MatrixXd data;

  // we store auxiliary objects and parameters
  StatisticsSweep &statisticsSweep;
  int numCalculations;
  int numStates;
  int numChemPots;
  int numTemps;
  int dimensionality;

  /** glob2Loc and loc2Glob compress/decompress the indices on temperature,
   * chemical potential, and cartesian direction into/from a single index.
   * TODO: these indices, and how they are used elsewhere, is rather messy
   * That's because we have to work both with quantities such as line-widths,
   * which are a scalar over the Bloch states, and phonon populations, which
   * are cartesian vectors over the Bloch states.
   * I should probably create two different classes for these.
   */
  int glob2Loc(const ChemPotIndex &imu, const TempIndex &it,
                const CartIndex &iDim) const;
  std::tuple<ChemPotIndex, TempIndex, CartIndex> loc2Glob(const int &i) const;

  /** List of Bloch states to be excluded from the calculation (i.e. for
   * which vectorBTE values are 0), for example, the acoustic modes at the
   * gamma point, whose zero frequencies may cause problems.
   */
  std::vector<int> excludeIndices;

 protected:
  /** base class to implement +, -, / and * operations.
   * It's split separately so that subclasses can create the correct output
   * class, and also because operations are rather similar.
   */
   VectorEPA baseOperator(VectorEPA &that, const int &operatorType);
  const int operatorSums = 0;
  const int operatorDivs = 1;
  const int operatorProd = 2;
  const int operatorDiff = 3;
};

#endif
