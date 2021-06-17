#ifndef ELIASHBERG_H
#define ELIASHBERG_H

#include "delta_function.h"
#include "context.h"
#include "statistics_sweep.h"

class EliashbergFunction {
public:
  EliashbergFunction(Context &context, BaseBandStructure &phBandStructure,
                     StatisticsSweep &statisticsSweep);
  std::vector<std::pair<double, Eigen::VectorXd>> parallelLoopIterator();
  void allReduceSum();
  void add(const int &iCalc, const double &rate, StateIndex &iPhIdx);
  void getCriticalTemperature();

private:
  Context &context_;
  BaseBandStructure &phBandStructure_;
  StatisticsSweep &statisticsSweep_;
  TetrahedronDeltaFunction tetra_;
  Eigen::VectorXd energies_;
  int numBins_;
  double deltaEnergy_;
  Eigen::MatrixXd data;
};

#endif
