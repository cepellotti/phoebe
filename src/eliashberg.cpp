#include "bandstructure.h"
#include "eliashberg.h"
#include "exceptions.h"
#include "io.h"

EliashbergFunction::EliashbergFunction(Context &context,
                                       BaseBandStructure &phBandStructure,
                                       StatisticsSweep &statisticsSweep) :
    context_(context),
    phBandStructure_(phBandStructure), statisticsSweep_(statisticsSweep),
    tetra_(phBandStructure_) {
  double minEnergy = 0.;
  double maxEnergy = 0.;
  // find min/max phonon energy
  for (int is : phBandStructure.parallelIrrStateIterator()) {
    StateIndex isIdx(is);
    double en = phBandStructure.getEnergy(isIdx);
    if (en < minEnergy) {
      minEnergy = en;
    }
    if (en > maxEnergy) {
      maxEnergy = en;
    }
  }
  mpi->allReduceMin(&minEnergy);
  mpi->allReduceMax(&maxEnergy);

  numBins_ = context_.getNumEliashbergBins();
  deltaEnergy_ = (maxEnergy - minEnergy) / numBins_;
  energies_.resize(numBins_);
  for (int i = 0; i < numBins_; i++) {
    energies_(i) = deltaEnergy_ * (0.5 + i);
  }

  data.resize(statisticsSweep.getNumCalculations(), numBins_);
  data.setZero();
}

std::vector<std::pair<double, Eigen::VectorXd>>
EliashbergFunction::parallelLoopIterator() {
  std::vector<std::pair<double, Eigen::VectorXd>> result;
  for (int i : mpi->divideWorkIter(numBins_)) {
    double x = energies_(i);
    Eigen::VectorXd y = data.col(i);
    std::pair<double, Eigen::VectorXd> z = {x, y};
    result.push_back(z);
  }
  return result;
}

void EliashbergFunction::allReduceSum() {
  mpi->allReduceSum(&data);
  for (int iCalc = 0; iCalc < statisticsSweep_.getNumCalculations(); iCalc++) {
    double chemPot =
        statisticsSweep_.getCalcStatistics(iCalc).chemicalPotential;
    data.row(iCalc) /= tetra_.getDOS(chemPot);
  }
}

void EliashbergFunction::add(const int &iCalc, const double &rate,
                             StateIndex &iPhIdx) {
  double en = phBandStructure_.getEnergy(iPhIdx);

  // find the index of the bin where en falls into
  auto refBin = int((en - energies_(0)) / deltaEnergy_);
  // make sure it falls into bounds
  refBin = std::min(refBin, numBins_);
  refBin = std::max(refBin, 0);

  double maxEn = en * 1.1; // + 10%
  double minEn = en * 0.9; // - 10%

  auto minBin = int((minEn - energies_(0)) / deltaEnergy_);
  auto maxBin = int((maxEn - energies_(0)) / deltaEnergy_);
  minBin = std::min(minBin, numBins_);
  minBin = std::max(minBin, 0);
  maxBin = std::min(maxBin, numBins_);
  maxBin = std::max(maxBin, 0);

  for (int iBin = minBin; iBin < maxBin; iBin++) {
    data(iCalc, iBin) += rate * tetra_.getSmearing(energies_(iBin), iPhIdx);
  }
}

void EliashbergFunction::getCriticalTemperature() {
  double mu = context_.getCoulombRepulsion();

  double deltaOmega = deltaEnergy_;

  int numCalculations = statisticsSweep_.getNumCalculations();

  Eigen::VectorXd lambda(numCalculations);
  lambda.setZero();
  Eigen::VectorXd logAverageOmega(numCalculations);
  logAverageOmega.setZero();

  for (auto t : parallelLoopIterator()) {
    double omega = std::get<0>(t);
    if (omega <= 0.)
      continue;
    Eigen::VectorXd thisElFuncs = std::get<1>(t);
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      lambda(iCalc) += 2. * deltaOmega / omega * thisElFuncs(iCalc);
      logAverageOmega(iCalc) +=
          2. * deltaOmega / omega * thisElFuncs(iCalc) * log(omega);
    }
  }
  mpi->allReduceSum(&logAverageOmega);
  mpi->allReduceSum(&lambda);

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    logAverageOmega.array() = exp(logAverageOmega.array() / lambda.array());
  }

  Eigen::VectorXd criticalTemperature(numCalculations);
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    double arg = -1.04 * (1. + lambda(iCalc)) /
                 (lambda(iCalc) - mu * (1. + 0.62 * lambda(iCalc)));
    criticalTemperature(iCalc) = logAverageOmega(iCalc) / 1.2 * exp(-arg);
  }
}
