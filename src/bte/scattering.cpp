#include "scattering.h"
#include "constants.h"
#include "mpiHelper.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric> // std::iota
#include <set>
#include <utility>

ScatteringMatrix::ScatteringMatrix(Context &context_,
                                   StatisticsSweep &statisticsSweep_,
                                   BaseBandStructure &innerBandStructure_,
                                   BaseBandStructure &outerBandStructure_)
    : context(context_), statisticsSweep(statisticsSweep_),
      innerBandStructure(innerBandStructure_),
      outerBandStructure(outerBandStructure_),
      internalDiagonal(statisticsSweep, outerBandStructure, 1) {
  numStates = int(outerBandStructure.irrStateIterator().size());
  numPoints = int(outerBandStructure.irrPointsIterator().size());
  numCalculations = statisticsSweep.getNumCalculations();

  dimensionality_ = int(context.getDimensionality());

  highMemory = context.getScatteringMatrixInMemory();

  // we want to know the state index of acoustic modes at gamma,
  // so that we can set their populations to zero
  if (outerBandStructure.getParticle().isPhonon()) {
    for (int iBte = 0; iBte < numStates; iBte++) {
      auto iBteIdx = BteIndex(iBte);
      StateIndex isIdx = outerBandStructure.bteToState(iBteIdx);
      double en = outerBandStructure.getEnergy(isIdx);
      if (en < 0.1 / ryToCmm1) { // cutoff at 0.1 cm^-1
        excludeIndices.push_back(iBte);
      }
    }
  }

  if (context.getConstantRelaxationTime() > 0.) {
    constantRTA = true;
    return;
  }

  smearing = DeltaFunction::smearingFactory(context, innerBandStructure);
  if ( // innerBandStructure != outerBandStructure &&
      smearing->getType() == DeltaFunction::tetrahedron) {
    Error("Tetrahedron smearing for transport untested and thus blocked");
    // not for linewidths. Although this should be double-checked
  }
}

ScatteringMatrix::~ScatteringMatrix() {
  if (context.getConstantRelaxationTime() > 0.) {
    return; // smearing is not assigned in CRT case
  } else {
    delete smearing;
  }
}

// copy constructor
ScatteringMatrix::ScatteringMatrix(const ScatteringMatrix &that)
    : context(that.context), statisticsSweep(that.statisticsSweep),
      smearing(that.smearing), innerBandStructure(that.innerBandStructure),
      outerBandStructure(that.outerBandStructure),
      constantRTA(that.constantRTA), highMemory(that.highMemory),
      internalDiagonal(that.internalDiagonal), theMatrix(that.theMatrix),
      numStates(that.numStates), numPoints(that.numPoints),
      numCalculations(that.numCalculations), dimensionality_(that.dimensionality_),
      excludeIndices(that.excludeIndices) {}

// assignment operator
ScatteringMatrix &ScatteringMatrix::operator=(const ScatteringMatrix &that) {
  if (this != &that) {
    context = that.context;
    statisticsSweep = that.statisticsSweep;
    smearing = that.smearing;
    innerBandStructure = that.innerBandStructure;
    outerBandStructure = that.outerBandStructure;
    constantRTA = that.constantRTA;
    highMemory = that.highMemory;
    internalDiagonal = that.internalDiagonal;
    theMatrix = that.theMatrix;
    numStates = that.numStates;
    numPoints = that.numPoints;
    numCalculations = that.numCalculations;
    excludeIndices = that.excludeIndices;
    dimensionality_ = that.dimensionality_;
  }
  return *this;
}

void ScatteringMatrix::setup() {
  // note: here we want to build the matrix or its diagonal
  // builder is a pure virtual function, which is implemented in subclasses
  // c++ discourages calls to pure virtual functions in the constructor

  if (constantRTA)
    return; // nothing to construct

  std::vector<VectorBTE> emptyVector;

  // user info about memory
  memoryUsage();

  if (highMemory) {
    if (numCalculations > 1) {
      // note: one could write code around this
      // but the methods are very memory intensive for production runs
      Error("High memory BTE methods can only work with one "
            "temperature and/or chemical potential in a single run");
    }
    int matSize;
    if (context.getUseSymmetries()) {
      matSize = int(3 * numStates);
    } else {
      matSize = int(numStates);
    }

    // user info about memory
    if (mpi->mpiHead()) {
      double x = pow(matSize, 2) / pow(1024., 3) * sizeof(double);
      std::cout << "Allocating " << x << " GB (" << x / (mpi->getSize())
                << " GB per process) for the scattering matrix.\n"
                << std::endl;
    }

    theMatrix = ParallelMatrix<double>(matSize, matSize);

    // calc matrix and linewidth.
    builder(&internalDiagonal, emptyVector, emptyVector);
  } else {
    // calc linewidths only
    builder(&internalDiagonal, emptyVector, emptyVector);
  }
}

VectorBTE ScatteringMatrix::diagonal() {
  if (constantRTA) {
    VectorBTE diagonal(statisticsSweep, outerBandStructure, 1);
    diagonal.setConst(twoPi / context.getConstantRelaxationTime());
    return diagonal;
  } else {
    return internalDiagonal;
  }
}

VectorBTE ScatteringMatrix::offDiagonalDot(VectorBTE &inPopulation) {
  if (highMemory) {
    VectorBTE outPopulation(statisticsSweep, outerBandStructure, 3);
    // note: we are assuming that ScatteringMatrix has numCalculations = 1

#pragma omp declare reduction (+: VectorBTE: omp_out.data=omp_out.data+omp_in.data)\
initializer(omp_priv=VectorBTE(omp_orig.statisticsSweep, \
                               omp_orig.bandStructure, omp_orig.dimensionality))
    auto allLocalStates = theMatrix.getAllLocalStates();
    size_t numAllLocalStates = allLocalStates.size();

    if (context.getUseSymmetries()) {
#pragma omp parallel for reduction(+ : outPopulation)
      for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
        auto tup = allLocalStates[iTup];
        int iMat1 = std::get<0>(tup);
        int iMat2 = std::get<1>(tup);
        auto t1 = getSMatrixIndex(iMat1);
        auto t2 = getSMatrixIndex(iMat2);
        int iBte1 = std::get<0>(t1).get();
        int iBte2 = std::get<0>(t2).get();
        if (iBte1 == iBte2)
          continue;
        int i = std::get<1>(t1).get();
        int j = std::get<1>(t2).get();
        outPopulation(0, i, iBte1) +=
            theMatrix(iMat1, iMat2) * inPopulation(0, j, iBte2);
      }
    } else {
#pragma omp parallel for reduction(+ : outPopulation)
      for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
        auto tup = allLocalStates[iTup];
        auto iBte1 = std::get<0>(tup);
        auto iBte2 = std::get<1>(tup);
        if (iBte1 == iBte2)
          continue;
        for (int i : {0, 1, 2}) {
          outPopulation(0, i, iBte1) +=
              theMatrix(iBte1, iBte2) * inPopulation(0, i, iBte2);
        }
      }
    }

    mpi->allReduceSum(&outPopulation.data);
    return outPopulation;
  } else {
    VectorBTE outPopulation = dot(inPopulation);
#pragma omp parallel for collapse(3) default(none)                             \
    shared(outPopulation, internalDiagonal, inPopulation, numCalculations, numStates)
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      for (int iDim = 0; iDim < 3; iDim++) {
        for (int iBte = 0; iBte < numStates; iBte++) {
          outPopulation(iCalc, iDim, iBte) -= internalDiagonal(iCalc, 0, iBte) *
                                              inPopulation(iCalc, iDim, iBte);
        }
      }
    }
    return outPopulation;
  }
}

std::vector<VectorBTE>
ScatteringMatrix::offDiagonalDot(std::vector<VectorBTE> &inPopulations) {
  // outPopulation = outPopulation - internalDiagonal * inPopulation;
  std::vector<VectorBTE> outPopulations = dot(inPopulations);
  for (unsigned int iVec = 0; iVec < inPopulations.size(); iVec++) {
#pragma omp parallel for collapse(3)
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      for (int iDim = 0; iDim < 3; iDim++) {
        for (int iBte = 0; iBte < numStates; iBte++) {
          outPopulations[iVec](iCalc, iDim, iBte) -=
              internalDiagonal(iCalc, 0, iBte) *
              inPopulations[iVec](iCalc, iDim, iBte);
        }
      }
    }
  }
  return outPopulations;
}

VectorBTE ScatteringMatrix::dot(VectorBTE &inPopulation) {
  if (highMemory) {
    VectorBTE outPopulation(statisticsSweep, outerBandStructure, 3);
    // note: we are assuming that ScatteringMatrix has numCalculations = 1

    auto allLocalStates = theMatrix.getAllLocalStates();
    size_t numAllLocalStates = allLocalStates.size();

 #pragma omp declare reduction (+: VectorBTE: omp_out.data=omp_out.data+omp_in.data)\
 initializer(omp_priv=VectorBTE(omp_orig.statisticsSweep, \
                                omp_orig.bandStructure, omp_orig.dimensionality))

    if (context.getUseSymmetries()) {
#pragma omp parallel for reduction(+ : outPopulation)
      for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
        auto tup = allLocalStates[iTup];
        int iMat1 = std::get<0>(tup);
        int iMat2 = std::get<1>(tup);
        auto t1 = getSMatrixIndex(iMat1);
        auto t2 = getSMatrixIndex(iMat2);
        int iBte1 = std::get<0>(t1).get();
        int iBte2 = std::get<0>(t2).get();

        if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
            excludeIndices.end())
          continue;
        if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) !=
            excludeIndices.end())
          continue;

        int i = std::get<1>(t1).get();
        int j = std::get<1>(t2).get();
        outPopulation(0, i, iBte1) +=
            theMatrix(iMat1, iMat2) * inPopulation(0, j, iBte2);
      }
    } else {
#pragma omp parallel for reduction(+ : outPopulation)
      for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
        auto tup = allLocalStates[iTup];
        auto iBte1 = std::get<0>(tup);
        auto iBte2 = std::get<1>(tup);

        if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
            excludeIndices.end())
          continue;
        if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) !=
            excludeIndices.end())
          continue;

        for (int i : {0, 1, 2}) {
          outPopulation(0, i, iBte1) +=
              theMatrix(iBte1, iBte2) * inPopulation(0, i, iBte2);
        }
      }
    }

    mpi->allReduceSum(&outPopulation.data);
    return outPopulation;
  } else {
    VectorBTE outPopulation(statisticsSweep, outerBandStructure, 3);
    outPopulation.data.setZero();
    std::vector<VectorBTE> outPopulations;
    std::vector<VectorBTE> inPopulations;
    inPopulations.push_back(inPopulation);
    outPopulations.push_back(outPopulation);
    builder(nullptr, inPopulations, outPopulations);
    return outPopulations[0];
  }
}

std::vector<VectorBTE>
ScatteringMatrix::dot(std::vector<VectorBTE> &inPopulations) {
  if (highMemory) {
    std::vector<VectorBTE> outPopulations;
    for (auto inPopulation : inPopulations) {
      VectorBTE outPopulation(statisticsSweep, outerBandStructure, 3);
      outPopulation = dot(inPopulation);
      outPopulations.push_back(outPopulation);
    }
    return outPopulations;
  } else {
    std::vector<VectorBTE> outPopulations;
    for (unsigned int i = 0; i < inPopulations.size(); i++) {
      VectorBTE outPopulation(statisticsSweep, outerBandStructure, 3);
      outPopulations.push_back(outPopulation);
    }
    builder(nullptr, inPopulations, outPopulations);
    return outPopulations;
  }
}

// set,unset the scaling of omega = A/sqrt(bose1*bose1+1)/sqrt(bose2*bose2+1)
void ScatteringMatrix::a2Omega() {
  if (!highMemory) {
    Error("a2Omega only works if the matrix is stored in memory");
  }

  if (theMatrix.rows() == 0) {
    Error("The scattering matrix hasn't been built yet");
  }

  if (isMatrixOmega) { // it's already with the scaling of omega
    return;
  }

  int iCalc = 0; // as there can only be one temperature

  auto particle = outerBandStructure.getParticle();
  auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
  double temp = calcStatistics.temperature;
  double chemPot = calcStatistics.chemicalPotential;

  auto allLocalStates = theMatrix.getAllLocalStates();
  size_t numAllLocalStates = allLocalStates.size();
#pragma omp parallel for
  for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
    auto tup = allLocalStates[iTup];

    int iBte1, iBte2, iMat1, iMat2;
    StateIndex is1Idx(-1), is2Idx(-1);
    if (context.getUseSymmetries()) {
      iMat1 = std::get<0>(tup);
      iMat2 = std::get<1>(tup);
      auto tup1 = getSMatrixIndex(iMat1);
      auto tup2 = getSMatrixIndex(iMat2);
      BteIndex iBte1Idx = std::get<0>(tup1);
      BteIndex iBte2Idx = std::get<0>(tup2);
      iBte1 = iBte1Idx.get();
      iBte2 = iBte2Idx.get();
      is1Idx = outerBandStructure.bteToState(iBte1Idx);
      is2Idx = outerBandStructure.bteToState(iBte2Idx);
    } else {
      iBte1 = std::get<0>(tup);
      iBte2 = std::get<1>(tup);
      iMat1 = iBte1;
      iMat2 = iBte2;
      is1Idx = StateIndex(iBte1);
      is2Idx = StateIndex(iBte2);
    }
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
        excludeIndices.end())
      continue;
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) !=
        excludeIndices.end())
      continue;

    double en1 = outerBandStructure.getEnergy(is1Idx);
    double en2 = outerBandStructure.getEnergy(is2Idx);

    // n(n+1) for bosons, n(1-n) for fermions
    double term1 = particle.getPopPopPm1(en1, temp, chemPot);
    double term2 = particle.getPopPopPm1(en2, temp, chemPot);

    if (iBte1 == iBte2) {
      internalDiagonal(0, 0, iBte1) /= term1;
    }

    theMatrix(iMat1, iMat2) /= sqrt(term1 * term2);
  }
  isMatrixOmega = true;
}

// add a flag to remember if we have A or Omega
// void ScatteringMatrix::omega2A() {
//  if (!highMemory) {
//    Error("a2Omega only works if the matrix is stored in memory");
//  }
//
//  if (theMatrix.rows() == 0) {
//    Error("The scattering matrix hasn't been built yet");
//  }
//
//  if (!isMatrixOmega) { // it's already with the scaling of A
//    return;
//  }
//
//  int iCalc = 0; // as there can only be one temperature
//
//  auto particle = outerBandStructure.getParticle();
//  auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
//  double temp = calcStatistics.temperature;
//  double chemPot = calcStatistics.chemicalPotential;
//
//  for (auto tup : theMatrix.getAllLocalStates()) {
//    auto ind1 = std::get<0>(tup);
//    auto ind2 = std::get<1>(tup);
//    if (std::find(excludeIndices.begin(), excludeIndices.end(), ind1) !=
//        excludeIndices.end())
//      continue;
//    if (std::find(excludeIndices.begin(), excludeIndices.end(), ind2) !=
//        excludeIndices.end())
//      continue;
//
//    double en1 = outerBandStructure.getEnergy(ind1);
//    double en2 = outerBandStructure.getEnergy(ind2);
//
//    // n(n+1) for bosons, n(1-n) for fermions
//    double term1 = particle.getPopPopPm1(en1, temp, chemPot);
//    double term2 = particle.getPopPopPm1(en2, temp, chemPot);
//
//    if (ind1 == ind2) {
//      internalDiagonal(iCalc, 0, ind1) *= term1;
//    }
//
//    theMatrix(ind1, ind2) *= sqrt(term1 * term2);
//  }
//  isMatrixOmega = false;
//}

// to compute the RTA, get the single mode relaxation times
VectorBTE ScatteringMatrix::getSingleModeTimes() {
  if (constantRTA) {
    VectorBTE times(statisticsSweep, outerBandStructure, 1);
    times.setConst(context.getConstantRelaxationTime() / twoPi);
    times.excludeIndices = excludeIndices;
    return times;
  } else {
    if (isMatrixOmega) {
      VectorBTE times = internalDiagonal.reciprocal();
      times.excludeIndices = excludeIndices;
      return times;
    } else { // A_nu,nu = N(1+-N) / tau
      VectorBTE times = internalDiagonal;
      auto particle = outerBandStructure.getParticle();
#pragma omp parallel for
      for (int iBte = 0; iBte < numStates; iBte++) {
        BteIndex iBteIdx(iBte);
        StateIndex isIdx = outerBandStructure.bteToState(iBteIdx);
        double en = outerBandStructure.getEnergy(isIdx);
        for (int iCalc = 0; iCalc < internalDiagonal.numCalculations; iCalc++) {
          auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
          double temp = calcStatistics.temperature;
          double chemPot = calcStatistics.chemicalPotential;
          // n(n+1) for bosons, n(1-n) for fermions
          double popTerm = particle.getPopPopPm1(en, temp, chemPot);
          times(iCalc, 0, iBte) = popTerm / internalDiagonal(iCalc, 0, iBte);
        }
      }
      times.excludeIndices = excludeIndices;
      return times;
    }
  }
}

VectorBTE ScatteringMatrix::getLinewidths() {
  if (constantRTA) {
    VectorBTE linewidths(statisticsSweep, outerBandStructure, 1);
    linewidths.setConst(twoPi / context.getConstantRelaxationTime());
    linewidths.excludeIndices = excludeIndices;
    return linewidths;
  } else {
    VectorBTE linewidths = internalDiagonal;
    linewidths.excludeIndices = excludeIndices;
    auto particle = outerBandStructure.getParticle();

    if (isMatrixOmega) {
      return linewidths;

    } else {
      // A_nu,nu = Gamma / N(1+N) for phonons, A_nu,nu = Gamma for electrons
      if (particle.isElectron()) {
        Error("Attempting to use a numerically unstable quantity");
        // popTerm could be = 0
      }
#pragma omp parallel for
      for (int iBte = 0; iBte < numStates; iBte++) {
        auto iBteIdx = BteIndex(iBte);
        StateIndex isIdx = outerBandStructure.bteToState(iBteIdx);
        double en = outerBandStructure.getEnergy(isIdx);
        for (int iCalc = 0; iCalc < internalDiagonal.numCalculations; iCalc++) {
          auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
          double temp = calcStatistics.temperature;
          double chemPot = calcStatistics.chemicalPotential;
          // n(n+1) for bosons, n(1-n) for fermions
          double popTerm = particle.getPopPopPm1(en, temp, chemPot);
          linewidths(iCalc, 0, iBte) =
              internalDiagonal(iCalc, 0, iBte) / popTerm;
        }
      }
      linewidths.excludeIndices = excludeIndices;
      return linewidths;
    }
  }
}

void ScatteringMatrix::outputToJSON(const std::string &outFileName) {

  if (!mpi->mpiHead())
    return;

  VectorBTE times = getSingleModeTimes();
  VectorBTE tmpLinewidths = getLinewidths();

  std::string particleType;
  auto particle = outerBandStructure.getParticle();
  double energyConversion = energyRyToEv;
  std::string energyUnit = "eV";
  // we need an extra factor of two pi, likely because of unit conversion
  // (perhaps h vs hbar)
  double energyToTime = energyRyToFs/twoPi;
  if (particle.isPhonon()) {
    particleType = "phonon";
    energyUnit = "meV";
    energyConversion *= 1000;
  } else {
    particleType = "electron";
  }

  // need to store as a vector format with dimensions
  // iCalc, ik. ib, iDim (where iState is unfolded into
  // ik, ib) for the velocities and lifetimes, no dim for energies
  std::vector<std::vector<std::vector<double>>> outTimes;
  std::vector<std::vector<std::vector<double>>> outLinewidths;
  std::vector<std::vector<std::vector<std::vector<double>>>> velocities;
  std::vector<std::vector<std::vector<double>>> energies;
  std::vector<double> temps;
  std::vector<double> chemPots;
  std::vector<double> dopings;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStatistics.temperature;
    double chemPot = calcStatistics.chemicalPotential;
    double doping = calcStatistics.doping;
    temps.push_back(temp * temperatureAuToSi);
    chemPots.push_back(chemPot * energyConversion);
    dopings.push_back(doping);

    std::vector<std::vector<double>> wavevectorsT;
    std::vector<std::vector<double>> wavevectorsL;
    std::vector<std::vector<std::vector<double>>> wavevectorsV;
    std::vector<std::vector<double>> wavevectorsE;
    // loop over wavevectors
    for (int ik : outerBandStructure.irrPointsIterator()) {
      auto ikIndex = WavevectorIndex(ik);

      std::vector<double> bandsT;
      std::vector<double> bandsL;
      std::vector<std::vector<double>> bandsV;
      std::vector<double> bandsE;
      // loop over bands here
      // get numBands at this point, in case it's an active band structure
      for (int ib = 0; ib < outerBandStructure.getNumBands(ikIndex); ib++) {
        auto ibIndex = BandIndex(ib);
        int is = outerBandStructure.getIndex(ikIndex, ibIndex);
        StateIndex isIdx(is);
        double ene = outerBandStructure.getEnergy(isIdx);
        auto vel = outerBandStructure.getGroupVelocity(isIdx);
        bandsE.push_back(ene * energyConversion);
        int iBte = int(outerBandStructure.stateToBte(isIdx).get());
        double tau = times(iCalc, 0, iBte); // only zero dim is meaningful
        bandsT.push_back(tau * energyToTime);
        double linewidth =
            tmpLinewidths(iCalc, 0, iBte); // only zero dim is meaningful
        bandsL.push_back(linewidth * energyConversion);

        std::vector<double> iDimsV;
        // loop over dimensions
        for (int iDim : {0, 1, 2}) {
          iDimsV.push_back(vel[iDim] * velocityRyToSi);
        }
        bandsV.push_back(iDimsV);
      }
      wavevectorsT.push_back(bandsT);
      wavevectorsL.push_back(bandsL);
      wavevectorsV.push_back(bandsV);
      wavevectorsE.push_back(bandsE);
    }
    outTimes.push_back(wavevectorsT);
    outLinewidths.push_back(wavevectorsL);
    velocities.push_back(wavevectorsV);
    energies.push_back(wavevectorsE);
  }

  auto points = outerBandStructure.getPoints();
  std::vector<std::vector<double>> meshCoordinates;
  for (int ik : outerBandStructure.irrPointsIterator()) {
    // save the wavevectors
    auto ikIndex = WavevectorIndex(ik);
    auto coord =
        points.cartesianToCrystal(outerBandStructure.getWavevector(ikIndex));
    meshCoordinates.push_back({coord[0], coord[1], coord[2]});
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";
  if (particle.isElectron()) {
    output["dopingConcentrations"] = dopings;
    output["dopingConcentrationUnit"] =
        "cm$^{-" + std::to_string(context.getDimensionality()) + "}$";
  }
  output["linewidths"] = outLinewidths;
  output["linewidthsUnit"] = energyUnit;
  output["relaxationTimes"] = outTimes;
  output["relaxationTimeUnit"] = "fs";
  output["velocities"] = velocities;
  output["velocityUnit"] = "m/s";
  output["energies"] = energies;
  output["energyUnit"] = energyUnit;
  output["wavevectorCoordinates"] = meshCoordinates;
  output["coordsType"] = "lattice";
  output["particleType"] = particleType;
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

void ScatteringMatrix::relaxonsToJSON(const std::string &outFileName,
                                      const Eigen::VectorXd &eigenvalues) {
  if (!mpi->mpiHead()) {
    return;
  }

  Eigen::VectorXd times = 1. / eigenvalues.array();

  std::string particleType;
  auto particle = outerBandStructure.getParticle();
  if (particle.isPhonon()) {
    particleType = "phonon";
  } else {
    particleType = "electron";
  }

  double energyToTime = energyRyToFs;
  double energyConversion = energyRyToEv;

  std::string energyUnit = "eV";

  // need to store as a vector format with dimensions
  // iCalc, ik. ib, iDim (where iState is unfolded into
  // ik, ib) for the velocities and lifetimes, no dim for energies
  std::vector<std::vector<double>> outTimes;
  std::vector<double> temps;
  std::vector<double> chemPots;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStatistics.temperature;
    double chemPot = calcStatistics.chemicalPotential;
    temps.push_back(temp * temperatureAuToSi);
    chemPots.push_back(chemPot * energyConversion);

    std::vector<double> thisTime;
    for (auto x : times) {
      thisTime.push_back(x * energyToTime);
    }
    outTimes.push_back(thisTime);
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["chemicalPotentials"] = chemPots;
  output["relaxationTimes"] = outTimes;
  output["relaxationTimeUnit"] = "fs";
  output["particleType"] = particleType;
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

std::tuple<Eigen::VectorXd, ParallelMatrix<double>>
ScatteringMatrix::diagonalize() {

  // user info about memory
  {
    memoryUsage();
    double xx;
    double x = 2 * pow(theMatrix.rows(), 2) / pow(1024., 3) * sizeof(xx);
    // 2 because one is for eigenvectors, another is the copy of the matrix
    std::cout << std::setprecision(4);
    if (mpi->mpiHead()) {
      std::cout << "About to allocate " << x
                << " (GB) for the scattering matrix diagonalization.\n"
                << std::endl;
    }
  }

  auto tup = theMatrix.diagonalize();
  auto eigenvalues = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);

  // place eigenvalues in an VectorBTE object
  Eigen::VectorXd eigenValues(theMatrix.rows());
  for (int is = 0; is < eigenValues.size(); is++) {
    eigenValues(is) = eigenvalues[is];
  }

  return {eigenValues, eigenvectors};
}

std::vector<std::tuple<std::vector<int>, int>>
ScatteringMatrix::getIteratorWavevectorPairs(const int &switchCase,
                                             const bool &rowMajor) {
  if (rowMajor) { // case for el-ph scattering
    std::vector<std::tuple<std::vector<int>, int>> pairIterator;

    if (switchCase == 1 || switchCase == 2) { // case for linewidth construction
      // here I parallelize over ik1
      // which is the outer loop on q-points
      std::vector<int> k1Iterator =
          outerBandStructure.parallelIrrPointsIterator();

      // I don't parallelize the inner band structure, the inner loop
      std::vector<int> k2Iterator(innerBandStructure.getNumPoints());
      // populate vector with integers from 0 to numPoints-1
      std::iota(std::begin(k2Iterator), std::end(k2Iterator), 0);

      for (int ik1 : k1Iterator) {
        auto t = std::make_tuple(k2Iterator, ik1);
        pairIterator.push_back(t);
      }
    } else { // case el-ph scattering, matrix in memory

      // here we operate assuming innerBandStructure=outerBandStructure
      // list in form [[0,0],[1,0],[2,0],...]
      std::set<std::pair<int, int>> localPairs;
      std::vector<std::tuple<int, int>> localStates = theMatrix.getAllLocalStates();
      int nlocalStates = localStates.size();
#pragma omp parallel default(none) shared(localPairs, theMatrix, localStates, nlocalStates)
      {
        std::vector<std::pair<int, int>> localPairsPrivate;
        // first unpack Bloch Index and get all wavevector pairs
#pragma omp for nowait
        for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){
          std::tuple<int,int> tup0 = localStates[ilocalState];
          auto iMat1 = std::get<0>(tup0);
          auto iMat2 = std::get<1>(tup0);
          auto tup1 = getSMatrixIndex(iMat1);
          auto tup2 = getSMatrixIndex(iMat2);
          BteIndex iBte1 = std::get<0>(tup1);
          BteIndex iBte2 = std::get<0>(tup2);
          // map the index on the irreducible points of BTE to band structure
          // index
          StateIndex is1 = outerBandStructure.bteToState(iBte1);
          StateIndex is2 = outerBandStructure.bteToState(iBte2);
          auto tuple1 = outerBandStructure.getIndex(is1);
          auto tuple2 = outerBandStructure.getIndex(is2);
          WavevectorIndex ik1Index = std::get<0>(tuple1);
          WavevectorIndex ik2Index = std::get<0>(tuple2);
          int ik1Irr = ik1Index.get();
          int ik2Irr = ik2Index.get();
          for (int ik2 :
               outerBandStructure.getReducibleStarFromIrreducible(ik2Irr)) {
            std::pair<int, int> xx = std::make_pair(ik1Irr, ik2);
            localPairsPrivate.push_back(xx);
          }
        }
#pragma omp critical
        for (auto y : localPairsPrivate) {
          localPairs.insert(y);
        }
      }

      // find set of q1
      // this because we have duplicates with different band indices.
      std::set<int> q1IndexesSet;
      for (std::pair<int, int> p : localPairs) {
        auto iq1 = p.first;
        q1IndexesSet.insert(iq1);
      }

      // we move the set into a vector
      // vector, unlike set, has well-defined indices for its elements
      std::vector<int> q1Indexes;
      for (int x : q1IndexesSet) {
        q1Indexes.push_back(x);
      }

      for (int iq1 : q1Indexes) {
        std::vector<int> x;
        auto y = std::make_tuple(x, iq1);
        pairIterator.push_back(y);
      }

      for (std::pair<int, int> p : localPairs) {
        int iq1 = p.first;
        int iq2 = p.second;

        auto idx = std::find(q1Indexes.begin(), q1Indexes.end(), iq1);
        if (idx != q1Indexes.end()) {
          auto i = idx - q1Indexes.begin();
          // note: get<> returns a reference to the tuple elements
          std::get<0>(pairIterator[i]).push_back(iq2);
        } else {
          Error("iq1 not found, not supposed to happen");
        }
      }
    }

    // Patch to make pooled interpolation of coupling work
    // we need to make sure each MPI process in the pool calls
    // the calculation of the coupling interpolation the same number of times
    // Hence, we correct pairIterator so that it does have the same size across
    // the pool

    if (mpi->getSize(mpi->intraPoolComm) > 1) {
      // std::vector<std::tuple<std::vector<int>, int>> pairIterator;
      auto myNumK1 = int(pairIterator.size());
      int numK1 = myNumK1;
      mpi->allReduceMax(&numK1, mpi->intraPoolComm);

      while (myNumK1 < numK1) {
        std::vector<int> dummyVec;
        dummyVec.push_back(-1);
        auto tt = std::make_tuple(dummyVec,-1);
        pairIterator.push_back(tt);
        myNumK1++;
      }
    }
    return pairIterator;

  } else { // case for ph_scattering

    if (switchCase == 1 || switchCase == 2) { // case for dot
      // must parallelize over the inner band structure (iq2 in phonons)
      // which is the outer loop on q-points
      size_t a = innerBandStructure.getNumPoints();
      auto q2Iterator = mpi->divideWorkIter(a);
      std::vector<int> q1Iterator = outerBandStructure.irrPointsIterator();

      std::vector<std::tuple<std::vector<int>, int>> pairIterator;
      for (int iq2 : q2Iterator) {
        auto t = std::make_tuple(q1Iterator, iq2);
        pairIterator.push_back(t);
      }

      return pairIterator;

    } else { // case for constructing A matrix

      std::set<std::pair<int, int>> localPairs;
      std::vector<std::tuple<int, int>> localStates = theMatrix.getAllLocalStates();
      int nlocalStates = localStates.size();
#pragma omp parallel default(none) shared(theMatrix, localPairs, localStates, nlocalStates)
      {
        std::vector<std::pair<int, int>> localPairsPrivate;
        // first unpack Bloch Index and get all wavevector pairs
#pragma omp for nowait
        for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){
          std::tuple<int,int> tup0 = localStates[ilocalState];
          auto iMat1 = std::get<0>(tup0);
          auto iMat2 = std::get<1>(tup0);
          auto tup1 = getSMatrixIndex(iMat1);
          auto tup2 = getSMatrixIndex(iMat2);
          BteIndex iBte1 = std::get<0>(tup1);
          BteIndex iBte2 = std::get<0>(tup2);
          // map the index on the irreducible points of BTE to band structure
          // index
          StateIndex is1 = outerBandStructure.bteToState(iBte1);
          StateIndex is2 = outerBandStructure.bteToState(iBte2);
          auto tuple1 = outerBandStructure.getIndex(is1);
          auto tuple2 = outerBandStructure.getIndex(is2);
          WavevectorIndex iq1Index = std::get<0>(tuple1);
          WavevectorIndex iq2Index = std::get<0>(tuple2);
          int iq1Irr = iq1Index.get();
          int iq2Irr = iq2Index.get();
          for (int iq2 :
               outerBandStructure.getReducibleStarFromIrreducible(iq2Irr)) {
            std::pair<int, int> xx = std::make_pair(iq1Irr, iq2);
            localPairsPrivate.push_back(xx);
          }
        }
#pragma omp critical
        for (auto y : localPairsPrivate) {
          localPairs.insert(y);
        }
      }

      // find set of q2
      std::set<int> q2IndexesSet;
      for (auto tup : localPairs) {
        auto iq2 = std::get<1>(tup);
        q2IndexesSet.insert(iq2);
      }

      // we move the set into a vector
      // vector, unlike set, has well-defined indices for its elements
      std::vector<int> q2Indexes;
      for (int x : q2IndexesSet) {
        q2Indexes.push_back(x);
      }

      std::vector<std::tuple<std::vector<int>, int>> pairIterator;
      for (int iq2 : q2Indexes) {
        std::vector<int> x;
        auto y = std::make_tuple(x, iq2);
        pairIterator.push_back(y);
      }

      for (std::pair<int, int> p : localPairs) {
        int iq1 = p.first;
        int iq2 = p.second;

        auto idx = std::find(q2Indexes.begin(), q2Indexes.end(), iq2);
        if (idx != q2Indexes.end()) {
          auto i = idx - q2Indexes.begin();
          // note: get<> returns a reference to the tuple elements
          std::get<0>(pairIterator[i]).push_back(iq1);
        } else {
          Error("iq2 not found, not supposed to happen");
        }
      }
      return pairIterator;
    }
  }
}

int ScatteringMatrix::getSMatrixIndex(BteIndex &bteIndex,
                                      CartIndex &cartIndex) {
  if (context.getUseSymmetries()) {
    int is = bteIndex.get();
    int alpha = cartIndex.get();
    return compress2Indices(is, alpha, numStates, 3);
  } else {
    return bteIndex.get();
  }
}

std::tuple<BteIndex, CartIndex>
ScatteringMatrix::getSMatrixIndex(const int &iMat) {
  if (context.getUseSymmetries()) {
    auto t = decompress2Indices(iMat, numStates, 3);
    return {BteIndex(std::get<0>(t)), CartIndex(std::get<1>(t))};
  } else {
    return {BteIndex(iMat), CartIndex(0)};
  }
}

void ScatteringMatrix::symmetrize() {
  // note: if the matrix is not stored in memory, it's not trivial to enforce
  // that the matrix is symmetric

  if (context.getUseSymmetries()) {
    Error("Symmetrization of the scattering matrix with symmetries"
          " is not verified");
    // not only, probably it doesn't make sense. To be checked
  }

  if (highMemory) {
    ParallelMatrix<double> newMatrix = theMatrix;
    newMatrix *= 0.;

    for (int iRank = 0; iRank < mpi->getSize(); iRank++) {
      int thisSize = 0;
      if (iRank == mpi->getRank()) {
        thisSize = int(theMatrix.getAllLocalStates().size());
      }
      mpi->allReduceSum(&thisSize);

      std::vector<int> index1(thisSize, 0);
      std::vector<int> index2(thisSize, 0);
      std::vector<int> index3(thisSize, 0);
      std::vector<int> index4(thisSize, 0);

      if (iRank == mpi->getRank()) {
        auto allLocalStates = theMatrix.getAllLocalStates();
        size_t numAllLocalStates = allLocalStates.size();

        for (size_t i=0; i<numAllLocalStates; i++) {
          auto tup = allLocalStates[i];
          int iMat1 = std::get<0>(tup);
          int iMat2 = std::get<1>(tup);
          int jMat1, jMat2;
//          if (context.getUseSymmetries()) {
//            Error("Symmetrization of the scattering matrix with symmetries"
//                  " is not verified");
//            // The matrix may not be symmetric
//
//            auto t1 = getSMatrixIndex(iMat1);
//            auto t2 = getSMatrixIndex(iMat2);
//            BteIndex iBte1 = std::get<0>(t1);
//            BteIndex iBte2 = std::get<0>(t2);
//            CartIndex ii = std::get<1>(t1);
//            CartIndex jj = std::get<1>(t2);
//            jMat1 = getSMatrixIndex(iBte1, ii);
//            jMat2 = getSMatrixIndex(iBte2, jj);
//          } else {
          jMat1 = iMat1;
          jMat2 = iMat2;
//          }

          index1[i] = iMat1;
          index2[i] = iMat2;
          index3[i] = jMat2;
          index4[i] = jMat1;
        }
      }

      mpi->allReduceSum(&index1);
      mpi->allReduceSum(&index2);
      mpi->allReduceSum(&index3);
      mpi->allReduceSum(&index4);

      std::vector<double> values(thisSize, 0.);
#pragma omp parallel for
      for (int i = 0; i < thisSize; i++) {
        values[i] = (theMatrix(index1[i], index2[i]) +
                     theMatrix(index3[i], index4[i])) *
                    0.5;
      }
      mpi->allReduceSum(&values);

#pragma omp parallel for
      for (int i = 0; i < thisSize; i++) {
        newMatrix(index1[i], index2[i]) = values[i];
        newMatrix(index3[i], index4[i]) = values[i];
      }
    }
    theMatrix = newMatrix;
  }
}

void ScatteringMatrix::degeneracyAveragingLinewidths(VectorBTE *linewidth) {
  for (int ik : outerBandStructure.irrPointsIterator()) {
    WavevectorIndex ikIdx(ik);
    Eigen::VectorXd en = outerBandStructure.getEnergies(ikIdx);
    auto numBands = int(en.size());

    Eigen::VectorXd linewidthTmp(numBands);
    linewidthTmp.setZero();

    for (int ib1 = 0; ib1 < numBands; ib1++) {
      double ekk = en(ib1);
      int n = 0;
      double tmp2 = 0.;
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        double ekk2 = en(ib2);

        BandIndex ibIdx(ib2);
        int is = outerBandStructure.getIndex(ikIdx, ibIdx);
        StateIndex isIdx(is);
        BteIndex ind1Idx = outerBandStructure.stateToBte(isIdx);
        int iBte1 = ind1Idx.get();

        if (abs(ekk2 - ekk) < 1.0e-6) {
          n++;
          tmp2 = tmp2 + linewidth->data(0, iBte1);
        }
      }
      linewidthTmp(ib1) = tmp2 / float(n);
    }

    for (int ib1 = 0; ib1 < numBands; ib1++) {
      BandIndex ibIdx(ib1);
      int is = outerBandStructure.getIndex(ikIdx, ibIdx);
      StateIndex isIdx(is);
      BteIndex ind1Idx = outerBandStructure.stateToBte(isIdx);
      int iBte1 = ind1Idx.get();
      linewidth->data(0, iBte1) = linewidthTmp(ib1);
    }
  }
}
