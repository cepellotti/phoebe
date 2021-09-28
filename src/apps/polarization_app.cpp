#include "polarization_app.h"

#include <algorithm>
#include <fstream>
#include <string>

#include "constants.h"
#include "elph_qe_to_phoebe_app.h"
#include "exceptions.h"
#include "periodic_table.h"
#include "pugixml.hpp"
#include "qe_input_parser.h"
#include "statistics_sweep.h"

// Compute the electronic polarization using the Berry connection
void ElectronPolarizationApp::run(Context &context) {
  std::cout << "Starting polarization calculation" << std::endl;

  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  }
  Warning("Make sure to set hasSpinOrbit=true if used in a DFT code");

  // Read the necessary input files
  auto tup = QEParser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto h0 = std::get<1>(tup);

  // first we make compute the band structure on the fine grid
  Points points(crystal, context.getKMesh());
  bool withVelocities = false;
  bool withEigenvectors = false;
  FullBandStructure bandStructure =
      h0.populate(points, withVelocities, withEigenvectors);

  // here we parse info on the coarse grid, and do Bloch to Wannier transform
  projectionBlochToWannier(context, crystal, h0);

  // now we build the atomic orbital projection
  int numAtoms = crystal.getNumAtoms();
  Eigen::Tensor<double, 3> projections(numAtoms, points.getNumPoints(), h0.getNumBands());
  projections = getProjectionsBlochOnAtoms(points, h0);

  // now we build the Berry connection

  Eigen::Tensor<double, 3> berryConnection(points.getNumPoints(),
                                           h0.getNumBands(), 3);
  {
    berryConnection.setZero();
#pragma omp parallel for default(none) shared(h0, points, berryConnection)
    for (int ik = 0; ik < points.getNumPoints(); ik++) {
      auto point = points.getPoint(ik);
      auto thisBerryConnection = h0.getBerryConnection(point);
      for (int ib = 0; ib < h0.getNumBands(); ib++) {
        for (int i : {0, 1, 2}) {
          berryConnection(ik, ib, i) = thisBerryConnection[i](ib, ib).real();
        }
      }
    }
  }

  // before moving on, we need to fix the chemical potential
  StatisticsSweep statisticsSweep(context, &bandStructure);
  int numCalculations = statisticsSweep.getNumCalculations();

  // now we can compute the electronic polarization

  auto tE = getElectronicPolarization(crystal, statisticsSweep,
      context, bandStructure, spinFactor, berryConnection, projections);
  auto electronicPolarization = std::get<0>(tE);
  auto electronicProjectedPolarization = std::get<1>(tE);

  auto tI = getIonicPolarization(crystal, statisticsSweep);
  auto ionicPolarization = std::get<0>(tI);
  auto ionicProjectedPolarization = std::get<1>(tI);

  auto polarization = electronicPolarization + ionicPolarization;
  double volume = crystal.getVolumeUnitCell();
  double conversionPolarization = electronSi / pow(bohrRadiusSi, 2);

  // Save results to file
  std::ofstream outfile("./polarization.dat");
  outfile << "# chemical potential (eV), doping (cm^-3), temperature (K), "
             "polarization[x,y,z] (a.u.)\n";
  outfile << "Polarization quantum e/V: "
          << conversionPolarization/volume << "\n";
  outfile << "Polarization conversion au to SI: "
          << conversionPolarization << "\n";
  outfile << "numCalc, numAtoms\n";
  outfile << numCalculations << " " << numAtoms << "\n";
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto sc = statisticsSweep.getCalcStatistics(iCalc);
    outfile << sc.chemicalPotential * energyRyToEv << "\t" << sc.doping
            << "\t" << sc.temperature * temperatureAuToSi;
    for (int i : {0, 1, 2}) {
      outfile << "\t" << polarization(iCalc, i);
    }
    outfile << "\n";

    for (int iAt=0; iAt<numAtoms; iAt++) {
      outfile << iAt;
      for (int i : {0, 1, 2}) {
        outfile << "\t" << electronicProjectedPolarization(iAt, iCalc, i);
      }
      outfile << "\n";
    }
    for (int iAt=0; iAt<numAtoms; iAt++) {
      outfile << iAt;
      for (int i : {0, 1, 2}) {
        outfile << "\t" << ionicProjectedPolarization(iAt, iCalc, i);
      }
      outfile << "\n";
    }
  }

  std::cout << "Electron polarization computed" << std::endl;
}

void ElectronPolarizationApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  if (context.getDopings().size() == 0 && context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or dopings must be set");
  }
  throwErrorIfUnset(context.getXMLPath(), "xmlPath");
  throwErrorIfUnset(context.getProjectionsFileName(), "projectionsFileName");
}

Eigen::Tensor<double, 3> ElectronPolarizationApp::getProjectionsBlochOnAtoms(Points &points,
                                                                             ElectronH0Wannier &h0) {
  auto crystal = points.getCrystal();
  int numAtoms = crystal.getNumAtoms();
  Eigen::Tensor<double, 3> projections(numAtoms, points.getNumPoints(), h0.getNumBands());
  projections.setZero();

  auto bVectors = h0.bravaisVectors;
  auto bVectorsDeg = h0.vectorsDegeneracies;

  int numBands = h0.getNumBands();

#pragma omp parallel for default(none) shared(projections, points, h0, numAtoms, bVectors, bVectorsDeg, numBands)
  for (int ik = 0; ik < points.getNumPoints(); ik++) {
    Eigen::Vector3d k = points.getPointCoordinates(ik, Points::cartesianCoordinates);

    std::vector<std::complex<double>> phases(bVectors.cols());
    for (int iR = 0; iR < bVectors.cols(); iR++) {
      Eigen::Vector3d R = bVectors.col(iR);
      double phase = k.dot(R);
      std::complex<double> phaseFactor = {cos(phase), sin(phase)};
      phases[iR] = phaseFactor / bVectorsDeg(iR);
    }

    // first we diagonalize the hamiltonian
    auto tup = h0.diagonalizeFromCoordinates(k);
    // auto ens = std::get<0>(tup);
    auto eigenvectors = std::get<1>(tup);

    for (int iAt = 0; iAt < numAtoms; iAt++) {

      // now we do the Fourier back-transform to reciprocal space
      Eigen::MatrixXcd projectionW = Eigen::MatrixXcd::Zero(numBands, numBands);
      for (int iR = 0; iR < bVectors.cols(); iR++) {
        for (int m = 0; m < numBands; m++) {
          for (int n = 0; n < numBands; n++) {
            projectionW(m, n) += phases[iR] * projectionWannierSpace(iAt, iR, m, n);
          }
        }
      }

      // rotation into Bloch space
      Eigen::MatrixXcd thisProjection(numBands, numBands);
      thisProjection = eigenvectors.adjoint() * projectionW * eigenvectors;

      for (int ib = 0; ib < numBands; ib++) {
        projections(iAt, ik, ib) = thisProjection(ib, ib).real();
      }
    }
  }
  return projections;
}

void ElectronPolarizationApp::projectionBlochToWannier(Context &context,
                                                       Crystal &crystal,
                                                       ElectronH0Wannier &h0) {
  std::string wannierPrefix = context.getWannier90Prefix();

  // here we parse the k-mesh
  Eigen::Vector3i kMesh;
  {
    // open input file
    std::string fileName = wannierPrefix + ".win";
    std::ifstream infile(fileName);
    if (not infile.is_open()) {
      Error("Wannier input file not found");
    }
    std::string line;
    while (std::getline(infile, line)) {

      if (!line.empty()) {// nothing to do
        auto tup = Context::parseParameterNameValue(line);
        auto parameterName = std::get<0>(tup);
        auto val = std::get<1>(tup);
        if (parameterName == "mp_grid") {
          auto x = Context::split(val, ' ');
          for (int i : {0, 1, 2}) {
            kMesh(i) = std::stoi(x[i]);
          }
        }
      }
    }
  }

  // initialize points class, to represent the coarse grid used in Wannier90
  Points kPoints(crystal, kMesh);
  int numPoints = kMesh.prod();

  // parse the coordinates of the kpoints used in the calculation
  Eigen::MatrixXd kGridFull = Eigen::MatrixXd::Zero(3, numPoints);
  {
    std::vector<std::string> lines;
    {
      // open input file
      std::string fileName = wannierPrefix + ".win";
      std::ifstream infile(fileName);
      if (not infile.is_open()) {
        Error("Wannier input file not found");
      }
      std::string line;
      while (std::getline(infile, line)) {
        lines.push_back(line);
      }
    }

    int counter = 0;
    {
      std::string x = "begin kpoints";
      for (auto line : lines) {
        if (line.find(x) != std::string::npos) {
          break;
        }
        counter++;
      }
    }

    for (int ik = 0; ik < numPoints; ik++) {
      int lineIndex = counter + 1 + ik;
      auto x = Context::split(lines[lineIndex], ' ');
      for (int i : {0, 1, 2}) {
        kGridFull(i, ik) = std::stod(x[i]);
      }
    }
  }

  // set up the bravais lattice vectors
  bVectors = h0.bravaisVectors;
  bVectorsDeg = h0.vectorsDegeneracies;
  int numBravaisVectors = h0.numVectors;
  int numWannier = h0.numBands;
  int numAtoms = crystal.getNumAtoms();

  //--------------------------------------------------------

  // here we parse the XML file
  int numQEBands, numElectrons;
  Eigen::VectorXd energies;
  {
    std::string fileName = context.getXMLPath();
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(fileName.c_str());
    if (not result) {
      Error("Error parsing XML file");
    }
    if (mpi->mpiHead())
      std::cout << "Reading in " << fileName << "." << std::endl;
    pugi::xml_node output = doc.child("qes:espresso").child("output");

    pugi::xml_node bandStructureXML = output.child("band_structure");
    numQEBands = bandStructureXML.child("nbnd").text().as_int();
    // note: nelec is written as double in the XML file!
    numElectrons = int(bandStructureXML.child("nelec").text().as_double());

    energies.resize(numQEBands);
    // parse energies at k=0
    {
      for (pugi::xml_node kPoint : bandStructureXML.children("ks_energies")) {
        std::string val = kPoint.child_value("k_point");
        std::vector<std::string> lineSplit = Context::split(val, ' ');
        Eigen::Vector3d p;
        p(0) = std::stod(lineSplit[0]);
        p(1) = std::stod(lineSplit[1]);
        p(2) = std::stod(lineSplit[2]);
        if (p.norm() < 1e-12) {
          std::string val2 = kPoint.child_value("eigenvalues");
          std::replace(val2.begin(), val2.end(), '\n', ' ');
          lineSplit = Context::split(val2, ' ');

          int j = 0;
          for (auto x : lineSplit) {
            energies(j) = std::stod(lineSplit[j]);
            j++;
          }
          // QE XML energies are in Hartree units. Must convert to rydberg
          energies *= 2.;
          break;
        }
      }
    }
  }

  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

  auto uMatrices = ElPhQeToPhoebeApp::setupRotationMatrices(wannierPrefix, kPoints);
  auto numEntangledBands = int(uMatrices.dimension(0));// number of entangled bands
  int bandsOffset = ElPhQeToPhoebeApp::computeOffset(energies, wannierPrefix);
  int numFilledWannier = numElectrons - bandsOffset * 2;
  // this makes the calculation of fermi level work
  context.setNumOccupiedStates(numFilledWannier);

  // build the map from coarse QE grid to our internal one
  Eigen::VectorXi ikMap(numPoints);
  {
#pragma omp parallel for default(none) shared(numPoints, kGridFull, kPoints, ikMap)
    for (int ikOld = 0; ikOld < numPoints; ikOld++) {
      Eigen::Vector3d kOld = kGridFull.col(ikOld);
      int ikNew = kPoints.getIndex(kOld);
      ikMap(ikOld) = ikNew;
    }
  }

  Eigen::Tensor<double, 3> projectionsQE;
  {// read the projections from QE

    // first, parse all the projections file
    std::vector<std::string> lines;
    {
      // open input file
      std::string fileName = context.getProjectionsFileName();
      std::ifstream infile(fileName);
      if (not infile.is_open()) {
        Error("Projection file not found");
      }
      std::string line;
      while (std::getline(infile, line)) {
        lines.push_back(line);
      }
    }

    // read the number of projections
    int numProjections;
    {
      std::string line;
      line = lines[4 + numAtoms + crystal.getNumSpecies()];
      auto x = Context::split(line, ' ');
      numProjections = std::stoi(x[0]);
    }

    valenceCharges.resize(crystal.getNumSpecies());
    {
      int offset = 4;
      for (int iType = 0; iType < crystal.getNumSpecies(); iType++) {
        std::string line = lines[offset + iType];
        auto x = Context::split(line, ' ');
        std::string speciesName = x[1];
        auto listOfSpecies = crystal.getSpeciesNames();

        int iSpecies = -1;
        int counter = 0;
        for (std::string y : listOfSpecies) {
          if (y.find(speciesName) != std::string::npos) {
            iSpecies = counter;
          }
          counter++;
        }
        if (iSpecies == -1) {
          Error("Species not found");
        }
        valenceCharges[iSpecies] = std::stod(x[2]);
      }
    }

    // read all projections, and integrate over the atom index
    Eigen::Tensor<double, 3> tmpProj(numAtoms, numPoints, numQEBands);
    tmpProj.setZero();
    int offset = 6 + numAtoms + crystal.getNumSpecies();
    int counter = offset;
    for (int iProj = 0; iProj < numProjections; iProj++) {
      auto x = Context::split(lines[counter], ' ');
      int iAt = std::stoi(x[1]);
      counter++;
      for (int ik = 0; ik < numPoints; ik++) {
        for (int ib = 0; ib < numQEBands; ib++) {
          auto xx = Context::split(lines[counter], ' ');
          double thisProj = std::stod(xx[2]);
          tmpProj(iAt, ik, ib) += thisProj;
          counter++;
        }
      }
    }
    projectionsQE.resize(numAtoms, numPoints, numEntangledBands);
    projectionsQE.setZero();
    for (int iAt = 0; iAt < numAtoms; iAt++) {
      for (int ik = 0; ik < numPoints; ik++) {
        for (int ib = 0; ib < numEntangledBands; ib++) {
          projectionsQE(iAt, ikMap(ik), ib) = tmpProj(iAt, ik, bandsOffset + ib);
        }
      }
    }
  }

  // now do transformation
  projectionWannierSpace.resize(numAtoms, numBravaisVectors, numWannier, numWannier);
  projectionWannierSpace.setZero();
  for (int ik = 0; ik < numPoints; ik++) {
    Eigen::Vector3d k1C = kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);

    // u has size (numBands, numWannier, numKPoints)
    Eigen::MatrixXcd uK(numEntangledBands, numWannier);
    for (int i = 0; i < numEntangledBands; i++) {
      for (int j = 0; j < numWannier; j++) {
        uK(i, j) = uMatrices(i, j, ik);
      }
    }

    // Eq. 26 of Giustino PRB 2007. Note that the U are inverted
    Eigen::Tensor<std::complex<double>, 3> h0K1(numAtoms, numEntangledBands, numEntangledBands);
    for (int iAt = 0; iAt < numAtoms; iAt++) {
      Eigen::MatrixXcd tmp(numEntangledBands, numEntangledBands);
      for (int ib = 0; ib < numEntangledBands; ib++) {
        tmp(ib, ib) = {projectionsQE(iAt, ik, ib), 0};
      }
      Eigen::MatrixXcd tmp2(numWannier, numWannier);
      tmp2 = uK * tmp * uK.adjoint();
      for (int iw1 = 0; iw1 < numWannier; iw1++) {
        for (int iw2 = 0; iw2 < numWannier; iw2++) {
          h0K1(iAt, iw1, iw2) = tmp2(iw1, iw2);
        }
      }
    }

    for (int iR = 0; iR < numBravaisVectors; iR++) {
      Eigen::Vector3d R = bVectors.col(iR);
      double arg = k1C.dot(R);
      std::complex<double> phase = exp(-complexI * arg) / double(numPoints);
      for (int m = 0; m < numWannier; m++) {
        for (int n = 0; n < numWannier; n++) {
          for (int iAt = 0; iAt < numAtoms; iAt++) {
            projectionWannierSpace(iAt, iR, m, n) += phase * h0K1(iAt, m, n);
          }
        }
      }
    }
  }
}

std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
ElectronPolarizationApp::getElectronicPolarization(
    Crystal &crystal,
    StatisticsSweep &statisticsSweep,
    Context &context,
    BaseBandStructure &bandStructure, const double &spinFactor,
    const Eigen::Tensor<double, 3> &berryConnection,
    const Eigen::Tensor<double, 3> &projections) {

  int numCalculations = statisticsSweep.getNumCalculations();
  int numAtoms = crystal.getNumAtoms();
  auto particle = bandStructure.getParticle();

  Eigen::MatrixXd polarization = Eigen::MatrixXd::Zero(numCalculations, 3);
  Eigen::Tensor<double, 3> projectedPolarization(numAtoms, numCalculations, 3);
  projectedPolarization.setConstant(0.);
  double volume = crystal.getVolumeUnitCell();
  double norm = spinFactor / context.getKMesh().prod() / volume;

#pragma omp parallel for default(none) shared(std::cout, bandStructure, statisticsSweep, numCalculations, particle, berryConnection, polarization, norm, numAtoms, projections, projectedPolarization)
  for (int is : bandStructure.parallelStateIterator()) {
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    auto t = bandStructure.getIndex(isIdx);
    int ik = std::get<0>(t).get();
    int ib = std::get<1>(t).get();

    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto sc = statisticsSweep.getCalcStatistics(iCalc);
      double temp = sc.temperature;
      double chemPot = sc.chemicalPotential;
      double population = particle.getPopulation(energy, temp, chemPot);
      for (int i : {0, 1, 2}) {
        polarization(iCalc, i) -= population * berryConnection(ik, ib, i) * norm;
        for (int iAt = 0; iAt < numAtoms; iAt++) {
          projectedPolarization(iAt, iCalc, i) -= projections(iAt, ik, ib) * population * berryConnection(ik, ib, i) * norm;
        }
      }
    }
  }
  mpi->allReduceSum(&polarization);
  mpi->allReduceSum(&projectedPolarization);

  return {polarization, projectedPolarization};
}

// now we add the ionic polarization
std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
ElectronPolarizationApp::getIonicPolarization(
    Crystal &crystal, StatisticsSweep &statisticsSweep) {

  int numCalculations = statisticsSweep.getNumCalculations();
  double volume = crystal.getVolumeUnitCell();
  int numAtoms = crystal.getNumAtoms();

  Eigen::MatrixXd polarization = Eigen::MatrixXd::Zero(numCalculations, 3);
  Eigen::Tensor<double, 3> projectedPolarization(numAtoms, numCalculations, 3);
  projectedPolarization.setConstant(0.);


  Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
  std::vector<std::string> atomicNames = crystal.getAtomicNames();
  for (int iAt = 0; iAt < numAtoms; iAt++) {
    Eigen::Vector3d position = atomicPositions.row(iAt);
    int iType = crystal.getAtomicSpecies()(iAt);
    int valenceCharge = valenceCharges(iType);
    for (int i : {0, 1, 2}) {
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        polarization(iCalc, i) += valenceCharge * position(i) / volume;
        projectedPolarization(iAt, iCalc, i) += valenceCharge * position(i) / volume;
      }
    }
  }
  return {polarization, projectedPolarization};
}
