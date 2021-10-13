#include "polarization_app.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <iomanip>

#include "constants.h"
#include "elph_qe_to_phoebe_app.h"
#include "exceptions.h"
#include "periodic_table.h"
#include "pugixml.hpp"
#include "qe_input_parser.h"
#include "statistics_sweep.h"

//Eigen::MatrixXd ElectronPolarizationApp::testElectronicPolarization(
//    ElectronH0Wannier &h0, StatisticsSweep &statisticsSweep, Crystal &crystal) {
//  // here I compute the polarization as the sum of the Wannier function centers
//  // i.e. summing <0n|r|0n>
//  int numCalculations = statisticsSweep.getNumCalculations();
//  Eigen::MatrixXd electronicPolarization = Eigen::MatrixXd::Zero(numCalculations,3);
//  double volume = crystal.getVolumeUnitCell();
//
//  // #NOTE::: this index is hard-coded for the test example!
//  int iR0 = 364;
//  for (int iw=0; iw<h0.getNumBands(); iw++) {
//    for (int iCalc=0; iCalc<numCalculations; iCalc++) {
//      for (int i : {0, 1, 2}) {
//        // the 2 is the spin degeneracy
//        electronicPolarization(iCalc, i) -= 2. * h0.rMatrix(i, iR0, iw, iw).real() / volume;
//      }
//    }
//  }
//  for (int iCalc=0; iCalc<numCalculations; iCalc++) {
//    for (int i : {0, 1, 2}) {
//      std::cout << i << " " << electronicPolarization(iCalc,i) << "\n";
//    }
//  }
//  return electronicPolarization;
//}

std::vector<std::string> splitt(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  if (delimiter == ' ') {
    for (std::string s2; tokenStream >> s2;) {
      tokens.push_back(s2);
    }
  } else {
    while (std::getline(tokenStream, token, delimiter)) {
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                  token.end());
      tokens.push_back(token);
    }
  }

  return tokens;
}

Crystal parseCrystalFromXml(Context &context, const std::string &xmlPath) {
  //  Here we read the XML file of quantum espresso to get the crystal info.
  if (xmlPath.empty()) {
    Error("Must provide an XML file name");
  }
  // load and parse XML file using a library
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());
  if (not result) {
    Error("Error parsing XML file");
  }
  if(mpi->mpiHead()) {
    std::cout << "Reading crystal from " << xmlPath << "." << std::endl;
  }

  pugi::xml_node output = doc.child("qes:espresso").child("output");

  // atomic species

  pugi::xml_node atomicSpeciesXML = output.child("atomic_species");
  int numElements = atomicSpeciesXML.attribute("ntyp").as_int();
  std::vector<std::string> speciesNames;
  Eigen::VectorXd speciesMasses(numElements);
  int i = 0;
  for (pugi::xml_node species : atomicSpeciesXML.children("species")) {
    speciesNames.emplace_back(species.attribute("name").value());
    speciesMasses(i) = species.child("mass").text().as_double(); // in amu
    i += 1;
  }

  // atomic structure

  pugi::xml_node atomicStructure = output.child("atomic_structure");
  int numAtoms = atomicStructure.attribute("nat").as_int();

  //  we read the atomic positions

  pugi::xml_node atomicPositionsXML = atomicStructure.child("atomic_positions");
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  i = 0;
  int atomId = 0;
  std::string thisAtomName;
  for (pugi::xml_node atom : atomicPositionsXML.children("atom")) {
    thisAtomName = atom.attribute("name").value();
    // the XML doesn't describe atoms with a tag ID, but using names
    // here I find the index of the species in speciesNames, given the name
    auto itr =
        std::find(speciesNames.begin(), speciesNames.end(), thisAtomName);
    if (itr != speciesNames.cend()) {
      atomId = std::distance(speciesNames.begin(), itr);
    } else {
      Error("Element not found in XML");
    }
    atomicSpecies(i) = atomId;

    // note: atomic positions are in Cartesian coordinates in units of angstroms
    std::vector<std::string> lineSplit = splitt(atom.child_value(), ' ');
    atomicPositions(i, 0) = std::stod(lineSplit[0]);
    atomicPositions(i, 1) = std::stod(lineSplit[1]);
    atomicPositions(i, 2) = std::stod(lineSplit[2]);
    i++;
  }

  // we read the unit cell

  Eigen::Matrix3d directUnitCell;
  Eigen::Vector3d thisValues;
  pugi::xml_node cell = atomicStructure.child("cell");
  {
    std::vector<std::string> lineSplit = splitt(cell.child_value("a1"), ' ');
    directUnitCell(0, 0) = std::stod(lineSplit[0]);
    directUnitCell(1, 0) = std::stod(lineSplit[1]);
    directUnitCell(2, 0) = std::stod(lineSplit[2]);
    lineSplit = splitt(cell.child_value("a2"), ' ');
    directUnitCell(0, 1) = std::stod(lineSplit[0]);
    directUnitCell(1, 1) = std::stod(lineSplit[1]);
    directUnitCell(2, 1) = std::stod(lineSplit[2]);
    lineSplit = splitt(cell.child_value("a3"), ' ');
    directUnitCell(0, 2) = std::stod(lineSplit[0]);
    directUnitCell(1, 2) = std::stod(lineSplit[1]);
    directUnitCell(2, 2) = std::stod(lineSplit[2]);
  }

  // Now we parse the electronic structure

  // Initialize the crystal class

  int dimensionality = context.getDimensionality();
  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, dimensionality);
  crystal.print();

  if(mpi->mpiHead()) {
    std::cout << "Done reading crystal." << std::endl;
  }

  return crystal;
}

// Compute the electronic polarization using the Berry connection
void ElectronPolarizationApp::run(Context &context) {
  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  }
  Warning("Make sure to set hasSpinOrbit=true if used in a DFT code");

  Crystal crystal = parseCrystalFromXml(context, context.getXMLPath());

  // Read the necessary input files
  auto tup = QEParser::parseElHarmonicWannier(context, &crystal);
  // auto crystal = std::get<0>(tup);
  auto h0 = std::get<1>(tup);

  // here we parse info on the coarse grid, and do Bloch to Wannier transform
  setVariablesFromFiles(context, crystal);

  // first we make compute the band structure on the fine grid
  Points points(crystal, context.getKMesh());
  bool withVelocities = false;
  bool withEigenvectors = true; // used for Berry connection
  FullBandStructure bandStructure =
      h0.populate(points, withVelocities, withEigenvectors);

  // let's fix the chemical potential now that we have the band structure
  StatisticsSweep statisticsSweep(context, &bandStructure);
  int numCalculations = statisticsSweep.getNumCalculations();

  // now we build the Berry connection
  auto berryConnection = h0.getBerryConnection(bandStructure);

  // now compute the polarization
  auto electronicPolarization = getElectronicPolarization(crystal, statisticsSweep,
      context, bandStructure, spinFactor, berryConnection);
  auto ionicPolarization = getIonicPolarization(crystal, statisticsSweep);
  auto polarization = electronicPolarization + ionicPolarization;

  double conversionPolarization = electronSi / pow(bohrRadiusSi, 2);

  // compute the polarization quantum eR/V
  Eigen::Vector3d polarizationQuantum;
  {
    auto cell = crystal.getDirectUnitCell();
    // cell.row(i) is the lattice vector i
    double volume = crystal.getVolumeUnitCell();
    for (int i : {0,1,2}) {
      // this is the norm of R_i lattice vector
      double normI = cell.row(i).norm();
      // https://arxiv.org/pdf/1202.1831.pdf
      // Note that in non-magnetic systems, the polarization quantum is usually
      // multiplied by an additional factor of two because the up- and down-spin
      // electrons are equivalent, and shifting an up-spin electron by a lattice
      // vector also shifts the corresponding down-spin electron
      polarizationQuantum(i) = normI * spinFactor / volume;
    }
  }
  // normI should be updated most likely
  Warning("Some checks to do on P quantum for non cubic crystals");

  // print results to screen
  if (mpi->mpiHead()) {
    std::cout << "Polarization quantum eR/V: "
            << conversionPolarization * polarizationQuantum.transpose() << " (C/m^2)\n";
    std::cout << "\n";
    std::cout << "Temperature (K), doping (cm^-3), chemical potential (eV)\n";
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto sc = statisticsSweep.getCalcStatistics(iCalc);
      std::cout << std::fixed << std::setprecision(6);
      std::cout << sc.temperature * temperatureAuToSi << "\t" << sc.doping
                << "\t" << sc.chemicalPotential * energyRyToEv << "\n";
      std::cout << "Polarization (C/m^2):";
      std::cout << std::scientific;
      for (int i : {0, 1, 2}) {
        std::cout << "\t" << polarization(iCalc, i) * conversionPolarization;
      }
      std::cout << "\n";
    }
  }

  if (mpi->mpiHead()) {
    std::cout << "Electron polarization computed" << std::endl;
  }
}

void ElectronPolarizationApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  if (context.getDopings().size() == 0 && context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or dopings must be set");
  }
  throwErrorIfUnset(context.getXMLPath(), "xmlPath");
  throwErrorIfUnset(context.getScfOutputFileName(), "scfOutputFileName");
}

void ElectronPolarizationApp::setVariablesFromFiles(Context &context,
                                                       Crystal &crystal) {
  std::string wannierPrefix = context.getWannier90Prefix();

  // here we parse the k-mesh
  Eigen::Vector3i kMesh = Eigen::Vector3i::Zero();
  if (mpi->mpiHead()){
    // open input file
    std::string fileName = wannierPrefix + ".win";
    std::ifstream infile(fileName);
    if (not infile.is_open()) {
      Error("Wannier input file not found");
    }
    std::string line;
    while (std::getline(infile, line)) {
      if (!line.empty()) {
        auto x = Context::split(line, ' ');
        if (x[0] == "mp_grid") {
          for (int i : {0, 1, 2}) {
            kMesh(i) = std::stoi(x[i+2]);
          }
        }
      }
    }
  }
  mpi->allReduceSum(&kMesh);
  if (kMesh(0)==0 || kMesh(1)==0 || kMesh(2)==0) {
    Error("Failed to parse mp_grid from Wannier90");
  }

  // initialize points class, to represent the coarse grid used in Wannier90
  Points kPoints(crystal, kMesh);
  int numPoints = kMesh.prod();

  // parse the coordinates of the kpoints used in the calculation
  Eigen::MatrixXd kGridFull = Eigen::MatrixXd::Zero(3, numPoints);
  if (mpi->mpiHead()) {
    std::vector<std::string> lines;
    {
      // open input file
      std::string fileName = wannierPrefix + ".win";
      if (mpi->mpiHead())
        std::cout << "Reading in " << fileName << "." << std::endl;
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
  mpi->allReduceSum(&kGridFull);

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
//  auto numEntangledBands = int(uMatrices.dimension(0));// number of entangled bands
  int bandsOffset = ElPhQeToPhoebeApp::computeOffset(energies, wannierPrefix);
  int numFilledWannier = numElectrons - bandsOffset * 2;
  // this makes the calculation of fermi level work
  context.setNumOccupiedStates(numFilledWannier);

  //---------------------------------------------------------

  // parse valence charges
  // read the projections from QE scf.out file

  valenceCharges.resize(crystal.getNumSpecies());
  valenceCharges.setZero();
  if ( mpi->mpiHead() ){
    // first, parse all the scf output file
    std::vector<std::string> lines;
    {
      // open input file
      std::string fileName = context.getScfOutputFileName();
      std::ifstream infile(fileName);
      if (not infile.is_open()) {
        Error("Scf output file not found");
      }
      std::string line;
      while (std::getline(infile, line)) {
        lines.push_back(line);
      }
    }

    int counter = -1;
    for (auto line : lines) {
      counter++;
      if (line.empty()) continue;
      auto x = Context::split(line, ' ');
      if (x[0] == "PseudoPot.") {
        std::string species = x[4];
        species[0] = std::toupper(species[0]);
        auto xx = Context::split(lines[counter + 3], ' ');
        double charge = std::stod(xx[5]);

        int iSpecies = -1;
        int ctr = 0;
        for (std::string y : crystal.getSpeciesNames()) {
          if (y.find(species) != std::string::npos) {
            iSpecies = ctr;
          }
          ctr++;
        }
        if (iSpecies == -1) {
          Error("Species not found");
        }
        valenceCharges(iSpecies) = charge;
      }
    }
  }
  mpi->allReduceSum(&valenceCharges);

  if (mpi->mpiHead()) {
    std::cout << "Done reading files." << std::endl;
  }
}

Eigen::MatrixXd ElectronPolarizationApp::getElectronicPolarization(
    Crystal &crystal,
    StatisticsSweep &statisticsSweep,
    Context &context,
    BaseBandStructure &bandStructure, const double &spinFactor,
    const Eigen::Tensor<double, 3> &berryConnection) {

  int numCalculations = statisticsSweep.getNumCalculations();
  auto particle = bandStructure.getParticle();

  Eigen::MatrixXd polarization = Eigen::MatrixXd::Zero(numCalculations, 3);
  double volume = crystal.getVolumeUnitCell();
  double norm = spinFactor / context.getKMesh().prod() / volume;

//#pragma omp parallel default(none) shared(mpi, bandStructure, particle, polarization, statisticsSweep, numCalculations, norm, berryConnection)
  {
//    Eigen::MatrixXd privatePolarization = Eigen::MatrixXd::Zero(numCalculations, 3);
//#pragma omp for nowait
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
          polarization(iCalc, i) -= population * berryConnection(ib, i, ik) * norm;
        }
      }
    }

//#pragma omp critical
//    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
//      for (int i : {0, 1, 2}) {
//        polarization(iCalc, i) += privatePolarization(iCalc, i);
//      }
//    }

    mpi->allReduceSum(&polarization);
  }

  return polarization;
}

// now we add the ionic polarization
Eigen::MatrixXd ElectronPolarizationApp::getIonicPolarization(
    Crystal &crystal, StatisticsSweep &statisticsSweep) {

  if (mpi->mpiHead()) {
    std::cout << valenceCharges.transpose() << "\n";
    std::cout << crystal.getAtomicNames()[0] << " "
              << crystal.getAtomicNames()[1] << " "
              << crystal.getAtomicNames()[2] << "\n";
  }

  int numCalculations = statisticsSweep.getNumCalculations();
  double volume = crystal.getVolumeUnitCell();
  int numAtoms = crystal.getNumAtoms();

  Eigen::MatrixXd polarization = Eigen::MatrixXd::Zero(numCalculations, 3);
  Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
#pragma omp parallel for default(none) shared(polarization, numAtoms, atomicPositions, crystal, valenceCharges, volume, numCalculations)
  for (int iAt = 0; iAt < numAtoms; iAt++) {
    Eigen::Vector3d position = atomicPositions.row(iAt);
    int iType = crystal.getAtomicSpecies()(iAt);
    int valenceCharge = valenceCharges(iType);

    for (int i : {0, 1, 2}) {
      double x = valenceCharge * position(i) / volume;
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        polarization(iCalc, i) += x;
      }
    }
  }

  return polarization;
}
