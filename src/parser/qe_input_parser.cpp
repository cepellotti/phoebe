#include <algorithm> // to use .remove_if
#include <fstream>
#include <iomanip> // to declare istringstream
#include <iostream>
#include <math.h>   // round()
#include <stdlib.h> // abs()
#include <string>
#include <vector>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "particle.h"
#include "periodic_table.h"
#include "pugixml.hpp"
#include "qe_input_parser.h"
#include "utilities.h"

void latgen(const int ibrav, Eigen::VectorXd &celldm,
            Eigen::Matrix3d &unitCell) {
  //  sets up the crystallographic vectors a1, a2, and a3.
  //
  //  ibrav is the structure index:
  //    1  cubic P (sc)
  //    2  cubic F (fcc)
  //    3  cubic I (bcc)
  //    4  hexagonal and trigonal P
  //    5  trigonal R, 3-fold axis c
  //    6  tetragonal P (st)
  //    7  tetragonal I (bct)
  //    8  orthorhombic P
  //    9  1-face (C) centered orthorhombic
  //   10  all face centered orthorhombic
  //   11  body centered orthorhombic
  //   12  monoclinic P (unique axis: c)
  //   13  one face (base) centered monoclinic
  //   14  triclinic P
  //  Also accepted:
  //    0  "free" structure
  //  -12  monoclinic P (unique axis: b)
  //   -3  cubic bcc with a more symmetric choice of axis
  //   -5  trigonal R, threefold axis along (111)
  //   -9  alternate description for base centered orthorhombic
  //  -13  one face (base) centered monoclinic (unique axis: b)
  //   91  1-face (A) centered orthorombic
  //
  //  celldm are parameters which fix the shape of the unit cell
  //  volumeUnitCell is the unit-cell volume
  //
  //  NOTA BENE: all axis sets are right-handed
  //  Boxes for US PPs do not work properly with left-handed axis

  const double sr2 = 1.414213562373, sr3 = 1.732050807569;

  //  user-supplied lattice vectors

  Eigen::Vector3d a1, a2, a3;

  a1 = unitCell.col(0);
  a2 = unitCell.col(1);
  a3 = unitCell.col(2);

  if (ibrav == 0) {
    if (a1.norm() == 0. || a2.norm() == 0. || a3.norm() == 0.) {
      Error e("wrong at for ibrav=0", 1);
    }
    if (celldm(0) != 0.) {
      // ... input at are in units of alat => convert them to a.u.
      unitCell *= celldm(0);
    } else {
      // ... input at are in atomic units: define celldm(1) from a1
      celldm(0) = a1.norm();
    }
  } else {
    a1.setZero();
    a2.setZero();
    a3.setZero();
  }

  if (celldm(0) <= 0.) {
    Error e("wrong celldm(1)", 1);
  }

  //  index of bravais lattice supplied

  if (ibrav == 1) { // simple cubic lattice
    a1(0) = celldm(0);
    a2(1) = celldm(0);
    a3(2) = celldm(0);
  } else if (ibrav == 2) { //     fcc lattice
    double term = celldm(0) / 2.;
    a1(0) = -term;
    a1(2) = term;
    a2(1) = term;
    a2(2) = term;
    a3(0) = -term;
    a3(1) = term;
  } else if (abs(ibrav) == 3) { // bcc lattice
    double term = celldm(0) / 2.;
    for (int ir = 0; ir < 3; ir++) {
      a1(ir) = term;
      a2(ir) = term;
      a3(ir) = term;
    }
    if (ibrav < 0) {
      a1(0) = -a1(0);
      a2(1) = -a2(1);
      a3(2) = -a3(2);
    } else {
      a2(0) = -a2(0);
      a3(0) = -a3(0);
      a3(1) = -a3(1);
    }
  } else if (ibrav == 4) { // hexagonal lattice
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    double cbya = celldm(2);
    a1(0) = celldm(0);
    a2(0) = -celldm(0) / 2.;
    a2(1) = celldm(0) * sr3 / 2.;
    a3(2) = celldm(0) * cbya;

  } else if (abs(ibrav) == 5) { // trigonal lattice
    if (celldm(3) <= -0.5 || celldm(3) >= 1.) {
      Error e("wrong celldm(4)", abs(ibrav));
    }

    double term1 = sqrt(1. + 2. * celldm(3));
    double term2 = sqrt(1. - celldm(3));

    if (ibrav == 5) { // threefold axis along c (001)
      a2(1) = sr2 * celldm(0) * term2 / sr3;
      a2(2) = celldm(0) * term1 / sr3;
      a1(0) = celldm(0) * term2 / sr2;
      a1(1) = -a1(0) / sr3;
      a1(2) = a2(2);
      a3(0) = -a1(0);
      a3(1) = a1(1);
      a3(2) = a2(2);
    } else if (ibrav == -5) { // threefold axis along (111)
      // Notice that in the cubic limit (alpha=90, celldm(4)=0,
      // term1=term2=1)
      // does not yield the x,y,z axis, but an equivalent rotated triplet:
      //   a/3 (-1,2,2), a/3 (2,-1,2), a/3 (2,2,-1)
      // If you prefer the x,y,z axis as cubic limit, you should modify
      // the definitions of a1(1) and a1(2) as follows:'
      // a1(1) = celldm(1)*(term1+2.0_dp*term2)/3.0_dp
      // a1(2) = celldm(1)*(term1-term2)/3.0_dp
      // (info by G. Pizzi and A. Cepellotti)
      a1(0) = celldm(0) * (term1 - 2. * term2) / 3.;
      a1(1) = celldm(0) * (term1 + term2) / 3.;
      a1(2) = a1(1);
      a2(0) = a1(2);
      a2(1) = a1(0);
      a2(2) = a1(1);
      a3(0) = a1(1);
      a3(1) = a1(2);
      a3(2) = a1(0);
    }
  } else if (ibrav == 6) { // tetragonal lattice
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", 6);
    }
    double cbya = celldm(2);
    a1(0) = celldm(0);
    a2(1) = celldm(0);
    a3(2) = celldm(0) * cbya;

  } else if (ibrav == 7) { // body centered tetragonal lattice
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", 7);
    }
    double cbya = celldm(2);
    a2(0) = celldm(0) / 2.;
    a2(1) = a2(0);
    a2(2) = cbya * celldm(0) / 2.;
    a1(0) = a2(0);
    a1(1) = -a2(0);
    a1(2) = a2(2);
    a3(0) = -a2(0);
    a3(1) = -a2(0);
    a3(2) = a2(2);
  } else if (ibrav == 8) { // Simple orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1);
    a3(2) = celldm(0) * celldm(2);
  } else if (abs(ibrav) == 9) {
    // One face (base) centered orthorhombic lattice  (C type)
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", abs(ibrav));
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", abs(ibrav));
    }
    if (ibrav == 9) { // old PWscf description
      a1(0) = 0.5 * celldm(0);
      a1(1) = a1(0) * celldm(1);
      a2(0) = -a1(0);
      a2(1) = a1(1);
    } else { // alternate description
      a1(0) = 0.5 * celldm(0);
      a1(1) = -a1(0) * celldm(1);
      a2(0) = a1(0);
      a2(1) = -a1(1);
    }
    a3(2) = celldm(0) * celldm(2);
  } else if (ibrav == 91) {
    // One face(base)centered orthorhombic lattice (A type)
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1) * 0.5;
    a2(2) = -celldm(0) * celldm(2) * 0.5;
    a3(1) = a2(1);
    a3(2) = -a2(2);
  } else if (ibrav == 10) { // All face centered orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    a2(0) = 0.5 * celldm(0);
    a2(1) = a2(0) * celldm(1);
    a1(0) = a2(0);
    a1(2) = a2(0) * celldm(2);
    a3(1) = a2(0) * celldm(1);
    a3(2) = a1(2);
  } else if (ibrav == 11) { // Body centered orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    a1(0) = 0.5 * celldm(0);
    a1(1) = a1(0) * celldm(1);
    a1(2) = a1(0) * celldm(2);
    a2(0) = -a1(0);
    a2(1) = a1(1);
    a2(2) = a1(2);
    a3(0) = -a1(0);
    a3(1) = -a1(1);
    a3(2) = a1(2);
  } else if (ibrav == 12) {
    // Simple monoclinic lattice, unique (i.e. orthogonal to a) axis: c
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    if (abs(celldm(3)) >= 1.) {
      Error e("wrong celldm(4)", ibrav);
    }
    double sen = sqrt(1. - celldm(3) * celldm(3));
    a1(0) = celldm(0);
    a2(0) = celldm(0) * celldm(1) * celldm(3);
    a2(1) = celldm(0) * celldm(1) * sen;
    a3(2) = celldm(0) * celldm(2);
  } else if (ibrav == -12) {
    // Simple monoclinic lattice, unique axis: b (more common)
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", -ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", -ibrav);
    }
    if (abs(celldm(4)) >= 1.) {
      Error e("wrong celldm(5)", -ibrav);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1);
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(2) = celldm(0) * celldm(2) * sen;
  } else if (ibrav == 13) {
    // One face centered monoclinic lattice unique axis c
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    if (abs(celldm(3)) >= 1.) {
      Error e("wrong celldm(4)", ibrav);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = 0.5 * celldm(0);
    a1(2) = -a1(0) * celldm(2);
    a2(0) = celldm(0) * celldm(1) * celldm(2);
    a2(1) = celldm(0) * celldm(1) * sen;
    a3(0) = a1(0);
    a3(2) = -a1(2);
  } else if (ibrav ==
             -13) { // One face centered monoclinic lattice unique axis b
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", -ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", -ibrav);
    }
    if (abs(celldm(4)) >= 1.) {
      Error e("wrong celldm(5)", -ibrav);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = 0.5 * celldm(0);
    a1(1) = -a1(0) * celldm(1);
    a2(0) = a1(0);
    a2(1) = -a1(1);
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(2) = celldm(0) * celldm(2) * sen;
  } else if (ibrav == 14) { // Triclinic lattice
    if (celldm(1) <= 0.) {
      Error e("wrong celldm(2)", ibrav);
    }
    if (celldm(2) <= 0.) {
      Error e("wrong celldm(3)", ibrav);
    }
    if (abs(celldm(3)) >= 1.) {
      Error e("wrong celldm(4)", ibrav);
    }
    if (abs(celldm(4)) >= 1.) {
      Error e("wrong celldm(5)", ibrav);
    }
    if (abs(celldm(5)) >= 1.) {
      Error e("wrong celldm(6)", ibrav);
    }
    double singam = sqrt(1. - celldm(5) * celldm(5));
    double term =
        (1. + 2. * celldm(3) * celldm(4) * celldm(5) - celldm(3) * celldm(3) -
         celldm(4) * celldm(4) - celldm(5) * celldm(5));
    if (term < 0.) {
      Error e("celldm does not make sense, check your data", ibrav);
    }
    term = sqrt(term / (1. - celldm(5) * celldm(5)));
    a1(0) = celldm(0);
    a2(0) = celldm(0) * celldm(1) * celldm(5);
    a2(1) = celldm(0) * celldm(1) * singam;
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(1) =
        celldm(0) * celldm(2) * (celldm(3) - celldm(4) * celldm(5)) / singam;
    a3(2) = celldm(0) * celldm(2) * term;

  } else {
    Error e("nonexistent bravais lattice", ibrav);
  }

  if (ibrav != 0) {
    unitCell.col(0) = a1;
    unitCell.col(1) = a2;
    unitCell.col(2) = a3;
  }
}

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  if (delimiter == ' ') {
    for (std::string s; tokenStream >> s;) {
      tokens.push_back(s);
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

std::tuple<Crystal, PhononH0> QEParser::parsePhHarmonic(Context &context) {
  //  Here we read the dynamical matrix of interatomic force constants
  //	in real space.

  std::string fileName = context.getPhD2FileName();
  if (fileName == "") {
    Error e("Must provide a D2 file name", 1);
  }

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open()) {
    Error e("Dynamical matrix file not found", 1);
  }

  //  First line contains ibrav, celldm and other variables

  std::getline(infile, line);
  lineSplit = split(line, ' ');

  int numElements = std::stoi(lineSplit[0]);
  int numAtoms = std::stoi(lineSplit[1]);
  int ibrav = std::stoi(lineSplit[2]);

  Eigen::VectorXd celldm(6);
  celldm(0) = std::stod(lineSplit[3]);
  celldm(1) = std::stod(lineSplit[4]);
  celldm(2) = std::stod(lineSplit[5]);
  celldm(3) = std::stod(lineSplit[6]);
  celldm(4) = std::stod(lineSplit[7]);
  celldm(5) = std::stod(lineSplit[8]);

  Eigen::Matrix3d directUnitCell(3, 3);
  if (ibrav == 0) {
    // In this case, unitCell is written in the file (in angstroms?)
    for (int i = 0; i < 3; i++) {
      std::getline(infile, line);
      lineSplit = split(line, ' ');
      for (int j = 0; j < 3; j++) {
        directUnitCell(i, j) = std::stod(lineSplit[j]);
      }
    };
  };

  // generate the unit cell vectors (also for ibrav != 0)
  latgen(ibrav, celldm, directUnitCell);

  //  Next, we read the atomic species
  std::vector<std::string> speciesNames;
  Eigen::VectorXd speciesMasses(numElements);
  for (int i = 0; i < numElements; i++) {
    std::getline(infile, line);
    lineSplit = split(line, '\'');
    speciesNames.push_back(lineSplit[1]);
    speciesMasses(i) = std::stod(lineSplit[2]); // in rydbergs
  };

  //  we read the atomic positions
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    atomicSpecies(i) = std::stoi(lineSplit[1]) - 1;
    atomicPositions(i, 0) = std::stod(lineSplit[2]);
    atomicPositions(i, 1) = std::stod(lineSplit[3]);
    atomicPositions(i, 2) = std::stod(lineSplit[4]);
  }

  //  Read if hasDielectric
  std::getline(infile, line);
  line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
  bool hasDielectric;
  if (line == "T") {
    hasDielectric = true;
  } else {
    hasDielectric = false;
  }

  //	if there are the dielectric info, we can read dielectric matrix
  //	and the Born charges
  Eigen::MatrixXd dielectricMatrix(3, 3);
  dielectricMatrix.setZero();
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  if (hasDielectric) {
    for (int i = 0; i < 3; i++) {
      std::getline(infile, line);
      lineSplit = split(line, ' ');
      for (int j = 0; j < 3; j++) {
        dielectricMatrix(i, j) = std::stod(lineSplit[j]);
      }
    }

    for (int iAtom = 0; iAtom < numAtoms; iAtom++) {
      std::getline(infile, line);
      for (int i = 0; i < 3; i++) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        for (int j = 0; j < 3; j++) {
          bornCharges(iAtom, i, j) = std::stod(lineSplit[j]);
        }
      }
    }
  }

  //	Now we parse the coarse q grid
  std::getline(infile, line);
  lineSplit = split(line, ' ');
  Eigen::VectorXi qCoarseGrid(3);
  qCoarseGrid(0) = std::stoi(lineSplit[0]);
  qCoarseGrid(1) = std::stoi(lineSplit[1]);
  qCoarseGrid(2) = std::stoi(lineSplit[2]);

  Eigen::Tensor<double, 7> forceConstants(
      qCoarseGrid[0], qCoarseGrid[1], qCoarseGrid[2], 3, 3, numAtoms, numAtoms);

  int m1Test;
  int m2Test;
  int m3Test;
  double x;

  for (int ic = 0; ic < 3; ic++) {
    for (int jc = 0; jc < 3; jc++) {
      for (int iat = 0; iat < numAtoms; iat++) {
        for (int jat = 0; jat < numAtoms; jat++) {
          // a line containing ic, jc, iat, jat
          std::getline(infile, line);

          for (int r3 = 0; r3 < qCoarseGrid[2]; r3++) {
            for (int r2 = 0; r2 < qCoarseGrid[1]; r2++) {
              for (int r1 = 0; r1 < qCoarseGrid[0]; r1++) {
                std::getline(infile, line);
                std::istringstream iss(line);
                iss >> m1Test >> m2Test >> m3Test >> x;
                forceConstants(r1, r2, r3, ic, jc, iat, jat) = x;
              }
            }
          }
        }
      }
    }
  }

  infile.close();

  // Now we do postprocessing

  long dimensionality = context.getDimensionality();
  Crystal crystal(directUnitCell, atomicPositions, atomicSpecies, speciesNames,
                  speciesMasses, dimensionality);

  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error e("qCoarseGrid smaller than zero", 1);
  }

  //	Now, let's try to diagonalize some points, and start debugging at q=0

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, bornCharges,
                           forceConstants);
  dynamicalMatrix.setAcousticSumRule(context.getSumRuleD2());

  return {crystal, dynamicalMatrix};
};

bool QEParser::isQuantumEspressoXml(const std::string &fileName) {
  if (fileName.empty()) {
    Error e("Must provide an input H0 file name");
  }
  // open input file
  std::ifstream infile(fileName);
  if (not infile.is_open()) {
    Error e("H0 file not found", 1);
  }
  std::string tag = "espresso";
  std::string line;
  bool tagFound = false;
  while (std::getline(infile, line) && !tagFound) {
    if (line.find(tag) != std::string::npos) {
      tagFound = true;
      break;
    }
  }
  infile.close();
  return tagFound;
}

std::tuple<Crystal, ElectronH0Fourier>
QEParser::parseElHarmonicFourier(Context &context) {
  //  Here we read the XML file of quantum espresso.

  std::string fileName;
  fileName = context.getElectronH0Name();
  double fourierCutoff = context.getElectronFourierCutoff();

  if (fileName.empty()) {
    Error e("Must provide an XML file name");
  }

  std::vector<std::string> lineSplit;

  // load and parse XML file using pugi library
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(fileName.c_str());

  if (not result) {
    Error e("Error parsing XML file");
  }

  pugi::xml_node output = doc.child("qes:espresso").child("output");

  // atomic species

  pugi::xml_node atomicSpeciesXML = output.child("atomic_species");
  int numElements = atomicSpeciesXML.attribute("ntyp").as_int();
  std::vector<std::string> speciesNames;
  Eigen::VectorXd speciesMasses(numElements);
  int i = 0;
  for (pugi::xml_node species : atomicSpeciesXML.children("species")) {
    speciesNames.push_back(species.attribute("name").value());
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
    std::vector<std::string>::iterator itr =
        std::find(speciesNames.begin(), speciesNames.end(), thisAtomName);
    if (itr != speciesNames.cend()) {
      atomId = std::distance(speciesNames.begin(), itr);
    } else {
      Error e("Element not found in XML");
    }
    atomicSpecies(i) = atomId;

    // note: atomic positions are cartesians, in units of angstroms
    lineSplit = split(atom.child_value(), ' ');
    atomicPositions(i, 0) = std::stod(lineSplit[0]);
    atomicPositions(i, 1) = std::stod(lineSplit[1]);
    atomicPositions(i, 2) = std::stod(lineSplit[2]);
    i += 1;
  }

  // we read the unit cell

  Eigen::Matrix3d directUnitCell;
  Eigen::Vector3d thisVals;
  pugi::xml_node cell = atomicStructure.child("cell");
  lineSplit = split(cell.child_value("a1"), ' ');
  directUnitCell(0, 0) = std::stod(lineSplit[0]);
  directUnitCell(0, 1) = std::stod(lineSplit[1]);
  directUnitCell(0, 2) = std::stod(lineSplit[2]);
  lineSplit = split(cell.child_value("a2"), ' ');
  directUnitCell(1, 0) = std::stod(lineSplit[0]);
  directUnitCell(1, 1) = std::stod(lineSplit[1]);
  directUnitCell(1, 2) = std::stod(lineSplit[2]);
  lineSplit = split(cell.child_value("a3"), ' ');
  directUnitCell(2, 0) = std::stod(lineSplit[0]);
  directUnitCell(2, 1) = std::stod(lineSplit[1]);
  directUnitCell(2, 2) = std::stod(lineSplit[2]);

  // Now we parse the electronic structure

  pugi::xml_node bandStructureXML = output.child("band_structure");
  bool lsda = bandStructureXML.child("lsda").text().as_bool();
  bool noncolin = bandStructureXML.child("noncolin").text().as_bool();
  bool spinOrbit = bandStructureXML.child("spinorbit").text().as_bool();
  int numBands = bandStructureXML.child("nbnd").text().as_int();
  int numElectrons = bandStructureXML.child("nelec").text().as_int();
  double homo =
      bandStructureXML.child("highestOccupiedLevel").text().as_double();
  int numIrredPoints = bandStructureXML.child("nks").text().as_int();

  pugi::xml_node startingKPoints = bandStructureXML.child("starting_k_points");
  // this may or may not be present! if so, I get mesh and offset
  pugi::xml_node mp = startingKPoints.child("monkhorst_pack");
  if (mp) {
    Error e("Grid found in QE:XML, should have used full kpoints grid", 1);
  }

  // Initialize the crystal class

  long dimensionality = context.getDimensionality();
  Crystal crystal(directUnitCell, atomicPositions, atomicSpecies, speciesNames,
                  speciesMasses, dimensionality);
  //	std::unique_ptr<Crystal> crystal(new Crystal(directUnitCell,
  //			atomicPositions, atomicSpecies, speciesNames,
  // speciesMasses, 			dimensionality));

  // initialize reciprocal lattice cell
  // I need this to convert kpoints from cartesian to crystal coordinates

  pugi::xml_node basisSet = output.child("basis_set");
  pugi::xml_node recCell = basisSet.child("reciprocal_lattice");
  Eigen::Matrix3d bVectors;
  lineSplit = split(recCell.child_value("b1"), ' ');
  bVectors(0, 0) = std::stod(lineSplit[0]);
  bVectors(0, 1) = std::stod(lineSplit[1]);
  bVectors(0, 2) = std::stod(lineSplit[2]);
  lineSplit = split(recCell.child_value("b2"), ' ');
  bVectors(1, 0) = std::stod(lineSplit[0]);
  bVectors(1, 1) = std::stod(lineSplit[1]);
  bVectors(1, 2) = std::stod(lineSplit[2]);
  lineSplit = split(recCell.child_value("b3"), ' ');
  bVectors(2, 0) = std::stod(lineSplit[0]);
  bVectors(2, 1) = std::stod(lineSplit[1]);
  bVectors(2, 2) = std::stod(lineSplit[2]);

  // parse k-points and energies

  Eigen::Matrix<double, 3, Eigen::Dynamic> irredPoints(3, numIrredPoints);
  Eigen::VectorXd irredWeights(numIrredPoints);
  Eigen::MatrixXd irredEnergies(numIrredPoints, numBands);
  Eigen::MatrixXd irredOccupations(numIrredPoints, numBands);
  irredPoints.setZero();
  irredWeights.setZero();
  irredEnergies.setZero();
  irredOccupations.setZero();
  i = 0;
  Eigen::Vector3d p;
  for (pugi::xml_node kpoint : bandStructureXML.children("ks_energies")) {
    irredWeights(i) = kpoint.child("k_point").attribute("weight").as_double();
    lineSplit = split(kpoint.child_value("k_point"), ' ');

    // note:
    // k_cart = bVectors * k_cryst

    p(0) = std::stod(lineSplit[0]);
    p(1) = std::stod(lineSplit[1]);
    p(2) = std::stod(lineSplit[2]);

    // convert from cartesian to crystal coordinates
    p = bVectors.inverse() * p;
    irredPoints.col(i) = p;

    lineSplit = split(kpoint.child_value("eigenvalues"), ' ');
    for (int j = 0; j < numBands; j++) {
      irredEnergies(i, j) = std::stod(lineSplit[j]);
    }

    lineSplit = split(kpoint.child_value("occupations"), ' ');
    for (int j = 0; j < numBands; j++) {
      irredOccupations(i, j) = std::stod(lineSplit[j]);
    }

    i += 1;
  }

  // QE XML energies are in Hartree units. Must convert to rydbergs
  irredEnergies *= 2.;

  // Now we do postprocessing

  if (lsda || noncolin) {
    Error e("spin is not yet supported");
  }

  auto tup = Points::findMesh(irredPoints);
  auto mesh = std::get<0>(tup);
  auto offset = std::get<1>(tup);
  FullPoints coarsePoints(crystal, mesh, offset);

  bool withVelocities = false;
  bool withEigenvectors = false;
  Particle particle(Particle::electron);
  FullBandStructure coarseBandStructure(numBands, particle, withVelocities,
                                        withEigenvectors, coarsePoints);
  // fill in the info on band structure
  Eigen::Vector3d pointCoords;
  Eigen::VectorXd thisEnergies(numBands);
  for (int ik = 0; ik < numIrredPoints; ik++) {
    // note: kpoints in the XML files are not ordered in the same way
    // as in the scf.in file
    pointCoords = irredPoints.col(ik);
    thisEnergies = irredEnergies.row(ik);
    coarseBandStructure.setEnergies(pointCoords, thisEnergies);
  }

  context.setHasSpinOrbit(spinOrbit);
  if (spinOrbit)
    numElectrons /= 2.;
  context.setNumOccupiedStates(numElectrons);
  context.setFermiLevel(homo);

  ElectronH0Fourier electronH0(crystal, coarsePoints, coarseBandStructure,
                               fourierCutoff);
  //	std::unique_ptr<ElectronH0Fourier> electronH0(new ElectronH0Fourier(
  //			*crystal, coarsePoints, coarseBandStructure,
  // fourierCutoff));

  //	return std::make_tuple(std::move(crystal),std::move(electronH0));
  return {crystal, electronH0};
};

std::tuple<Crystal, ElectronH0Wannier>
QEParser::parseElHarmonicWannier(Context &context) {
  //  Here we read the XML file of quantum espresso.

  std::string fileName = context.getElectronH0Name();

  if (fileName == "") {
    Error e("Must provide the Wannier90 TB file name", 1);
  }

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open()) {
    Error e("Wannier H0 file not found", 1);
  }

  //  First line contains the title and date
  std::getline(infile, line);

  // Then, we have the directUnitCell of the ctystal in angstroms
  Eigen::Matrix3d directUnitCell(3, 3);
  directUnitCell.setZero();
  for (int i = 0; i < 3; i++) {
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    for (int j = 0; j < 3; j++) {
      // unit cell is written in angstrom
      directUnitCell(i, j) = std::stod(lineSplit[j]) / distanceBohrToAng;
    }
  };

  // Next, we the number of Wannier functions / bands, after disentanglement
  std::getline(infile, line);
  long numWann = std::stoi(line);

  // The number of irreducible vectors in real space
  std::getline(infile, line);
  long numVectors = std::stoi(line);

  // now, we must read numVectors integers with the vector degeneracies
  // there can be only up to 15 numbers per line
  long numLines = numVectors / long(15);
  if (double(numVectors) / 15. > 0.)
    numLines += 1;
  Eigen::VectorXd vectorsDegeneracies(numVectors);
  vectorsDegeneracies.setZero();
  long j = 0;
  for (long i = 0; i < numLines; i++) {
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    for (auto x : lineSplit) {
      long deg = std::stoi(x);
      vectorsDegeneracies(j) = double(deg);
      j += 1;
    }
  }

  // now we read the Hamiltonian in real space
  Eigen::MatrixXd bravaisVectors(3, numVectors);
  Eigen::Tensor<std::complex<double>, 3> h0R(numVectors, numWann, numWann);
  Eigen::Tensor<std::complex<double>, 4> rMatrix(3, numVectors, numWann,
                                                 numWann);
  bravaisVectors.setZero();
  h0R.setZero();
  rMatrix.setZero();

  // parse the Hamiltonian

  for (long iR = 0; iR < numVectors; iR++) {
    // first we have an empty line
    std::getline(infile, line);

    // then we read the lattice vector coordinates
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    bravaisVectors(0, iR) = std::stod(lineSplit[0]);
    bravaisVectors(1, iR) = std::stod(lineSplit[1]);
    bravaisVectors(2, iR) = std::stod(lineSplit[2]);

    for (long i = 0; i < numWann; i++) {
      for (long j = 0; j < numWann; j++) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        double re = std::stod(lineSplit[2]) / energyRyToEv;
        double im = std::stod(lineSplit[3]) / energyRyToEv;
        h0R(iR, i, j) = {re, im}; // the matrix was in eV
      }
    }
  }

  // now parse the R matrix
  // the format is similar, but we have a complex vector

  for (long iR = 0; iR < numVectors; iR++) {
    // first we have an empty line
    std::getline(infile, line);

    // then we read the lattice vector coordinates
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    // they have been initialized above, and they are the same

    for (long i = 0; i < numWann; i++) {
      for (long j = 0; j < numWann; j++) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        double re = std::stod(lineSplit[2]) / distanceBohrToAng;
        double im = std::stod(lineSplit[3]) / distanceBohrToAng;
        rMatrix(0, iR, i, j) = {re, im}; // the matrix was in eV
        re = std::stod(lineSplit[4]) / distanceBohrToAng;
        im = std::stod(lineSplit[5]) / distanceBohrToAng;
        rMatrix(1, iR, i, j) = {re, im}; // the matrix was in eV
        re = std::stod(lineSplit[6]) / distanceBohrToAng;
        im = std::stod(lineSplit[7]) / distanceBohrToAng;
        rMatrix(2, iR, i, j) = {re, im}; // the matrix was in eV
      }
    }
  }

  // I need to convert crystalVectors in cartesian coordinates
  // must check if I am aligning the unit cell correctly
  bravaisVectors = directUnitCell * bravaisVectors;
  // note: for Wannier90, lattice vectors are the rows of the matrix

  ElectronH0Wannier electronH0(directUnitCell, bravaisVectors,
                               vectorsDegeneracies, h0R, rMatrix);
  //	std::unique_ptr<ElectronH0Wannier> electronH0(new ElectronH0Wannier(
  //			directUnitCell, crystalVectors, vectorsDegeneracies,
  // h0R));

  long dimensionality = context.getDimensionality();
  Eigen::MatrixXd atomicPositions = context.getInputAtomicPositions();
  Eigen::VectorXi atomicSpecies = context.getInputAtomicSpecies();
  std::vector<std::string> speciesNames = context.getInputSpeciesNames();

  // we default the masses to the conventional ones here.
  Eigen::VectorXd speciesMasses(speciesNames.size());
  PeriodicTable periodicTable;
  long i = 0;
  for (auto speciesName : speciesNames) {
    speciesMasses[i] = periodicTable.getMass(speciesName);
    i += 1;
  }

  Crystal crystal(directUnitCell, atomicPositions, atomicSpecies, speciesNames,
                  speciesMasses, dimensionality);
  //	std::unique_ptr<Crystal> crystal(new Crystal(directUnitCell,
  //			atomicPositions, atomicSpecies, speciesNames,
  // speciesMasses, 			dimensionality));

  //	return std::make_tuple(std::move(crystal),std::move(electronH0));
  return {crystal, electronH0};
};
