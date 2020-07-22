#include <Eigen/Dense>
#include <Eigen/Core>
#include "exceptions.h"
#include "crystal.h"
#include "spglib.h"
#include "constants.h"
#include "utilities.h"

Eigen::Matrix3d Crystal::calcReciprocalCell(
    const Eigen::Matrix3d directUnitCell) {
  Eigen::Matrix3d reciprocalCell = twoPi
      * directUnitCell.inverse().transpose();
  return reciprocalCell;
}

void Crystal::setDirectUnitCell(Eigen::Matrix3d directUnitCell_) {
  directUnitCell = directUnitCell_;
  reciprocalUnitCell = calcReciprocalCell(directUnitCell);
}

const Eigen::Matrix3d &Crystal::getDirectUnitCell() {
  return directUnitCell;
}

const Eigen::Matrix3d &Crystal::getReciprocalUnitCell() {
  // note: reciprocalUnitCell is  in units of twoPi
  // i.e. must be multiplied by twoPi
  return reciprocalUnitCell;
}

double calcVolume(const Eigen::Matrix3d &directUnitCell) {
  Eigen::Vector3d a1 = directUnitCell.row(0);
  Eigen::Vector3d a2 = directUnitCell.row(1);
  Eigen::Vector3d a3 = directUnitCell.row(2);
  double volume;
  volume = abs(a1.dot((a2.cross(a3))));
  volume += abs(a2.dot((a3.cross(a1))));
  volume += abs(a3.dot((a1.cross(a2))));
  volume /= 3.;
  return volume;
}

const int &Crystal::getNumAtoms() {
  return numAtoms;
}

double Crystal::getVolumeUnitCell(long dimensionality) {
  double volume;
  if (dimensionality == 3) {
    volume = volumeUnitCell;
  } else if (dimensionality == 2) {
    volume = abs(
        directUnitCell(0, 0) * directUnitCell(1, 1)
            - directUnitCell(0, 1) * directUnitCell(1, 0));
  } else {
    volume = directUnitCell(2, 2);
  }
  return volume;
}

const Eigen::MatrixXd &Crystal::getAtomicPositions() {
  return atomicPositions;
}

const Eigen::VectorXi &Crystal::getAtomicSpecies() {
  return atomicSpecies;
}

const std::vector<std::string> &Crystal::getAtomicNames() {
  return atomicNames;
}

const Eigen::VectorXd &Crystal::getAtomicMasses() {
  return atomicMasses;
}

const std::vector<std::string> &Crystal::getSpeciesNames() {
  return speciesNames;
}

const Eigen::VectorXd &Crystal::getSpeciesMasses() {
  return speciesMasses;
}

const std::vector<SymmetryOperation> &Crystal::getSymmetryOperations() {
  return symmetryOperations;
}

const int &Crystal::getNumSymmetries() {
  return numSymmetries;
}

long Crystal::getDimensionality() {
  return dimensionality;
}

long Crystal::getNumSpecies() {
  return numSpecies;
}

Crystal::Crystal(Eigen::Matrix3d &directUnitCell_,
                 Eigen::MatrixXd &atomicPositions_,
                 Eigen::VectorXi &atomicSpecies_,
                 std::vector<std::string> &speciesNames_,
                 Eigen::VectorXd &speciesMasses_, long &dimensionality_) {

  setDirectUnitCell(directUnitCell_); // sets both direct and reciprocal
  volumeUnitCell = calcVolume(directUnitCell);

  if (volumeUnitCell <= 0.) {
    Error e("Unexpected non positive volume");
  }

  dimensionality = dimensionality_;

  if (atomicSpecies_.size() != atomicPositions_.rows()) {
    Error e("atomic species and positions are not aligned");
  }
  if (atomicPositions_.cols() != 3) {
    Error e("atomic positions need three coordinates");
  }
  if ((int) speciesMasses_.size() != (int) speciesNames_.size()) {
    Error e("species masses and names are not aligned");
  }

  atomicSpecies = atomicSpecies_;
  atomicPositions = atomicPositions_;
  speciesMasses = speciesMasses_;
  speciesNames = speciesNames_;

  numAtoms = atomicPositions.rows();
  numSpecies = atomicSpecies.size();

  Eigen::VectorXd atomicMasses_(numAtoms);
  std::vector<std::string> atomicNames_(numAtoms);

  for (int i = 0; i < numAtoms; i++) {
    atomicMasses_(i) = speciesMasses(atomicSpecies(i));
    atomicNames_[i] = speciesNames[atomicSpecies(i)];
  }
  atomicMasses = atomicMasses_;
  atomicNames = atomicNames_;

  // We now look for the symmetry operations of the crystal
  // in this implementation, we rely on spglib

  // Declare and allocate c-style arrays for spglib calls
  double latticeSPG[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // note: directUnitCell has lattice vectors along rows (a1 = cell.row(0))
      latticeSPG[i][j] = directUnitCell(i, j);
    }
  }

  // note: spglib wants fractional positions
  double (*positionSPG)[3];
  allocate(positionSPG, numAtoms);
  Eigen::Vector3d positionCrystal;
  Eigen::Vector3d positionCartesian;
  for (int i = 0; i < numAtoms; i++) {
    positionCartesian = atomicPositions.row(i);
    positionCrystal = directUnitCell.inverse() * positionCartesian;
    for (int j = 0; j < 3; j++) {
      positionSPG[i][j] = positionCrystal(j);
    }
  }

  // also wants integer types >= 1
  int *typesSPG;
  allocate(typesSPG, numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    typesSPG[i] = atomicSpecies(i) + 1;
  }

  int maxSize = 50;
  int rotations[maxSize][3][3];
  double translations[maxSize][3];
  double symprec = 1e-5;
  numSymmetries = spg_get_symmetry(rotations,
                                   translations,
                                   maxSize,
                                   latticeSPG,
                                   positionSPG,
                                   typesSPG,
                                   numAtoms,
                                   symprec);

  // need to explicitly deallocate allocated arrays.
  delete[] typesSPG;
  delete[] positionSPG;

  if (numSymmetries == 0) {
    Error e("SPGlib failed at recognizing symmetries");
  }

  // store the symmetries inside the class
  // note: spglib returns rotation and translation in fractional coordinates

  for (int isym = 0; isym < numSymmetries; isym++) {
    Eigen::Vector3d thisTranslation;
    thisTranslation(0) = translations[isym][0];
    thisTranslation(1) = translations[isym][1];
    thisTranslation(2) = translations[isym][2];
    Eigen::Matrix3d thisMatrix;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        thisMatrix(i, j) = rotations[isym][i][j];
      }
    }

    SymmetryOperation s = {thisMatrix, thisTranslation};
    symmetryOperations.push_back(s);
  }

  // this is to remember to modify things if I break this symmetry
  // got to modify here and below
  Warning w("Using time reversal symmetry");
  numSymmetries *= 2;
  for (int isym = numSymmetries/2; isym < numSymmetries; isym++) {
    Eigen::Vector3d thisTranslation;
    thisTranslation(0) = -translations[isym-numSymmetries/2][0];
    thisTranslation(1) = -translations[isym-numSymmetries/2][1];
    thisTranslation(2) = -translations[isym-numSymmetries/2][2];
    Eigen::Matrix3d thisMatrix;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        thisMatrix(i, j) = -rotations[isym-numSymmetries/2][i][j];
      }
    }
    SymmetryOperation s = {thisMatrix, thisTranslation};
    symmetryOperations.push_back(s);
  }

}

// empty constructor
Crystal::Crystal() {
  volumeUnitCell = 0.;
  numAtoms = 0;
  numSpecies = 0;
  dimensionality = 0.;
  numSymmetries = 0;
}

// copy constructor
Crystal::Crystal(const Crystal &obj) {
  directUnitCell = obj.directUnitCell;
  reciprocalUnitCell = obj.reciprocalUnitCell;
  volumeUnitCell = obj.volumeUnitCell;
  numAtoms = obj.numAtoms;
  numSpecies = obj.numSpecies;
  dimensionality = obj.dimensionality;
  atomicPositions = obj.atomicPositions;
  atomicSpecies = obj.atomicSpecies;
  atomicNames = obj.atomicNames;
  atomicMasses = obj.atomicMasses;
  speciesNames = obj.speciesNames;
  speciesMasses = obj.speciesMasses;
  symmetryOperations = obj.symmetryOperations;
  numSymmetries = obj.numSymmetries;
}

// assignment operator
Crystal &Crystal::operator=(const Crystal &obj) {
  if (this != &obj) {
    directUnitCell = obj.directUnitCell;
    reciprocalUnitCell = obj.reciprocalUnitCell;
    volumeUnitCell = obj.volumeUnitCell;
    numAtoms = obj.numAtoms;
    numSpecies = obj.numSpecies;
    dimensionality = obj.dimensionality;
    atomicPositions = obj.atomicPositions;
    atomicSpecies = obj.atomicSpecies;
    atomicNames = obj.atomicNames;
    atomicMasses = obj.atomicMasses;
    speciesNames = obj.speciesNames;
    speciesMasses = obj.speciesMasses;
    symmetryOperations = obj.symmetryOperations;
    numSymmetries = obj.numSymmetries;
  }
  return *this;
}

