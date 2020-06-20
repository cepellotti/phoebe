#include "context.h"
#include "constants.h"
#include "exceptions.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// TODO: it would be nice to have a smoother handling of read errors, with
// some informations provided to the user.
// Also, we must provide default values if key is not found in input

struct ParameterNotFound : public std::exception {
  const char *what() const throw() { return "Input parameter not found"; }
};

bool lineHasPattern(std::string line, std::string pattern) {
  bool bx = false;

  std::string sep;
  std::string s;
  std::string substr2;
  size_t pos;

  //	If line is empty, there's nothing for sure. Return false
  if (line.empty()) {
    return bx;
  }

  sep = "=";
  pos = line.find(sep);
  if (std::string::npos == pos) {
    throw ParameterNotFound();
  }

  s = line.substr(0, pos);
  s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

  if (s == pattern) {
    bx = true;
  }
  return bx;
}

bool lineHasUnits(std::string target, std::string pattern) {
  bool hasIt = false;
  if (target.find(pattern) != std::string::npos) {
    hasIt = true;
  }
  return hasIt;
}

bool parseBool(std::vector<std::string> lines, std::string pattern) {
  bool x;
  bool found = false;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {
      found = true;
      std::string delimeter = "=";
      size_t pos = line.find(delimeter);
      std::string s = line.substr(pos + 1);

      std::transform(s.begin(), s.end(), s.begin(), ::tolower);
      s.erase(std::remove_if(s.begin(), s.end(),
                             [](char c) { return !isalpha(c); }),
              s.end());

      if (s == "false") {
        x = false;
      } else if (s == "true") {
        x = true;
      } else if (s == "0") {
        x = false;
      } else if (s == "1") {
        x = true;
      } else {
        throw "Couldn't fix boolean value";
      }
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

double parseDouble(std::vector<std::string> lines, std::string pattern) {
  double x = 0.;
  bool found = false;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {
      std::string delimeter = "=";
      size_t pos = line.find(delimeter);
      std::string value = line.substr(pos + 1);
      x = std::stod(value); // convert to double
      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

double parseDoubleWithUnits(std::vector<std::string> lines,
                            std::string pattern) {
  double x = 0.;
  bool found = false;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {
      std::string delimeter = "=";
      size_t pos = line.find(delimeter);
      std::string value = line.substr(pos + 1);
      x = std::stod(value); // convert to double

      // now check the units and convert
      if (lineHasUnits(line, "eV")) {
        x /= energyRyToEv;
      }
      if (lineHasUnits(line, "cmm1")) {
        x /= ryToCmm1;
      }
      if (lineHasUnits(line, "ps")) {
        x /= timeRyToFs * 1.0e-3;
      }
      if (lineHasUnits(line, "fs")) {
        x /= timeRyToFs;
      }
      if (lineHasUnits(line, "mum")) {
        x /= distanceBohrToMum;
      }
      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

std::vector<double> parseDoubleList(std::vector<std::string> lines,
                                    std::string pattern) {
  std::vector<double> x;
  double xTemp;
  bool found = false;
  std::string token, s;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {

      std::string delimeter;
      size_t pos1;
      size_t pos2;

      delimeter = "[";
      pos1 = line.find_first_of(delimeter);
      delimeter = "]";
      pos2 = line.find_last_of(delimeter);

      if (pos1 == std::string::npos) {
        throw "Error in parseDoubleList";
      }
      if (pos2 == std::string::npos) {
        throw "Error in parseDoubleList";
      }

      s = line.substr(pos1 + 1, pos2 - pos1 - 1);
      delimeter = ",";

      while ((pos1 = s.find(delimeter)) != std::string::npos) {
        token = s.substr(0, pos1);

        xTemp = std::stod(token); // convert to integer
        x.push_back(xTemp);

        s.erase(0, pos1 + delimeter.length());
      }
      //			Must not forget the last element in the list
      xTemp = std::stod(s);
      x.push_back(xTemp);

      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

long parseLong(std::vector<std::string> lines, std::string pattern) {
  long x = 0;
  bool found = false;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {
      std::string delimeter = "=";
      size_t pos = line.find(delimeter);
      std::string value = line.substr(pos + 1);
      x = std::stoi(value); // convert to integer
      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

std::vector<long> parseLongList(std::vector<std::string> lines,
                                std::string pattern) {
  std::vector<long> x;
  long xTemp;
  bool found = false;
  std::string token, s;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {

      std::string delimeter;
      size_t pos1;
      size_t pos2;

      delimeter = "[";
      pos1 = line.find_first_of(delimeter);
      delimeter = "]";
      pos2 = line.find_last_of(delimeter);

      if (pos1 == std::string::npos) {
        throw "Error in parseLongList";
      }
      if (pos2 == std::string::npos) {
        throw "Error in parseLongList";
      }

      s = line.substr(pos1 + 1, pos2 - pos1 - 1);
      delimeter = ",";

      while ((pos1 = s.find(delimeter)) != std::string::npos) {
        token = s.substr(0, pos1);

        xTemp = std::stoi(token); // convert to integer
        x.push_back(xTemp);

        s.erase(0, pos1 + delimeter.length());
      }
      //			Must not forget the last element in the list
      xTemp = std::stoi(s);
      x.push_back(xTemp);

      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

std::string parseString(std::vector<std::string> lines, std::string pattern) {
  std::string x = "";
  bool found = false;

  for (std::string line : lines) {

    if (lineHasPattern(line, pattern)) {
      std::string delimeter;
      size_t pos1;
      size_t pos2;

      delimeter = "'";
      pos1 = line.find_first_of(delimeter);
      pos2 = line.find_last_of(delimeter);

      if (pos1 == std::string::npos) {

        delimeter = "\"";
        pos1 = line.find_first_of(delimeter);
        pos2 = line.find_last_of(delimeter);
        if (pos1 == std::string::npos) {
          throw "Couldn't solve string parsing of "
                "pattern: " +
              pattern;
        }
      }

      if (pos1 == pos2) {
        throw "Error parsing string from user input. "
              "Pattern " +
            pattern;
      }
      x = line.substr(pos1 + 1, pos2 - pos1 - 1);
      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

std::vector<std::string> parseStringList(std::vector<std::string> lines,
                                         std::string pattern) {
  std::vector<std::string> x;
  std::string xTemp;
  bool found = false;
  std::string token, s;

  for (std::string line : lines) {
    if (lineHasPattern(line, pattern)) {

      // remove empty spaces
      line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

      std::string delimeter;
      size_t pos1;
      size_t pos2;

      delimeter = "[";
      pos1 = line.find_first_of(delimeter);
      delimeter = "]";
      pos2 = line.find_last_of(delimeter);

      if (pos1 == std::string::npos) {
        throw "Error in parseDoubleList";
      }
      if (pos2 == std::string::npos) {
        throw "Error in parseDoubleList";
      }

      s = line.substr(pos1 + 1, pos2 - pos1 - 1);
      delimeter = ",";

      while ((pos1 = s.find(delimeter)) != std::string::npos) {
        token = s.substr(0, pos1);

        // we also remove the " symbols
        token.erase(std::remove(token.begin(), token.end(), '"'), token.end());

        xTemp = token;
        x.push_back(xTemp);

        s.erase(0, pos1 + delimeter.length());
      }
      // Must not forget the last element in the list
      xTemp = s;
      xTemp.erase(std::remove(xTemp.begin(), xTemp.end(), '"'), xTemp.end());
      x.push_back(xTemp);

      found = true;
      break;
    }
  }
  if (not found) {
    throw ParameterNotFound();
  }
  return x;
};

bool patternInString(const std::string &s, const std::string &pattern) {
  if (s.find(pattern) != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXi, std::vector<std::string>>
parseCrystal(std::vector<std::string> &lines) {

  long iStart = -1;
  long iEnd = -1;
  long counter = 0;
  for (auto line : lines) {
    if (patternInString(line, "begin crystal")) {
      iStart = counter;
    }
    if (patternInString(line, "end crystal")) {
      iEnd = counter;
      break;
    }
    counter += 1;
  }
  if ((iStart == -1) || (iEnd == -1)) {
    throw ParameterNotFound();
  }

  long numAtoms = iEnd - iStart - 1;
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  std::vector<std::string> speciesNames;

  for (counter = iStart + 1; counter < iEnd; counter++) {
    std::string line = lines[counter];

    // split line by spaces
    std::stringstream ss(line);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> splitLine(begin, end);

    std::string thisElement = splitLine[0];
    // if new element, insert into list of species
    if (std::find(speciesNames.begin(), speciesNames.end(), thisElement) ==
        speciesNames.end()) {
      // thisElement not in speciesNames, add it
      speciesNames.push_back(thisElement);
    }
    // find the index of the current element
    long index = 0;
    for (auto speciesName : speciesNames) {
      if (speciesName == thisElement) {
        break;
      }
      index += 1;
    }
    // save species and positions
    atomicSpecies(counter - iStart - 1) = index;
    atomicPositions(counter - iStart - 1, 0) = std::stod(splitLine[1]);
    atomicPositions(counter - iStart - 1, 1) = std::stod(splitLine[2]);
    atomicPositions(counter - iStart - 1, 2) = std::stod(splitLine[3]);
  }
  atomicPositions /= distanceBohrToAng;
  return {atomicPositions, atomicSpecies, speciesNames};
}

Eigen::Tensor<double, 3> parsePathExtrema(std::vector<std::string> &lines) {
  long iStart = -1;
  long iEnd = -1;
  long counter = 0;
  for (auto line : lines) {
    if (patternInString(line, "begin point path")) {
      iStart = counter;
    }
    if (patternInString(line, "end point path")) {
      iEnd = counter;
      break;
    }
    counter += 1;
  }
  if ((iStart == -1) || (iEnd == -1)) {
    throw ParameterNotFound();
  }

  long numSegments = iEnd - iStart - 1;
  Eigen::Tensor<double, 3> pathExtrema(numSegments, 2, 3);
  pathExtrema.setZero();

  long i = 0;
  for (counter = iStart + 1; counter < iEnd; counter++) {
    std::string line = lines[counter];

    // split line by spaces
    std::stringstream ss(line);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> splitLine(begin, end);

    pathExtrema(i, 0, 0) = std::stod(splitLine[1]);
    pathExtrema(i, 0, 1) = std::stod(splitLine[2]);
    pathExtrema(i, 0, 2) = std::stod(splitLine[3]);

    pathExtrema(i, 1, 0) = std::stod(splitLine[5]);
    pathExtrema(i, 1, 1) = std::stod(splitLine[6]);
    pathExtrema(i, 1, 2) = std::stod(splitLine[7]);

    i++;
  }

  return pathExtrema;
}

void Context::setupFromInput(std::string fileName) {
  std::vector<std::string> lines;
  std::string line;

  // open input file and read content
  std::ifstream infile(fileName);
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }
  infile.close();

  try {
    std::string tmp = parseString(lines, "phD2FileName");
    setPhD2FileName(tmp);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::string tmp = parseString(lines, "phD3FileName");
    setPhD3FileName(tmp);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    setSumRuleD2(parseString(lines, "sumRuleD2"));
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::string tmp = parseString(lines, "electronH0Name");
    setElectronH0Name(tmp);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double dval = parseDouble(lines, "electronFourierCutoff");
    setElectronFourierCutoff(dval);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<long> vecMesh = parseLongList(lines, "qMesh");
    Eigen::Vector3i qMesh_;
    qMesh_(0) = vecMesh[0];
    qMesh_(1) = vecMesh[1];
    qMesh_(2) = vecMesh[2];
    setQMesh(qMesh_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<long> vecMesh = parseLongList(lines, "kMesh");
    Eigen::Vector3i kMesh_;
    kMesh_(0) = vecMesh[0];
    kMesh_(1) = vecMesh[1];
    kMesh_(2) = vecMesh[2];
    setKMesh(kMesh_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::string win = parseString(lines, "windowType");
    setWindowType(win);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<double> winLim = parseDoubleList(lines, "windowEnergyLimit");
    Eigen::Vector2d winLim_;
    // we just make sure to order it
    winLim_[0] = std::min(winLim[0], winLim[1]);
    winLim_[1] = std::max(winLim[0], winLim[1]);
    setWindowEnergyLimit(winLim_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDouble(lines, "populationLimit");
    setWindowPopulationLimit(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<double> x = parseDoubleList(lines, "chemicalPotentials");
    Eigen::VectorXd x_(x.size());
    for (long unsigned i = 0; i < x.size(); i++) {
      x_(i) = x[i] / energyRyToEv;
    }
    setChemicalPotentials(x_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<double> x = parseDoubleList(lines, "dopings");
    Eigen::VectorXd x_(x.size());
    for (long unsigned i = 0; i < x.size(); i++) {
      x_(i) = x[i];
    }
    setDopings(x_); // note: we store it as provided in input. in cm^-D
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<double> x = parseDoubleList(lines, "temperatures");
    Eigen::VectorXd x_(x.size());
    for (long unsigned i = 0; i < x.size(); i++) {
      x_(i) = x[i] / temperatureAuToSi;
    }
    setTemperatures(x_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::string x = parseString(lines, "appName");
    setAppName(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<std::string> x = parseStringList(lines, "solverBTE");
    setSolverBTE(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDouble(lines, "convergenceThresholdBTE");
    setConvergenceThresholdBTE(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    long x = parseLong(lines, "maxIterationsBTE");
    setMaxIterationsBTE(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    long x = parseLong(lines, "dimensionality");
    setDimensionality(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "dosMinEnergy");
    setDosMinEnergy(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "dosMaxEnergy");
    setDosMaxEnergy(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "dosDeltaEnergy");
    setDosDeltaEnergy(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    // note: these should be given in input in cartesian coordinates
    auto tup = parseCrystal(lines);
    // auto [atomicPositions, atomicSpecies, speciesNames] =
    // parseCrystal(lines);
    auto atomicPositions = std::get<0>(tup);
    auto atomicSpecies = std::get<1>(tup);
    auto speciesNames = std::get<2>(tup);
    setInputAtomicPositions(atomicPositions);
    setInputAtomicSpecies(atomicSpecies);
    setInputSpeciesNames(speciesNames);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    // note: these should be given in input in cartesian coordinates
    auto pathExtrema_ = parsePathExtrema(lines);
    setPathExtrema(pathExtrema_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDouble(lines, "deltaPath");
    setDeltaPath(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDouble(lines, "fermiLevel"); // in eV
    setFermiLevel(x / energyRyToEv);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseBool(lines, "hasSpinOrbit");
    setHasSpinOrbit(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    // note: numOccupiedStates refers to the number of states that are
    // occupied
    // for Wannier: the number of Wannier states that are full
    // for Fourier: the number of occupied bands
    // remember to NOT count the spin degeneracy
    double x = parseDouble(lines, "numOccupiedStates");
    if (!hasSpinOrbit)
      x *= 2;
    setNumOccupiedStates(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::string x_ = parseString(lines, "smearingMethod");
    int x;
    if (x_ == "gaussian") {
      x = 0;
    } else if (x_ == "adaptiveGaussian") {
      x = 1;
    } else if (x_ == "tetrahedron") {
      x = 2;
    } else {
      x = -1;
    }
    setSmearingMethod(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "smearingWidth");
    setSmearingWidth(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "constantRelaxationTime");
    setConstantRelaxationTime(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    bool x = parseBool(lines, "scatteringMatrixInMemory");
    setScatteringMatrixInMemory(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    bool x = parseBool(lines, "withIsotopeScattering");
    setWithIsotopeScattering(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  try {
    std::vector<double> x = parseDoubleList(lines, "massVariance");
    Eigen::VectorXd x_(x.size());
    for (long unsigned i = 0; i < x.size(); i++) {
      x_(i) = x[i];
    }
    setMassVariance(x_);
  } catch (ParameterNotFound &e) {
  } // Do nothing!

  //	try {
  //		bool x = parseBool(lines, "withRTABoundaryScattering");
  //		setWithRTABoundaryScattering(x);
  //	} catch (ParameterNotFound& e) {} // Do nothing!

  try {
    double x = parseDoubleWithUnits(lines, "boundaryLength");
    setBoundaryLength(x);
  } catch (ParameterNotFound &e) {
  } // Do nothing!
};

void Context::setPhD2FileName(std::string x) { phD2FileName = x; }

std::string Context::getPhD2FileName() { return phD2FileName; }

void Context::setPhD3FileName(std::string x) { phD3FileName = x; }

std::string Context::getPhD3FileName() { return phD3FileName; }

void Context::setSumRuleD2(std::string x) { sumRuleD2 = x; }

std::string Context::getSumRuleD2() { return sumRuleD2; }

void Context::setElectronH0Name(std::string x) { electronH0Name = x; }

std::string Context::getElectronH0Name() {
  if (electronH0Name == "") {
    Error e("Electronic H0 filename not set", 1);
  }
  return electronH0Name;
}

void Context::setElectronFourierCutoff(const double &x) {
  electronFourierCutoff = x;
}

double &Context::getElectronFourierCutoff() {
  if (electronFourierCutoff == 0.) {
    Error e("Electronic Fourier Cutoff not set", 1);
  }
  return electronFourierCutoff;
}

void Context::setAppName(const std::string &x) { appName = x; }

std::string Context::getAppName() { return appName; }

void Context::setQMesh(const Eigen::Vector3i &x) { qMesh = x; }

Eigen::Vector3i Context::getQMesh() { return qMesh; }

void Context::setKMesh(const Eigen::Vector3i &x) { kMesh = x; }

Eigen::Vector3i Context::getKMesh() { return kMesh; }

void Context::setWindowType(std::string x) { windowType = x; }

std::string Context::getWindowType() { return windowType; }

void Context::setWindowEnergyLimit(Eigen::Vector2d x) { windowEnergyLimit = x; }

Eigen::Vector2d Context::getWindowEnergyLimit() { return windowEnergyLimit; }

void Context::setWindowPopulationLimit(double x) { windowPopulationLimit = x; }

double Context::getWindowPopulationLimit() { return windowPopulationLimit; }

void Context::setChemicalPotentials(Eigen::VectorXd x) {
  chemicalPotentials = x;
}

Eigen::VectorXd Context::getChemicalPotentials() { return chemicalPotentials; }

void Context::setDopings(Eigen::VectorXd x) { dopings = x; }

Eigen::VectorXd Context::getDopings() { return dopings; }

void Context::setTemperatures(Eigen::VectorXd x) { temperatures = x; }

Eigen::VectorXd Context::getTemperatures() { return temperatures; }

void Context::setSolverBTE(std::vector<std::string> x) { solverBTE = x; }

std::vector<std::string> Context::getSolverBTE() { return solverBTE; }

void Context::setConvergenceThresholdBTE(double x) {
  convergenceThresholdBTE = x;
}

double Context::getConvergenceThresholdBTE() { return convergenceThresholdBTE; }

void Context::setMaxIterationsBTE(long x) { maxIterationsBTE = x; }

long Context::getMaxIterationsBTE() { return maxIterationsBTE; }

void Context::setDimensionality(long x) { dimensionality = x; }

long Context::getDimensionality() { return dimensionality; }

void Context::setDosMinEnergy(double x) { dosMinEnergy = x; }

double Context::getDosMinEnergy() { return dosMinEnergy; }

void Context::setDosMaxEnergy(double x) { dosMaxEnergy = x; }

double Context::getDosMaxEnergy() { return dosMaxEnergy; }

void Context::setDosDeltaEnergy(double x) { dosDeltaEnergy = x; }

double Context::getDosDeltaEnergy() { return dosDeltaEnergy; }

Eigen::MatrixXd Context::getInputAtomicPositions() {
  return inputAtomicPositions;
}
Eigen::VectorXi Context::getInputAtomicSpecies() { return inputAtomicSpecies; }
std::vector<std::string> Context::getInputSpeciesNames() {
  return inputSpeciesNames;
}
void Context::setInputAtomicPositions(Eigen::MatrixXd &atomicPositions) {
  inputAtomicPositions = atomicPositions;
}
void Context::setInputAtomicSpecies(Eigen::VectorXi &atomicSpecies) {
  inputAtomicSpecies = atomicSpecies;
}
void Context::setInputSpeciesNames(std::vector<std::string> &speciesNames) {
  inputSpeciesNames = speciesNames;
}

void Context::setPathExtrema(Eigen::Tensor<double, 3> x) { pathExtrema = x; }
Eigen::Tensor<double, 3> Context::getPathExtrema() { return pathExtrema; }

void Context::setDeltaPath(double x) { deltaPath = x; }
double Context::getDeltaPath() { return deltaPath; }

void Context::setFermiLevel(const double &x) { fermiLevel = x; }
double Context::getFermiLevel() { return fermiLevel; }
void Context::setNumOccupiedStates(double x) { numOccupiedStates = x; }
double Context::getNumOccupiedStates() { return numOccupiedStates; }
void Context::setHasSpinOrbit(bool x) { hasSpinOrbit = x; }
bool Context::getHasSpinOrbit() { return hasSpinOrbit; }

void Context::setSmearingMethod(const int &x) { smearingMethod = x; }

int Context::getSmearingMethod() { return smearingMethod; }

void Context::setSmearingWidth(const double &x) { smearingWidth = x; }

double Context::getSmearingWidth() { return smearingWidth; }

void Context::setConstantRelaxationTime(const double &x) {
  constantRelaxationTime = x;
}

double Context::getConstantRelaxationTime() { return constantRelaxationTime; }

void Context::setScatteringMatrixInMemory(const bool &x) {
  scatteringMatrixInMemory = x;
}

bool Context::getScatteringMatrixInMemory() { return scatteringMatrixInMemory; }

void Context::setMassVariance(const Eigen::VectorXd &x) { massVariance = x; }

Eigen::VectorXd Context::getMassVariance() { return massVariance; }

void Context::setWithIsotopeScattering(const bool &x) {
  withIsotopeScattering = x;
}

bool Context::getWithIsotopeScattering() { return withIsotopeScattering; }

void Context::setBoundaryLength(const double &x) { boundaryLength = x; }

double Context::getBoundaryLength() { return boundaryLength; }

// void Context::setWithRTABoundaryScattering(const bool & x) {
//	doIsotopes = x;
//}
//
// bool Context::getWithRTABoundaryScattering() {
//	return doIsotopes;
//}
