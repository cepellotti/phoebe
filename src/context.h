#ifndef CONTEXT_H
#define CONTEXT_H

#include <string>
#include <vector>
#include "eigen.h"
#include <limits> // NaN

using namespace std;

class Context {
private:
	std::string phD2FileName = "";
	std::string phD3FileName = "";
	std::string electronH0Name = "";
	std::string appName = "";
	std::string sumRuleD2 = "";
	int smearingMethod = -1;
	double smearingWidth = 0.;
	Eigen::VectorXd temperatures;
	std::vector<std::string> solverBTE = {"RTA"};
	double convergenceThresholdBTE = 1e-5;
	long maxIterationsBTE = 50;

	bool scatteringMatrixInMemory = true;

	std::string windowType = "nothing";
	Eigen::Vector2d windowEnergyLimit = Eigen::Vector2d::Zero();
	double windowPopulationLimit;

	Eigen::VectorXd dopings;
	Eigen::VectorXd chemicalPotentials = Eigen::VectorXd::Zero(1);
	double electronFourierCutoff = 0.;

	Eigen::Vector3i qMesh = Eigen::Vector3i::Zero();
	Eigen::Vector3i kMesh = Eigen::Vector3i::Zero();


	double fermiLevel = std::numeric_limits<double>::quiet_NaN();
	double numOccupiedStates = std::numeric_limits<double>::quiet_NaN();
	bool hasSpinOrbit = false;

	long dimensionality = 3;

	double dosMinEnergy = 0.;
	double dosMaxEnergy = 1.;
	double dosDeltaEnergy = 0.01;

	Eigen::MatrixXd inputAtomicPositions;
	Eigen::VectorXi inputAtomicSpecies;
	std::vector<std::string> inputSpeciesNames;

	Eigen::Tensor<double,3> pathExtrema;
	double deltaPath = 0.05;

	double constantRelaxationTime = 0.;

//  Setter and getter for all the variables above
public:
	/** sets the name of the file containing the lattice force constants.
	 * For Quantum Espresso, this is the path to the file produced by q2r.
	 * @param x: the file path.
	 */
	void setPhD2FileName(std::string x);
	/** gets the name of the file containing the lattice force constants.
	 * For Quantum Espresso, this is the path to the file produced by q2r.
	 * @return x: the file path.
	 */
	std::string getPhD2FileName();

	void setPhD3FileName(std::string x);
	std::string getPhD3FileName();

//	void setScratchFolder(std::string x);
//	std::string getScratchFolder();

	/** sets the name of the file containing the electronic band structure.
	 * For Quantum Espresso, this is the path to the XML file.
	 * @param x: the file path.
	 */
	void setElectronH0Name(std::string path);
	/** gets the name of the file containing the electronic band structure.
	 * For Quantum Espresso, this is the path to the XML file.
	 * @return path: the file path.
	 */
	std::string getElectronH0Name();

	/** sets the value of the cutoff to be used for the Fourier interpolation
	 * of the band structure.
	 * @param r: the cutoff value. r is the radius up to which all lattice
	 * vectors (in real space) are used for the interpolation.
	 */
	void setElectronFourierCutoff(const double & x);
	/** gets the value of the cutoff to be used for the Fourier interpolation
	 * of the band structure.
	 * @return r: the cutoff value.
	 */
	double& getElectronFourierCutoff();

//	void setElPhFileName(std::string x);
//	std::string getElPhFileName();

	/** sets the calculation type to be run.
	 * @param x: the name of the calculation, e.g. "electron-phonon",
	 * "phonon-phonon", or "coupled-electron-phonon".
	 */
	void setAppName(const std::string & x);
	/** gets the type of calculation to be run.
	 * @return x: the name of the calculation, e.g. "electron-phonon" or
	 * "phonon-phonon".
	 */
	std::string getAppName();

	/** sets the sum rule to be imposed on the lattice force constants.
	 * @param x: the name of the sum rule, i.e. "simple" or "crystal".
	 * where simple is a rescaling of diagonal elements, and crystal finds the
	 * closest matrix compliant with the translational rules
	 */
	void setSumRuleD2(std::string x);
	/** gets the sum rule to be imposed on the lattice force constants.
	 * @return x: the name of the sum rule, i.e. "simple" or "crystal".
	 */
	std::string getSumRuleD2();

	/** sets the fine mesh of phonon properties.
	 * @param x: a vector of 3 integers representing the mesh of points.
	 */
	void setQMesh(const Eigen::Vector3i & x);
	/** gets the mesh of points for harmonic phonon properties.
	 * @return path: an array with 3 integers representing the q-point mesh.
	 */
	Eigen::Vector3i getQMesh();

	/** sets the fine mesh of electronic properties.
	 * @param x: a vector of 3 integers representing the mesh of k-points.
	 */
	void setKMesh(const Eigen::Vector3i & x);
	/** gets the mesh of points for harmonic electronic properties.
	 * @return path: an array with 3 integers representing the k-point mesh.
	 */
	Eigen::Vector3i getKMesh();

	/** sets the Window type to be used to filter out states that don't
	 * contribute to transport.
	 * @param x: a string, which can take values "none", "energy", "population"
	 */
	void setWindowType(std::string windowType);
	/** gets the Window type to be used to filter out states that don't
	 * contribute to transport.
	 * @return path: an array with 3 integers representing the k-point mesh.
	 * @param windowType: a string, which can take values "none", "energy",
	 * or "population"
	 */
	std::string getWindowType();

	/** sets the values of energy limits to be used with a window on energies.
	 * @param x: a vector of 2 doubles representing the minimum and maximum
	 * values of energies that will be used
	 */
	void setWindowEnergyLimit(Eigen::Vector2d windowEnergyLimit);
	/** gets the values of energy limits to be used with a window on energies.
	 * @return x: a vector of 2 doubles representing the minimum and maximum
	 * values of energies that will be used
	 */
	Eigen::Vector2d getWindowEnergyLimit();

	/** sets the value of population above which a state is considered active.
	 * i.e. the state will be used if its occupation number deviates from 0 or
	 * 1 by at least this amount.
	 * @param x: the <double> value of the population threshold.
	 */
	void setWindowPopulationLimit(double windowPopulationLimit);
	/** gets the value of population above which a state is considered active.
	 * i.e. the state will be used if its occupation number deviates from 0 or
	 * 1 by at least this amount.
	 * @return x: the <double> value of the population threshold.
	 */
	double getWindowPopulationLimit();

	/** sets the value of chemical potentials to be used in the calculation
	 * of transport properties. Values in rydbergs.
	 * @param x: the <double> values of the chemical potentials.
	 */
	void setChemicalPotentials(Eigen::VectorXd x);
	/** gets the value of chemical potentials (in Rydbergs) to be used in the
	 * calculation of transport properties
	 * @return x: the vector of values for chemical potentials
	 */
	Eigen::VectorXd getChemicalPotentials();

	/** sets the value of chemical potentials to be used in the calculation
	 * of transport properties. Values in rydbergs.
	 * @param x: the <double> values of the chemical potentials.
	 */
	void setDopings(Eigen::VectorXd x);
	/** gets the value of chemical potentials (in Rydbergs) to be used in the
	 * calculation of transport properties
	 * @return x: the vector of values for chemical potentials
	 */
	Eigen::VectorXd getDopings();

	/** sets the value of temperatures to be used in the calculation
	 * of transport properties. Values in rydbergs.
	 * @param x: the <double> values of the temperatures.
	 */
	void setTemperatures(Eigen::VectorXd x);
	/** gets the value of temperatures (in Rydbergs) to be used in the
	 * calculation of transport properties
	 * @return x: the vector of values for temperatures
	 */
	Eigen::VectorXd getTemperatures();

	void setSolverBTE(std::vector<std::string> solverBTE);
	std::vector<std::string> getSolverBTE();

	void setConvergenceThresholdBTE(double convergenceThresholdBTE);
	double getConvergenceThresholdBTE();

	void setMaxIterationsBTE(long maxIterationsBTE);
	long getMaxIterationsBTE();

	void setDimensionality(long dimensionality);
	long getDimensionality();

	void setDosMinEnergy(double x);
	double getDosMinEnergy();

	void setDosMaxEnergy(double x);
	double getDosMaxEnergy();

	void setDosDeltaEnergy(double x);
	double getDosDeltaEnergy();

	// Wannier90 output doesn't contain the crystal information.
	// the user must then supplement it in the input
	// at least, if there's no phonon run
	// we may change the behavior in the future, parsing another file
	Eigen::MatrixXd getInputAtomicPositions();
	Eigen::VectorXi getInputAtomicSpecies();
	std::vector<std::string> getInputSpeciesNames();
	void setInputAtomicPositions(Eigen::MatrixXd & atomicPositions);
	void setInputAtomicSpecies(Eigen::VectorXi & atomicSpecies);
	void setInputSpeciesNames(std::vector<std::string> & speciesNames);


	void setPathExtrema(Eigen::Tensor<double,3> x);
	Eigen::Tensor<double,3> getPathExtrema();
	void setDeltaPath(double x);
	double getDeltaPath();

	void setFermiLevel(const double & x);
	double getFermiLevel();
	void setNumOccupiedStates(double x);
	double getNumOccupiedStates();
	void setHasSpinOrbit(bool x);
	bool getHasSpinOrbit();

	void setSmearingMethod(const int & x);
	int getSmearingMethod();
	void setSmearingWidth(const double & x);
	double getSmearingWidth();

	void setConstantRelaxationTime(const double & x);
	double getConstantRelaxationTime();

	void setScatteringMatrixInMemory(const bool & x);
	bool getScatteringMatrixInMemory();


	/** Reads the user-provided input file and saves the input parameters
	 * @param fileName: path to the input file
	 */
	void setupFromInput(string fileName);
};

#endif
