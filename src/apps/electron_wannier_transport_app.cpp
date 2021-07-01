#include "electron_wannier_transport_app.h"
#include "bandstructure.h"
#include "context.h"
#include "drift.h"
#include "el_scattering.h"
#include "electron_viscosity.h"
#include "exceptions.h"
#include "observable.h"
#include "onsager.h"
#include "parser.h"
#include "wigner_electron.h"

void ElectronWannierTransportApp::run(Context &context) {

  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  // load the elph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  InteractionElPhWan couplingElPh(crystal);
  if (std::isnan(context.getConstantRelaxationTime())) {
    couplingElPh = InteractionElPhWan::parse(context, crystal, &phononH0);
  }

  // compute the band structure on the fine grid
  if (mpi->mpiHead()) {
    std::cout << "\nComputing electronic band structure." << std::endl;
  }
  Points fullPoints(crystal, context.getKMesh());
  auto t3 = ActiveBandStructure::builder(context, electronH0, fullPoints);
  auto bandStructure = std::get<0>(t3);
  auto statisticsSweep = std::get<1>(t3);

  // Old code for using all the band structure
  //  bool withVelocities = true;
  //  bool withEigenvectors = true;
  //  FullBandStructure bandStructure = electronH0.populate(
  //      fullPoints, withVelocities, withEigenvectors);
  //  // set the chemical potentials to zero, load temperatures
  //  StatisticsSweep statisticsSweep(context, &bandStructure);

  // build/initialize the scattering matrix and the smearing
  ElScatteringMatrix scatteringMatrix(context, statisticsSweep, bandStructure,
                                      bandStructure, phononH0, &couplingElPh);
  scatteringMatrix.setup();
  scatteringMatrix.outputToJSON("rta_el_relaxation_times.json");

  // solve the BTE at the relaxation time approximation level
  // we always do this, as it's the cheapest solver and is required to know
  // the diagonal for the exact method.

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n\n";
    std::cout << "Solving BTE within the relaxation time approximation.\n";
  }

  // compute the phonon populations in the relaxation time approximation.
  // Note: this is the total phonon population n (n != f(1+f) Delta n)

  auto dimensionality = context.getDimensionality();
  BulkEDrift driftE(statisticsSweep, bandStructure, dimensionality);
  BulkTDrift driftT(statisticsSweep, bandStructure, dimensionality);
  VectorBTE relaxationTimes = scatteringMatrix.getSingleModeTimes();
  VectorBTE nERTA = -driftE * relaxationTimes;
  VectorBTE nTRTA = -driftT * relaxationTimes;

  // compute the electrical conductivity
  OnsagerCoefficients transportCoefficients(statisticsSweep, crystal,
                                            bandStructure, context);
  transportCoefficients.calcFromPopulation(nERTA, nTRTA);
  transportCoefficients.print();
  transportCoefficients.outputToJSON("rta_onsager_coefficients.json");

  // compute the Wigner transport coefficients
  WignerElCoefficients wignerCoefficients(
      statisticsSweep, crystal, bandStructure, context, relaxationTimes);
  wignerCoefficients.calcFromPopulation(nERTA, nTRTA);
  wignerCoefficients.print();
  wignerCoefficients.outputToJSON("rta_wigner_coefficients.json");

  // compute the electron viscosity
  ElectronViscosity elViscosity(context, statisticsSweep, crystal,
                                bandStructure);
  elViscosity.calcRTA(relaxationTimes);
  elViscosity.print();
  elViscosity.outputToJSON("rta_electron_viscosity.json");

  // compute the specific heat
  SpecificHeat specificHeat(context, statisticsSweep, crystal, bandStructure);
  specificHeat.calc();
  specificHeat.print();
  specificHeat.outputToJSON("el_specific_heat.json");

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n" << std::endl;
  }

  if (!std::isnan(context.getConstantRelaxationTime())) {
    return; // if we used the constant RTA, we can't solve the BTE exactly
  }

  //---------------------------------------------------------------------------
  // start section on exact BTE solvers

  std::vector<std::string> solverBTE = context.getSolverBTE();

  bool doIterative = false;
  bool doVariational = false;
  bool doRelaxons = false;
  for (const std::string &s : solverBTE) {
    if (s.compare("iterative") == 0)
      doIterative = true;
    if (s.compare("variational") == 0)
      doVariational = true;
    if (s.compare("relaxons") == 0)
      doRelaxons = true;
  }

  // here we do validation of the input, to check for consistency
  if (doRelaxons && !context.getScatteringMatrixInMemory()) {
    Error("Relaxons require matrix kept in memory");
  }
  if (context.getScatteringMatrixInMemory() &&
      statisticsSweep.getNumCalculations() != 1) {
    Error("If scattering matrix is kept in memory, only one "
          "temperature/chemical potential is allowed in a run");
  }
  if (doRelaxons && context.getUseSymmetries()) {
    Error("Relaxon solver only works without symmetries");
    // Note: this is a problem of the theory I suppose
    // because the scattering matrix may not be anymore symmetric
    // or may require some modifications to make it work
    // that we didn't have yet thought of
  }
  if (doVariational && context.getUseSymmetries()) {
    Error("Variational solver only works without symmetries");
    // Note: this is a problem of the theory I suppose
    // because the scattering matrix may not be anymore symmetric
    // or may require some modifications to make it work
    // that we didn't yet think of
  }

  if (context.getScatteringMatrixInMemory() && !context.getUseSymmetries()) {
    if (doVariational || doRelaxons || doIterative) {
      // reinforce the condition that the scattering matrix is symmetric
      // A -> ( A^T + A ) / 2
      // this helps removing negative eigenvalues which may appear due to noise
      scatteringMatrix.symmetrize();
      // it may not be necessary, so it's commented out
    }
  }

  if (doIterative) {

    if (mpi->mpiHead()) {
      std::cout << "Starting Omini Sparavigna BTE solver\n" << std::endl;
    }

    Eigen::Tensor<double, 3> elCond =
        transportCoefficients.getElectricalConductivity();
    Eigen::Tensor<double, 3> thCond =
        transportCoefficients.getThermalConductivity();
    Eigen::Tensor<double, 3> elCondOld = elCond;
    Eigen::Tensor<double, 3> thCondOld = thCond;

    VectorBTE nENext(statisticsSweep, bandStructure, 3);
    VectorBTE nTNext(statisticsSweep, bandStructure, 3);

    VectorBTE lineWidths = scatteringMatrix.getLinewidths();
    VectorBTE nEOld = nERTA;
    VectorBTE nTOld = nTRTA;

    auto threshold = context.getConvergenceThresholdBTE();

    for (int iter = 0; iter < context.getMaxIterationsBTE(); iter++) {

      std::vector<VectorBTE> nIn;
      nIn.push_back(nEOld);
      nIn.push_back(nTOld);
      auto nOut = scatteringMatrix.offDiagonalDot(nIn);
      nENext = nOut[0] / lineWidths;
      nTNext = nOut[1] / lineWidths;
      nENext = nERTA - nENext;
      nTNext = nTRTA - nTNext;

      transportCoefficients.calcFromPopulation(nENext, nTNext);
      transportCoefficients.print(iter);
      elCond = transportCoefficients.getElectricalConductivity();
      thCond = transportCoefficients.getThermalConductivity();

      // this exit condition must be improved
      // different temperatures might converge differently

      Eigen::Tensor<double, 3> diffE = ((elCond - elCondOld) / elCondOld).abs();
      Eigen::Tensor<double, 3> diffT = ((thCond - thCondOld) / thCondOld).abs();
      double dE = 10000.;
      double dT = 10000.;
      for (int i = 0; i < diffE.dimension(0); i++) {
        for (int j = 0; j < diffE.dimension(0); j++) {
          for (int k = 0; k < diffE.dimension(0); k++) {
            if (diffE(i, j, k) < dE)
              dE = diffE(i, j, k);
            if (diffT(i, j, k) < dT)
              dT = diffT(i, j, k);
          }
        }
      }
      if ((dE < threshold) && (dT < threshold)) {
        break;
      } else {
        elCondOld = elCond;
        thCondOld = thCond;
        nEOld = nENext;
        nTOld = nTNext;
      }

      if (iter == context.getMaxIterationsBTE() - 1) {
        Error("Reached max BTE iterations without convergence");
      }
    }
    transportCoefficients.print();
    transportCoefficients.outputToJSON("omini_onsager_coefficients.json");

    if (mpi->mpiHead()) {
      std::cout << "Finished Omini-Sparavigna BTE solver\n\n";
      std::cout << std::string(80, '-') << "\n" << std::endl;
    }
  }

  if (doVariational) {
    runVariationalMethod(context, crystal, statisticsSweep, bandStructure,
                         scatteringMatrix);
  }

  if (doRelaxons) {
    if (mpi->mpiHead()) {
      std::cout << "Starting relaxons BTE solver" << std::endl;
    }
    // scatteringMatrix.a2Omega(); Important!! Must use the matrix A, non-scaled
    // this because the scaling used for phonons here would cause instability
    // as it could contains factors like 1/0
    auto tup = scatteringMatrix.diagonalize();
    Eigen::VectorXd eigenvalues = std::get<0>(tup);
    ParallelMatrix<double> eigenvectors = std::get<1>(tup);
    // EV such that Omega = V D V^-1

    transportCoefficients.calcFromRelaxons(eigenvalues, eigenvectors,
                                           scatteringMatrix);
    transportCoefficients.print();
    transportCoefficients.outputToJSON("relaxons_onsager_coefficients.json");
    scatteringMatrix.relaxonsToJSON("exact_relaxation_times.json", eigenvalues);

    if (!context.getUseSymmetries()) {
      Vector0 fermiEigenvector(statisticsSweep, bandStructure, specificHeat);
      elViscosity.calcFromRelaxons(eigenvalues, eigenvectors);
      elViscosity.print();
      elViscosity.outputToJSON("relaxons_electron_viscosity.json");
    }

    if (mpi->mpiHead()) {
      std::cout << "Finished relaxons BTE solver\n\n";
      std::cout << std::string(80, '-') << "\n" << std::endl;
    }
  }
}

void ElectronWannierTransportApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");

  if (std::isnan(context.getConstantRelaxationTime())) { // non constant tau
    throwErrorIfUnset(context.getElphFileName(), "elphFileName");
    throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
    if (context.getSmearingMethod() == DeltaFunction::gaussian) {
      throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
    }
  } else {
    if (std::isnan(context.getNumOccupiedStates()) &&
        std::isnan(context.getFermiLevel())) {
      Error("For constant tau calculations, you must provide either the number "
            "of occupied Kohn-Sham states in the valence band or the Fermi "
            "level at T=0K");
    }
  }

  if (context.getDopings().size() == 0 &&
      context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or dopings must be set");
  }
}

void ElectronWannierTransportApp::runVariationalMethod(
    Context &context, Crystal &crystal, StatisticsSweep &statisticsSweep,
    ActiveBandStructure &bandStructure, ElScatteringMatrix &scatteringMatrix) {

  // here we implement a conjugate gradient solution to Az = b

  if (mpi->mpiHead()) {
    std::cout << "Starting variational BTE solver\n";
    std::cout << std::endl;
  }

  OnsagerCoefficients transportCoefficients(statisticsSweep, crystal,
                                            bandStructure, context);

  // initialize the transport coefficients
  // to check how these change during iterations
  Eigen::Tensor<double, 3> elCond =
      transportCoefficients.getElectricalConductivity();
  Eigen::Tensor<double, 3> thCond =
      transportCoefficients.getThermalConductivity();
  auto elCondOld = elCond;
  auto thCondOld = thCond;

  // set the initial guess to the RTA solution
  VectorBTE preconditioning2 = scatteringMatrix.diagonal();
  VectorBTE preconditioning = preconditioning2.sqrt();

  int numCalculations = statisticsSweep.getNumCalculations();

  // initialize b
  // Note: we solve Az=b not with z as electron population, but as canonical
  // population, i.e. with z being f = f0 + f0 (1-f0) z
  // where f0 is the Fermi-Dirac distribution
  // so, here we compute b without the factor n(1-n)
  // note also, we cannot do divisions by n(1-n) because it could be zero
  VectorBTE bE(statisticsSweep, bandStructure, 3);
  VectorBTE bT(statisticsSweep, bandStructure, 3);
  Particle particle = bandStructure.getParticle();
  for (int is : bandStructure.parallelIrrStateIterator()) {
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      auto chemPot = calcStat.chemicalPotential;
      auto temp = calcStat.temperature;
      for (int i : {0, 1, 2}) {
        bE(iCalc, i, iBte) = vel(i) / temp;
        bT(iCalc, i, iBte) = -vel(i) * (energy - chemPot) / temp / temp;
      }
    }
  }
  mpi->allReduceSum(&bE.data);
  mpi->allReduceSum(&bT.data);

  // apply CG preconditioning
  bE = bE / preconditioning;
  bT = bT / preconditioning;

  // populations, initial guess
  VectorBTE zNewE = bE;
  VectorBTE zNewT = bT;
  VectorBTE zE = zNewE;
  VectorBTE zT = zNewT;

  // Compute E^-1 A E^-1 z
  // since the preconditioning E is diagonal,
  // we can avoid explicitly making the matrix-matrix multiplications
  // and correct the diagonal contributions. Similar things will happen below
  std::vector<VectorBTE> inVec;
  inVec.push_back(zE);
  inVec.push_back(zT);
  std::vector<VectorBTE> w0s = scatteringMatrix.dot(inVec);
  VectorBTE w0E_ = zE * preconditioning2;
  VectorBTE w0T_ = zT * preconditioning2;
  VectorBTE w0E = w0s[0] - w0E_ + zE;
  VectorBTE w0T = w0s[1] - w0T_ + zT;

  // residual
  VectorBTE rE = bE - w0E;
  VectorBTE rT = bT - w0T;

  // search direction
  VectorBTE dE = rE;
  VectorBTE dT = rT;

  double threshold = context.getConvergenceThresholdBTE();

  for (int iter = 0; iter < context.getMaxIterationsBTE(); iter++) {
    // execute CG step, as in
    // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    // compute w = E^-1 A E^-1 d
    std::vector<VectorBTE> inW;
    inW.push_back(dE);
    inW.push_back(dT);
    std::vector<VectorBTE> outW = scatteringMatrix.dot(inW);
    VectorBTE wE_ = dE * preconditioning2;
    VectorBTE wT_ = dT * preconditioning2;
    VectorBTE wE = outW[0] - wE_ + dE;
    VectorBTE wT = outW[1] - wT_ + dT;

    // amount of descent along the search direction
    Eigen::MatrixXd alphaE = (rE.dot(rE)).array() / (dE.dot(wE)).array();
    Eigen::MatrixXd alphaT = (rT.dot(rT)).array() / (dT.dot(wT)).array();

    // new guess of population
    zNewE = dE * alphaE + zE;
    zNewT = dT * alphaT + zT;

    // new estimate of residual
    VectorBTE tmpE = wE * alphaE;
    VectorBTE tmpT = wT * alphaT;
    VectorBTE rNewE = rE - tmpE;
    VectorBTE rNewT = rT - tmpT;

    // amount of correction for the search direction
    Eigen::MatrixXd betaE = (rNewE.dot(rNewE)).array() / (rE.dot(rE)).array();
    Eigen::MatrixXd betaT = (rNewT.dot(rNewT)).array() / (rT.dot(rT)).array();

    // new search direction
    VectorBTE dNewE = dE * betaE + rNewE;
    VectorBTE dNewT = dT * betaT + rNewT;

    // now we update the guess of transport coefficients
    // first though, we add the factors n(1-n) that we removed from the CG
    // we need to do this because we didn't remove this factor from sMatrix
    VectorBTE z2E = zNewE;
    VectorBTE z2T = zNewT;
    for (int is : bandStructure.parallelIrrStateIterator()) {
      StateIndex isIdx(is);
      double energy = bandStructure.getEnergy(isIdx);
      int iBte = bandStructure.stateToBte(isIdx).get();
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
        auto chemPot = calcStat.chemicalPotential;
        auto temp = calcStat.temperature;
        double pop = sqrt(particle.getPopPopPm1(energy, temp, chemPot));
        for (int i : {0, 1, 2}) {
          z2E(iCalc, i, iBte) *= pop;
          z2T(iCalc, i, iBte) *= pop;
        }
      }
    }
    // now we compute the product E^-1 A E^-1 f, where f is the population
    // that is already preconditioned
    std::vector<VectorBTE> inF;
    inF.push_back(z2E);
    inF.push_back(z2T);
    auto outF = scatteringMatrix.dot(inF);
    VectorBTE ttE_ = z2E * preconditioning2;
    VectorBTE ttT_ = z2T * preconditioning2;
    VectorBTE az2E = outF[0] - ttE_ + z2E;
    VectorBTE az2T = outF[1] - ttT_ + z2T;
    transportCoefficients.calcVariational(az2E, az2T, z2E, z2T, zNewE, zNewT,
                                          preconditioning);
    transportCoefficients.print(iter);
    elCond = transportCoefficients.getElectricalConductivity();
    thCond = transportCoefficients.getThermalConductivity();

    // decide whether to exit or run the next iteration
    Eigen::Tensor<double, 3> diffE = ((elCond - elCondOld) / elCondOld).abs();
    Eigen::Tensor<double, 3> diffT = ((thCond - thCondOld) / thCondOld).abs();
    double deltaE = 10000.;
    double deltaT = 10000.;
    for (int i = 0; i < diffE.dimension(0); i++) {
      for (int j = 0; j < diffE.dimension(0); j++) {
        for (int k = 0; k < diffE.dimension(0); k++) {
          if (diffE(i, j, k) < deltaE) {
            deltaE = diffE(i, j, k);
          }
          if (diffT(i, j, k) < deltaT) {
            deltaT = diffT(i, j, k);
          }
        }
      }
    }
    if ((deltaE < threshold) && (deltaT < threshold)) {
      // remove the preconditioning from the population
      zNewE = zNewE / preconditioning;
      zNewT = zNewT / preconditioning;
      // recompute our final guess for transport coefficients
      transportCoefficients.calcFromCanonicalPopulation(zNewE, zNewT);
      break;
    } else {
      elCondOld = elCond;
      thCondOld = thCond;
      zE = zNewE;
      zT = zNewT;
      rE = rNewE;
      rT = rNewT;
      dE = dNewE;
      dT = dNewT;
    }

    if (iter == context.getMaxIterationsBTE() - 1) {
      Error("Reached max BTE iterations without convergence");
    }
  }

  // nice formatting of the transport properties at the last step
  transportCoefficients.print();
  transportCoefficients.outputToJSON("variational_onsager_coefficients.json");

  if (mpi->mpiHead()) {
    std::cout << "Finished variational BTE solver\n\n";
    std::cout << std::string(80, '-') << "\n" << std::endl;
  }
}
