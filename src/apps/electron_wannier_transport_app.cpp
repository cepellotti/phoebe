#include "electron_wannier_transport_app.h"
#include "bandstructure.h"
#include "constants.h"
#include "context.h"
#include "drift.h"
#include "exceptions.h"
#include "io.h"
#include "observable.h"
#include "particle.h"
#include "el_scattering.h"
#include "onsager.h"
#include "qe_input_parser.h"
#include "specific_heat.h"

void ElectronWannierTransportApp::run(Context &context) {

  auto t1 = QEParser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);
  auto t2 = QEParser::parsePhHarmonic(context);
  auto crystalPh = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  // first we make compute the band structure on the fine grid

  FullPoints fullPoints(crystal, context.getQMesh());

  //	bool withVelocities = true;
  //	bool withEigenvectors = true;
  //	FullBandStructure bandStructure = phononH0.populate(
  //			fullPoints, withVelocities, withEigenvectors);
  //	// set the chemical potentials to zero, load temperatures
  //	StatisticsSweep statisticsSweep(context);

  auto t3 = ActiveBandStructure::builder(context, electronH0, fullPoints);
  auto bandStructure = std::get<0>(t3);
  auto statisticsSweep = std::get<1>(t3);

  // load the 3phonon coupling
  auto couplingElPh = InteractionElPhWan::parse(context.getEpwFileName());

  // build/initialize the scattering matrix and the smearing
  ElScatteringMatrix scatteringMatrix(context, statisticsSweep, bandStructure,
                                      bandStructure, &couplingElPh,
                                      &phononH0);
  scatteringMatrix.setup();

  // solve the BTE at the relaxation time approximation level
  // we always do this, as it's the cheapest solver and is required to know
  // the diagonal for the exact method.

  if ( mpi->mpiHead()) {
    std::cout << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "\n";
    std::cout << "Solving BTE within the relaxation time approximation.\n";
  }

  // compute the phonon populations in the relaxation time approximation.
  // Note: this is the total phonon population n (n != f(1+f) Delta n)

  auto dimensionality = context.getDimensionality();
  BulkEDrift driftE(statisticsSweep, bandStructure, dimensionality);
  BulkTDrift driftT(statisticsSweep, bandStructure, dimensionality);
  VectorBTE relaxationTimes = scatteringMatrix.getSingleModeTimes();
  VectorBTE nERTA = driftE * relaxationTimes;
  VectorBTE nTRTA = driftT * relaxationTimes;

  // compute the electrical conductivity
  OnsagerCoefficients transportCoeffs(statisticsSweep, crystal, bandStructure,
                                      context);
  transportCoeffs.calcFromPopulation(nERTA, nTRTA);
  transportCoeffs.print();

  if ( mpi->mpiHead()) {
    std::cout << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "\n";
  }

  mpi->barrier();
}

void ElectronWannierTransportApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getEpwFileName(), "EPWFileName");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
  throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
}
