#include "phonon_elph_properties.h"
#include "bandstructure.h"
#include "context.h"
#include "el_scattering.h"
#include "eliashberg.h"
#include "exceptions.h"
#include "parser.h"

void PhononElPhPropertiesApp::run(Context &context) {
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
  couplingElPh = InteractionElPhWan::parse(context, crystal, &phononH0);

  // compute the electron band structure on the fine grid
  if (mpi->mpiHead()) {
    std::cout << "\nComputing electronic band structure." << std::endl;
  }

  Points fullKPoints(crystal, context.getQMesh());
  auto t3 = ActiveBandStructure::builder(context, electronH0, fullKPoints);
  auto elBandStructure = std::get<0>(t3);
  auto statisticsSweep = std::get<1>(t3);

  // compute the phonon band structure
  if (mpi->mpiHead()) {
    std::cout << "\nComputing phonon band structure." << std::endl;
  }
  // Here we don't filter any q(=k)-point or any band
  Points fullQPoints(crystal, context.getQMesh());
  // fullQPoints.setIrreduciblePoints();
  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure phBandStructure =
      phononH0.populate(fullQPoints, withVelocities, withEigenvectors);

  //-------------------

  EliashbergFunction eliashbergFunction(context, phBandStructure,
                                        statisticsSweep);
  VectorBTE phLinewidths(statisticsSweep, phBandStructure, 1);

  // build/initialize the scattering matrix and the smearing
  ElScatteringMatrix scatteringMatrix(context, statisticsSweep, elBandStructure,
                                      elBandStructure, phononH0, &couplingElPh);
  scatteringMatrix.buildPhProperties(phBandStructure, phLinewidths, eliashbergFunction);

  eliashbergFunction.getCriticalTemperature();

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n" << std::endl;
  }
}

void PhononElPhPropertiesApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPhD2FileName(), "phD2FileName");
  throwErrorIfUnset(context.getQMesh(), "qMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  throwErrorIfUnset(context.getCoulombRepulsion(), "coulombRepulsion");

  if ( context.getUseSymmetries() ) {
    Error("Cannot compute phonon properties with symmetries");
    // this is because currently the scattering rates are computed with a loop
    // on k and then a loop on k'. Using symmetries to compute the phonon el-ph
    // linewidth would require rewriting the code and do a loop on q first, then
    // a loop on k. Not something we can do without rewriting the scattering
    // builder() function
  }

  throwErrorIfUnset(context.getElphFileName(), "elphFileName");
  throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
  if (context.getSmearingMethod() == DeltaFunction::gaussian) {
    throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
  }

  if (context.getDopings().size() == 0 &&
      context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or doping concentrations must be set");
  }
}
