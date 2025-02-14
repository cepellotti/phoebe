#include "active_bandstructure.h"
#include "ifc3_parser.h"
#include "points.h"
#include "qe_input_parser.h"
#include <fstream>
#include <gtest/gtest.h>

/* This test checks that the active band structure construction
 * for a phonon Hamiltonian is generated the same way regardless
 * of if it was made via buildOnTheFly or buildAsPostProcessing.
 *
 * We use the phonon case, because the two functions should behave
 * the same when they are done using the same chemical potential,
 * and for phonons, of course, the chemical potential is zero.
 */
TEST(ActiveBandStructureTest, BandStructureStorage) {

  std::vector<Context> testContexts;

  // set up a phononH0
  Context context1;
  context1.setPhFC2FileName("../test/data/444_silicon.fc");
  context1.setWindowType("energy");
  Eigen::Vector2d x2;
  x2 << 0, 0.004;
  context1.setWindowEnergyLimit(x2);
  Eigen::VectorXd x3(1);
  x3(0) = 300. / temperatureAuToSi;
  context1.setTemperatures(x3);

  testContexts.push_back(context1);

  Context context2;
  context2.setPhFC2FileName("../test/data/444_silicon.fc");
  context2.setWindowType("population");
  context2.setWindowPopulationLimit(0.5e-8);
  context2.setTemperatures(x3);
  testContexts.push_back(context2);

  for (Context context : testContexts) {
    context.setUseSymmetries(true);

    auto tup = QEParser::parsePhHarmonic(context);
    auto crystal = std::get<0>(tup);
    auto phH0 = std::get<1>(tup);

    // Number of atoms
    int numAtoms = crystal.getNumAtoms();

    // setup parameters for active band structure creation
    Eigen::Vector3i qMesh;
    qMesh << 10, 10, 10;
    Points points(crystal, qMesh);
    bool withVelocities = true;
    bool withEigenvectors = true;

    // create two active band structures for phonons, built with different
    // methods
    //  call OTF or APP based on builder(..., forceBuildAsAPP)
    auto bsTup1 = ActiveBandStructure::builder(
        context, phH0, points, withEigenvectors, withVelocities, true);
    ActiveBandStructure absAPP = std::get<0>(bsTup1);

    auto bsTup2 = ActiveBandStructure::builder(
        context, phH0, points, withEigenvectors, withVelocities, false);
    ActiveBandStructure absOTF = std::get<0>(bsTup2);

    // TEST check that they selected the same number of states
    ASSERT_EQ(absOTF.getNumPoints(), absAPP.getNumPoints());

    // TEST check that the number of bands are the same
    int sumBands = 0;
    for (int point = 0; point < absOTF.getNumPoints(); point++) {
      auto ikTemp = WavevectorIndex(point);
      sumBands += absOTF.getNumBands(ikTemp) - absAPP.getNumBands(ikTemp);
    }
    ASSERT_EQ(sumBands, 0);

    // pick a wavevector to check energies, eigenvectors, velocities
    int ik = 7;
    auto ikIndex = WavevectorIndex(ik);

    // grab the points associated with this wavevector
    // TODO might be nice to check that they reference the same point
    Point pointOTF = absOTF.getPoint(ik);
    // Point pointAPP = absAPP.getPoint(ik);

    // get the number of bands at this point
    double nbOTF = absOTF.getNumBands(ikIndex);
    double nbAPP = absAPP.getNumBands(ikIndex);

    // generate values directly from the hamiltonian
    auto tup1 = phH0.diagonalize(pointOTF);
    auto ensT = std::get<0>(tup1);
    auto eigenVectorsT = std::get<1>(tup1);
    auto velocitiesT = phH0.diagonalizeVelocity(pointOTF);

    // get the values stored in each active band structure
    auto ensOTF = absOTF.getEnergies(ikIndex);
    auto ensAPP = absAPP.getEnergies(ikIndex);

    Eigen::Tensor<std::complex<double>, 3> eigenVectorsOTF =
        absOTF.getPhEigenvectors(ikIndex);
    Eigen::Tensor<std::complex<double>, 3> eigenVectorsAPP =
        absAPP.getPhEigenvectors(ikIndex);

    auto velocitiesOTF = absOTF.getVelocities(ikIndex);
    auto velocitiesAPP = absAPP.getVelocities(ikIndex);

    // TEST check OTF built band structure -----------------------

    // check the energies
    double otfEns = (ensT - ensOTF).norm();
    ASSERT_EQ(otfEns, 0.);

    // check the velocities
    std::complex<double> otfVelocities = complexZero;
    for (int ib1 = 0; ib1 < nbOTF; ib1++) {
      for (int ib2 = 0; ib2 < nbOTF; ib2++) {
        for (int ic = 0; ic < 3; ic++) {
          otfVelocities +=
              pow(velocitiesT(ib1, ib2, ic) - velocitiesOTF(ib1, ib2, ic), 2);
        }
      }
    }
    ASSERT_EQ(otfVelocities, complexZero);

    // check the eigenvectors
    std::complex<double> otfEigenVectors = complexZero;
    for (int i = 0; i < nbOTF; i++) {
      auto tup2 = decompress2Indices(i, numAtoms, 3);
      auto iat = std::get<0>(tup2);
      auto ic = std::get<1>(tup2);
      for (int j = 0; j < nbOTF; j++) {
        otfEigenVectors +=
            pow(eigenVectorsT(i, j) - eigenVectorsOTF(ic, iat, j), 2);
      }
    }
    ASSERT_EQ(otfEigenVectors, complexZero);

    // TEST check APP built band structure -----------------------

    // check the energies
    double appEns = (ensT - ensAPP).norm();
    ASSERT_EQ(appEns, 0.);

    // check the velocities
    std::complex<double> appVelocities = complexZero;
    for (int ib1 = 0; ib1 < nbAPP; ib1++) {
      for (int ib2 = 0; ib2 < nbAPP; ib2++) {
        for (int ic = 0; ic < 3; ic++) {
          appVelocities +=
              pow(velocitiesT(ib1, ib2, ic) - velocitiesAPP(ib1, ib2, ic), 2);
        }
      }
    }
    ASSERT_EQ(appVelocities, complexZero);

    // check the eigenvectors
    std::complex<double> appEigenVectors = complexZero;
    for (int i = 0; i < nbAPP; i++) {
      auto tup2 = decompress2Indices(i, numAtoms, 3);
      auto iat = std::get<0>(tup2);
      auto ic = std::get<1>(tup2);
      for (int j = 0; j < nbAPP; j++) {
        appEigenVectors +=
            pow(eigenVectorsT(i, j) - eigenVectorsAPP(ic, iat, j), 2);
      }
    }
    ASSERT_EQ(appEigenVectors, complexZero);
  }
}

/** In the test above, no state was filtered
 * Here we are testing that we can successfully throw away some states
 */
TEST(ActiveBandStructureTest, WindowFilter) {
  // set up input parameters
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setWindowType("population");
  context.setWindowPopulationLimit(1.0e-2);
  Eigen::VectorXd x3(1);
  x3(0) = 50. / temperatureAuToSi;
  context.setTemperatures(x3);
  Eigen::Vector3i qMesh;
  qMesh << 10, 10, 10;

  // read the phonon H0
  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phH0 = std::get<1>(tup);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();

  // setup parameters for active band structure creation
  Points points(crystal, qMesh);

  bool withVelocities = true;
  bool withEigenvectors = true;

  // create two active band structures for phonons, built with different methods
  //  call OTF or APP based on builder(..., forceBuildAsAPP)
  auto bsTup1 = ActiveBandStructure::builder(
      context, phH0, points, withEigenvectors, withVelocities, true);
  ActiveBandStructure absAPP = std::get<0>(bsTup1);
  auto bsTup2 = ActiveBandStructure::builder(
      context, phH0, points, withEigenvectors, withVelocities, false);
  ActiveBandStructure absOTF = std::get<0>(bsTup2);

  // TEST check that they selected the same number of states
  ASSERT_EQ(absOTF.getNumPoints(), absAPP.getNumPoints());
  ASSERT_LT(absOTF.getNumPoints(), qMesh(0) * qMesh(1) * qMesh(2));

  // TEST check that the number of bands are the same
  int sumBands = 0;
  for (int point = 0; point < absOTF.getNumPoints(); point++) {
    auto ikTemp = WavevectorIndex(point);
    sumBands += absOTF.getNumBands(ikTemp) - absAPP.getNumBands(ikTemp);
  }
  ASSERT_EQ(sumBands, 0);

  // pick a wavevector to check energies, eigenvectors, velocities
  int ik = 12;
  auto ikIndex = WavevectorIndex(ik);
  double nb = absOTF.getNumBands(ikIndex);

  // get the values stored in each active band structure
  auto ensOTF = absOTF.getEnergies(ikIndex);
  auto ensAPP = absAPP.getEnergies(ikIndex);

  Eigen::Tensor<std::complex<double>, 3> eigenVectorsOTF =
      absOTF.getPhEigenvectors(ikIndex);
  Eigen::Tensor<std::complex<double>, 3> eigenVectorsAPP =
      absAPP.getPhEigenvectors(ikIndex);

  auto velocitiesOTF = absOTF.getVelocities(ikIndex);
  auto velocitiesAPP = absAPP.getVelocities(ikIndex);

  // check the energies
  double ens = (ensOTF - ensAPP).norm();
  ASSERT_EQ(ens, 0.);

  // check the velocities
  std::complex<double> velocities = complexZero;
  for (int ib1 = 0; ib1 < nb; ib1++) {
    for (int ib2 = 0; ib2 < nb; ib2++) {
      for (int ic = 0; ic < 3; ic++) {
        velocities +=
            pow(velocitiesOTF(ib1, ib2, ic) - velocitiesAPP(ib1, ib2, ic), 2);
      }
    }
  }
  ASSERT_EQ(velocities, complexZero);

  // check the eigenvectors
  std::complex<double> eigenVectors = complexZero;
  for (int i = 0; i < nb; i++) {
    auto tup2 = decompress2Indices(i, numAtoms, 3);
    auto iat = std::get<0>(tup2);
    auto ic = std::get<1>(tup2);
    for (int j = 0; j < nb; j++) {
      eigenVectors +=
          pow(eigenVectorsOTF(ic, iat, j) - eigenVectorsAPP(ic, iat, j), 2);
    }
  }
  ASSERT_EQ(eigenVectors, complexZero);
}
