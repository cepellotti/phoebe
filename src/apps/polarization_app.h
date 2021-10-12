#ifndef POLARIZATION_APP_H
#define POLARIZATION_APP_H

#include "app.h"
#include "electron_h0_wannier.h"

class ElectronPolarizationApp : public App {
 public:
  void checkRequirements(Context &context) override;
  void run(Context &context) override;

 private:
  Eigen::Tensor<double, 3> getProjectionsBlochOnAtoms(Points &points,
                                                      ElectronH0Wannier &h0);

  Eigen::Tensor<std::complex<double>, 4> projectionWannierSpace;
  Eigen::MatrixXd bVectors;
  Eigen::VectorXd bVectorsDeg;
  Eigen::VectorXd valenceCharges;

  void projectionBlochToWannier(Context &context, Crystal &crystal,
                                ElectronH0Wannier &h0);

  std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
  getElectronicPolarization(
      Crystal &crystal,
      StatisticsSweep &statisticsSweep,
      Context &context,
      BaseBandStructure &bandStructure, const double &spinFactor,
      const Eigen::Tensor<double, 3> &berryConnection,
      const Eigen::Tensor<double, 3> &projections);

  std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
    getIonicPolarization(Crystal &crystal, StatisticsSweep &statisticsSweep);

  Eigen::MatrixXd testElectronicPolarization(ElectronH0Wannier &h0,
                                             StatisticsSweep &statisticsSweep,
                                             Crystal &crystal);
};

#endif
