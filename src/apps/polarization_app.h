#ifndef POLARIZATION_APP_H
#define POLARIZATION_APP_H

#include "app.h"
#include "electron_h0_wannier.h"

class ElectronPolarizationApp : public App {
 public:
  void checkRequirements(Context &context) override;
  void run(Context &context) override;

 private:
  Eigen::VectorXd valenceCharges;

  void setVariablesFromFiles(Context &context, Crystal &crystal);

  Eigen::MatrixXd getElectronicPolarization(
      Crystal &crystal,
      StatisticsSweep &statisticsSweep,
      Context &context,
      BaseBandStructure &bandStructure, const double &spinFactor,
      const Eigen::Tensor<double, 3> &berryConnection);

  Eigen::MatrixXd getIonicPolarization(Crystal &crystal,
                                       StatisticsSweep &statisticsSweep);

//  Eigen::MatrixXd testElectronicPolarization(ElectronH0Wannier &h0,
//                                             StatisticsSweep &statisticsSweep,
//                                             Crystal &crystal);
};

#endif
