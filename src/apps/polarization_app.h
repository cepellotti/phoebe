#ifndef POLARIZATION_APP_H
#define POLARIZATION_APP_H

#include "app.h"

class ElectronPolarizationApp : public App {
public:
  void checkRequirements(Context &context) override;
  void run(Context &context) override;
};

#endif
