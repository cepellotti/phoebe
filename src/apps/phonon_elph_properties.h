#ifndef PHELPHPROPERTIESAPP_H
#define PHELPHPROPERTIESAPP_H

#include "app.h"
#include "el_scattering.h"
#include <string>

/** Main driver for the transport calculation
 */
class PhononElPhPropertiesApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

#endif
