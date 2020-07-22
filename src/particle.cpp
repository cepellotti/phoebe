#include "particle.h"
#include "exceptions.h"

Particle::Particle(int kind_) {
    if (kind_ == phonon) {
        statistics = bose;
        kind = phonon;
    } else if (kind_ == electron) {
        statistics = fermi;
        kind = electron;
    } else {
        Error e("Wrong initialization of Particle", 1);
    }
}

// copy constructor
Particle::Particle(const Particle &obj) {
    statistics = obj.statistics;
    kind = obj.kind;
}

// copy assignment operator
Particle& Particle::operator=(const Particle &obj) {
    if (this != &obj) {
        statistics = obj.statistics;
        kind = obj.kind;
    }
    return *this;
}

bool Particle::isFermi() {
    if (statistics == fermi) {
        return true;
    } else {
        return false;
    }
}

bool Particle::isBose() {
    if (statistics == bose) {
        return true;
    } else {
        return false;
    }
}

bool Particle::isElectron() {
    if (kind == electron) {
        return true;
    } else {
        return false;
    }
}

bool Particle::isPhonon() {
    if (kind == phonon) {
        return true;
    } else {
        return false;
    }
}

double Particle::getPopulation(const double &energy, const double &temperature,
        const double &chemicalPotential) {
    double population = 0.;
    double y = (energy - chemicalPotential) / temperature;

    //note: since the value of the exponential might be noisy for small/large
    // numbers, we make sure that the population is >=0 (or <=1 for fermions)

    if (statistics == bose) {
        population = 1. / (exp(y) - 1);
        if (population < 0.) {
            population = 0.;
        }
    } else {
        population = 1. / (exp(y) + 1);

        if (population < 0.) {
            population = 0.;
        }
        if (population > 1.) {
            population = 1.;
        }
    }
    return population;
}

double Particle::getDndt(const double &energy, const double &temperature,
        const double &chemicalPotential) {
    double x = getPopPopPm1(energy, temperature, chemicalPotential);
    double y = energy - chemicalPotential;
    double dndt = x * y / temperature / temperature;
    return dndt;
}

double Particle::getDnde(const double &energy, const double &temperature,
        const double &chemicalPotential) {
    double x = getPopPopPm1(energy, temperature, chemicalPotential);
    double dnde = -x / temperature;
    return dnde;
}

double Particle::getPopPopPm1(const double &energy, const double &temperature,
        const double &chemicalPotential) {
    double x = energy - chemicalPotential;
    double arg = x / 2. / temperature;
    if (statistics == bose) {
        x = sinh(arg); // numerically more stable than n(n+1)
    } else {
        x = cosh(arg); // numerically more stable than f(1-f)
    }
    x = 0.25 / x / x;
    return x;
}

int Particle::getParticleKind() {
  return kind;
}
