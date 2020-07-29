#include "interaction_3ph.h"
#include "mpiHelper.h"

long findIndexRow(Eigen::MatrixXd &cellPositions2, Eigen::Vector3d &position2) {
  long ir2 = -1;
  for (int i = 0; i < cellPositions2.cols(); i++) {
    if ((position2 - cellPositions2.col(i)).norm() == 0.) {
      ir2 = i;
      return ir2;
    }
  }
  if (ir2 == -1) {
    Error e("index not found");
  }
  return ir2;
}

// default constructor
Interaction3Ph::Interaction3Ph(Crystal &crystal_, long &numTriplets_,
                               Eigen::Tensor<double, 4> &ifc3Tensor_,
                               Eigen::Tensor<double, 3> &cellPositions_,
                               Eigen::Tensor<long, 2> &displacedAtoms_)
    : crystal(crystal_), numTriplets(numTriplets_), ifc3Tensor(ifc3Tensor_),
      cellPositions(cellPositions_), displacedAtoms(displacedAtoms_), dts(10),
      newdts(3) {

  numAtoms = crystal.getNumAtoms();
  numBands = numAtoms * 3;

  nr2 = 0;
  nr3 = 0;
  std::vector<Eigen::Vector3d> tmpCellPositions2, tmpCellPositions3;

  for (long it = 0; it < numTriplets; it++) {
    // load the position of the 2 atom in the current triplet
    Eigen::Vector3d position2, position3;
    for (int ic : {0, 1, 2}) {
      position2(ic) = cellPositions(it, 0, ic);
      position3(ic) = cellPositions(it, 1, ic);
    }
    // now check if this element is in the list.
    bool found2 = false;
    if (std::find(tmpCellPositions2.begin(), tmpCellPositions2.end(),
                  position2) != tmpCellPositions2.end()) {
      found2 = true;
    }
    bool found3 = false;
    if (std::find(tmpCellPositions3.begin(), tmpCellPositions3.end(),
                  position3) != tmpCellPositions3.end()) {
      found3 = true;
    }

    if (!found2) {
      tmpCellPositions2.push_back(position2);
      nr2++;
    }
    if (!found3) {
      tmpCellPositions3.push_back(position3);
      nr3++;
    }
  }

  cellPositions2 = Eigen::MatrixXd::Zero(3, nr2);
  cellPositions3 = Eigen::MatrixXd::Zero(3, nr3);
  for (int i = 0; i < nr2; i++) {
    cellPositions2.col(i) = tmpCellPositions2[i];
  }
  for (int i = 0; i < nr3; i++) {
    cellPositions3.col(i) = tmpCellPositions3[i];
  }

  D3 = Eigen::Tensor<double, 5>(numBands, numBands, numBands, nr2, nr3);
  D3.setZero();

  for (long it = 0; it < numTriplets; it++) { // sum over all triplets
    long ia1 = displacedAtoms(it, 0);
    long ia2 = displacedAtoms(it, 1);
    long ia3 = displacedAtoms(it, 2);

    Eigen::Vector3d position2, position3;
    for (int ic : {0, 1, 2}) {
      position2(ic) = cellPositions(it, 0, ic);
      position3(ic) = cellPositions(it, 1, ic);
    }

    long ir2 = findIndexRow(cellPositions2, position2);
    long ir3 = findIndexRow(cellPositions3, position3);

    for (int ic1 : {0, 1, 2}) {
      for (int ic2 : {0, 1, 2}) {
        for (int ic3 : {0, 1, 2}) {

          auto ind1 = compress2Indeces(ia1, ic1, numAtoms, 3);
          auto ind2 = compress2Indeces(ia2, ic2, numAtoms, 3);
          auto ind3 = compress2Indeces(ia3, ic3, numAtoms, 3);

          D3(ind1, ind2, ind3, ir2, ir3) = ifc3Tensor(ic3, ic2, ic1, it);
        }
      }
    }
  }
  ifc3Tensor.resize(0, 0, 0, 0);
  cellPositions.resize(0, 0, 0);
  displacedAtoms.resize(0, 0);

  // Copy everything to kokkos views

  Kokkos::realloc(cellPositions2_k, nr2, 3);
  Kokkos::realloc(cellPositions3_k, nr3, 3);

  auto cellPositions2_h = Kokkos::create_mirror_view(cellPositions2_k);
  auto cellPositions3_h = Kokkos::create_mirror_view(cellPositions3_k);
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < nr2; i++) {
      cellPositions2_h(i, j) = cellPositions2(j, i);
    }
    for (int i = 0; i < nr3; i++) {
      cellPositions3_h(i, j) = cellPositions3(j, i);
    }
  }
  Kokkos::deep_copy(cellPositions2_k, cellPositions2_h);
  Kokkos::deep_copy(cellPositions3_k, cellPositions3_h);

  ifc3Tensor.resize(0, 0, 0, 0);
  cellPositions.resize(0, 0, 0);
  displacedAtoms.resize(0, 0);

  cachedCoords << -111., -111., -111.;

  Kokkos::realloc(D3_k, numBands, numBands, numBands, nr3, nr2);
  Kokkos::realloc(D3PlusCached_k, numBands, numBands, numBands, nr3);
  Kokkos::realloc(D3MinsCached_k, numBands, numBands, numBands, nr3);
  auto D3_h = Kokkos::create_mirror_view(D3_k);
  for (int i1 = 0; i1 < numBands; i1++) {
    for (int i2 = 0; i2 < numBands; i2++) {
      for (int i3 = 0; i3 < numBands; i3++) {
        for (int i4 = 0; i4 < nr2; i4++) {
          for (int i5 = 0; i5 < nr3; i5++) {
            D3_h(i1, i2, i3, i5, i4) = D3(i1, i2, i3, i4, i5);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(D3_k, D3_h);
}

// copy constructor
Interaction3Ph::Interaction3Ph(const Interaction3Ph &that)
    : crystal(that.crystal), numTriplets(that.numTriplets),
      ifc3Tensor(that.ifc3Tensor), cellPositions(that.cellPositions),
      displacedAtoms(that.displacedAtoms), tableAtCIndex1(that.tableAtCIndex1),
      tableAtCIndex2(that.tableAtCIndex2), tableAtCIndex3(that.tableAtCIndex3),
      useD3Caching(that.useD3Caching), cellPositions2(that.cellPositions2),
      cellPositions3(that.cellPositions3), D3(that.D3), nr2(that.nr2),
      nr3(that.nr3), numAtoms(that.numAtoms), numBands(that.numBands),
      cachedCoords(that.cachedCoords), D3PlusCached(that.D3PlusCached),
      D3MinsCached(that.D3MinsCached) {
  std::cout << "copy constructor called\n";
}

// assignment operator
Interaction3Ph &Interaction3Ph::operator=(const Interaction3Ph &that) {
  if (this != &that) {
    crystal = that.crystal;
    numTriplets = that.numTriplets;
    ifc3Tensor = that.ifc3Tensor;
    cellPositions = that.cellPositions;
    displacedAtoms = that.displacedAtoms;
    tableAtCIndex1 = that.tableAtCIndex1;
    tableAtCIndex2 = that.tableAtCIndex2;
    tableAtCIndex3 = that.tableAtCIndex3;

    useD3Caching = that.useD3Caching;
    cellPositions2 = that.cellPositions2;
    cellPositions3 = that.cellPositions3;
    D3 = that.D3;
    nr2 = that.nr2;
    nr3 = that.nr3;
    numAtoms = that.numAtoms;
    numBands = that.numBands;
    cachedCoords = that.cachedCoords;
    D3PlusCached = that.D3PlusCached;
    D3MinsCached = that.D3MinsCached;
  }
  return *this;
  std::cout << "assignment operator called\n";
}

void Interaction3Ph::cacheD3(Eigen::Vector3d q2_e) {
  // copy q2 to kokkos
  Kokkos::View<double *> q2("q2", 3);
  auto q2_h = Kokkos::create_mirror_view(q2);
  for (int i = 0; i < 3; i++) {
    q2_h(i) = q2_e(i);
  }
  Kokkos::deep_copy(q2, q2_h);

  int nr2 = this->nr2;
  int nr3 = this->nr3;
  int numBands = this->numBands;
  Kokkos::complex<double> complexI(0.0, 1.0);

  auto D3PlusCached_k = this->D3PlusCached_k;
  auto D3MinsCached_k = this->D3MinsCached_k;
  auto cellPositions2_k = this->cellPositions2_k;
  auto cellPositions3_k = this->cellPositions3_k;
  auto D3_k = this->D3_k;

  // precompute phases
  Kokkos::View<Kokkos::complex<double> **> phasePlus("pp", nr3, nr2),
      phaseMins("pm", nr3, nr2);
  time_point t0 = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "phase1loop", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nr3, nr2}),
      KOKKOS_LAMBDA(int ir3, int ir2) {
        double argP = 0, argM = 0;
        for (int ic = 0; ic < 3; ic++) {
          argP +=
              +q2(ic) * (cellPositions2_k(ir2, ic) - cellPositions3_k(ir3, ic));
          argM +=
              -q2(ic) * (cellPositions2_k(ir2, ic) - cellPositions3_k(ir3, ic));
        }
        phasePlus(ir3, ir2) = Kokkos::exp(complexI * argP);
        phaseMins(ir3, ir2) = Kokkos::exp(complexI * argM);
      });
  time_point t1 = std::chrono::steady_clock::now();
  dts[0] += t1 - t0;

  // create cached D3
  Kokkos::parallel_for(
      "D3cacheloop",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, 0, 0, 0}, {numBands, numBands, numBands, nr3}),
      KOKKOS_LAMBDA(int ind1, int ind2, int ind3, int ir3) {
        Kokkos::complex<double> tmpp = 0, tmpm = 0;
        for (int ir2 = 0; ir2 < nr2; ir2++) { // sum over all triplets
          //          std::cout << ind1 << ", " << ind2 << ", " << ind3 << ",
          //          "
          //          << ir3 << ", " << ir2 << "\n";

          tmpp += D3_k(ind1, ind2, ind3, ir3, ir2) * phasePlus(ir3, ir2);
          tmpm += D3_k(ind1, ind2, ind3, ir3, ir2) * phaseMins(ir3, ir2);
        }
        D3PlusCached_k(ind1, ind2, ind3, ir3) = tmpp;
        D3MinsCached_k(ind1, ind2, ind3, ir3) = tmpm;
      });
  time_point t2 = std::chrono::steady_clock::now();
  dts[1] += t2 - t1;
}

std::tuple<std::vector<Eigen::Tensor<double, 3>>,
           std::vector<Eigen::Tensor<double, 3>>>
Interaction3Ph::getCouplingsSquared(
    std::vector<Eigen::Vector3d> q1s_e, Eigen::Vector3d q2_e,
    std::vector<Eigen::MatrixXcd> ev1s_e, Eigen::MatrixXcd ev2_e,
    std::vector<Eigen::MatrixXcd> ev3Pluss_e,
    std::vector<Eigen::MatrixXcd> ev3Minss_e, std::vector<int> nb1s_e, int nb2,
    std::vector<int> nb3Pluss_e, std::vector<int> nb3Minss_e) {

  int nr2 = this->nr2;
  int nr3 = this->nr3;
  int numBands = this->numBands;
  Kokkos::complex<double> complexI(0.0, 1.0);

  auto cellPositions2_k = this->cellPositions2_k;
  auto cellPositions3_k = this->cellPositions3_k;
  auto D3_k = this->D3_k;
  auto D3PlusCached_k = this->D3PlusCached_k;
  auto D3MinsCached_k = this->D3MinsCached_k;

  int nq1 = q1s_e.size();

  int maxnb1 = *std::max_element(nb1s_e.begin(), nb1s_e.end()),
      maxnb3Plus = *std::max_element(nb3Pluss_e.begin(), nb3Pluss_e.end()),
      maxnb3Mins = *std::max_element(nb3Minss_e.begin(), nb3Minss_e.end());

  Kokkos::View<double **> q1s("q1s", nq1, 3);
  Kokkos::View<Kokkos::complex<double> **> ev2("ev2", nb2, numBands);
  Kokkos::View<Kokkos::complex<double> ***> ev1s("ev1s", nq1, maxnb1, numBands),
      ev3Pluss("ev3p", nq1, maxnb3Plus, numBands),
      ev3Minss("ev3m", nq1, maxnb3Mins, numBands);
  Kokkos::View<int *> nb1s("nb1s", nq1), nb3Pluss("nb3ps", nq1),
      nb3Minss("nb3ms", nq1);

  // copy everything to kokkos views
  {
    auto q1s_h = Kokkos::create_mirror_view(q1s);
    auto ev1s_h = Kokkos::create_mirror_view(ev1s);
    auto ev2_h = Kokkos::create_mirror_view(ev2);
    auto ev3Pluss_h = Kokkos::create_mirror_view(ev3Pluss);
    auto ev3Minss_h = Kokkos::create_mirror_view(ev3Minss);
    auto nb1s_h = Kokkos::create_mirror_view(nb1s);
    auto nb3Pluss_h = Kokkos::create_mirror_view(nb3Pluss);
    auto nb3Minss_h = Kokkos::create_mirror_view(nb3Minss);
    for (int i = 0; i < nq1; i++) {
      nb1s_h(i) = nb1s_e[i];
      nb3Pluss_h(i) = nb3Pluss_e[i];
      nb3Minss_h(i) = nb3Minss_e[i];
      for (int j = 0; j < 3; j++) {
        q1s_h(i, j) = q1s_e[i][j];
      }
      for (int j = 0; j < numBands; j++) {
        for (int k = 0; k < nb1s_e[i]; k++) {
          ev1s_h(i, k, j) = ev1s_e[i](j, k);
        }
      }
      for (int j = 0; j < nb3Pluss_e[i]; j++) {
        for (int k = 0; k < numBands; k++) {
          ev3Pluss_h(i, j, k) = ev3Pluss_e[i](k, j);
        }
      }
      for (int j = 0; j < nb3Minss_e[i]; j++) {
        for (int k = 0; k < numBands; k++) {
          ev3Minss_h(i, j, k) = ev3Minss_e[i](k, j);
        }
      }
    }
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < nb2; j++) {
        ev2_h(j, i) = ev2_e(i, j);
      }
    }
    Kokkos::deep_copy(q1s, q1s_h);
    Kokkos::deep_copy(ev1s, ev1s_h);
    Kokkos::deep_copy(ev2, ev2_h);
    Kokkos::deep_copy(ev3Pluss, ev3Pluss_h);
    Kokkos::deep_copy(ev3Minss, ev3Minss_h);
    Kokkos::deep_copy(nb1s, nb1s_h);
    Kokkos::deep_copy(nb3Pluss, nb3Pluss_h);
    Kokkos::deep_copy(nb3Minss, nb3Minss_h);
  }

  Kokkos::View<Kokkos::complex<double> **> phasePlus("pp", nq1, nr3),
      phaseMins("pm", nq1, nr3);
  time_point t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nq1, nr3}),
      KOKKOS_LAMBDA(int iq1, int ir3) {
        double argP = 0, argM = 0;
        for (int ic : {0, 1, 2}) {
          argP += -q1s(iq1, ic) * cellPositions3_k(ir3, ic);
          argM += -q1s(iq1, ic) * cellPositions3_k(ir3, ic);
        }
        phasePlus(iq1, ir3) = exp(complexI * argP);
        phaseMins(iq1, ir3) = exp(complexI * argM);
      });
  time_point t1 = std::chrono::steady_clock::now();
  dts[2] += t1 - t0;

  Kokkos::View<Kokkos::complex<double> ****> tmpPlus("tmpp", nq1, numBands,
                                                     numBands, numBands),
      tmpMins("tmpm", nq1, numBands, numBands, numBands);
  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, 0, 0, 0}, {nq1, numBands, numBands, numBands}),
      KOKKOS_LAMBDA(int iq1, int iac1, int iac2, int iac3) {
        Kokkos::complex<double> tmpp = 0, tmpm = 0;
        for (int ir3 = 0; ir3 < nr3; ir3++) { // sum over all triplets
          tmpp += D3PlusCached_k(iac1, iac2, iac3, ir3) * phasePlus(iq1, ir3);
          tmpm += D3MinsCached_k(iac1, iac2, iac3, ir3) * phaseMins(iq1, ir3);
        }
        tmpPlus(iq1, iac1, iac2, iac3) = tmpp;
        tmpMins(iq1, iac1, iac2, iac3) = tmpm;
      });
  t1 = std::chrono::steady_clock::now();
  dts[3] += t1 - t0;
  Kokkos::realloc(phasePlus, 0, 0);
  Kokkos::realloc(phaseMins, 0, 0);

  Kokkos::View<Kokkos::complex<double> ****> tmp1Plus("t1p", nq1, maxnb1,
                                                      numBands, numBands),
      tmp1Mins("t1m", nq1, maxnb1, numBands, numBands);
  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, numBands, numBands}),
      KOKKOS_LAMBDA(int iq1, int ib1, int iac2, int iac3) {
        int mask = ib1 < nb1s(iq1);
        Kokkos::complex<double> tmpp = 0, tmpm = 0;

        for (int iac1 = 0; iac1 < numBands; iac1++) {
          tmpp += tmpPlus(iq1, iac1, iac2, iac3) * ev1s(iq1, ib1, iac1);
          tmpm += tmpMins(iq1, iac1, iac2, iac3) * ev1s(iq1, ib1, iac1);
        }
        tmp1Plus(iq1, ib1, iac3, iac2) = tmpp * mask;
        tmp1Mins(iq1, ib1, iac3, iac2) = tmpm * mask;
      });
  t1 = std::chrono::steady_clock::now();
  dts[4] += t1 - t0;
  Kokkos::realloc(tmpPlus, 0, 0, 0, 0);
  Kokkos::realloc(tmpMins, 0, 0, 0, 0);

  Kokkos::View<Kokkos::complex<double> ****> tmp2Plus("t2p", nq1, maxnb1, nb2,
                                                      numBands),
      tmp2Mins("t2m", nq1, maxnb1, nb2, numBands);
  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, nb2, numBands}),
      KOKKOS_LAMBDA(int iq1, int ib1, int ib2, int iac3) {
        int mask = ib1 < nb1s(iq1);

        Kokkos::complex<double> tmpp = 0, tmpm = 0;
        for (int iac2 = 0; iac2 < numBands; iac2++) {
          tmpp += tmp1Plus(iq1, ib1, iac3, iac2) * ev2(ib2, iac2);
          tmpm += tmp1Mins(iq1, ib1, iac3, iac2) * Kokkos::conj(ev2(ib2, iac2));
        }
        tmp2Plus(iq1, ib1, ib2, iac3) = tmpp * mask;
        tmp2Mins(iq1, ib1, ib2, iac3) = tmpm * mask;
      });
  t1 = std::chrono::steady_clock::now();
  dts[5] += t1 - t0;
  Kokkos::realloc(tmp1Plus, 0, 0, 0, 0);
  Kokkos::realloc(tmp1Mins, 0, 0, 0, 0);

  Kokkos::View<Kokkos::complex<double> ****> vPlus("vp", nq1, maxnb1, nb2,
                                                   maxnb3Plus),
      vMins("vm", nq1, maxnb1, nb2, maxnb3Mins);
  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, nb2, maxnb3Plus}),
      KOKKOS_LAMBDA(int iq1, int ib1, int ib2, int ib3) {
        int mask = ib1 < nb1s(iq1) && ib3 < nb3Pluss(iq1);
        Kokkos::complex<double> tmpp = 0;
        for (int iac3 = 0; iac3 < numBands; iac3++) {
          tmpp += tmp2Plus(iq1, ib1, ib2, iac3) *
                  Kokkos::conj(ev3Pluss(iq1, ib3, iac3));
        }
        vPlus(iq1, ib1, ib2, ib3) = tmpp * mask;
      });
  t1 = std::chrono::steady_clock::now();
  dts[6] += t1 - t0;

  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, nb2, maxnb3Mins}),
      KOKKOS_LAMBDA(int iq1, int ib1, int ib2, int ib3) {
        int mask = ib1 < nb1s(iq1) && ib3 < nb3Minss(iq1);

        Kokkos::complex<double> tmpp = 0;
        for (int iac3 = 0; iac3 < numBands; iac3++) {
          tmpp += tmp2Mins(iq1, ib1, ib2, iac3) *
                  Kokkos::conj(ev3Minss(iq1, ib3, iac3));
        }
        vMins(iq1, ib1, ib2, ib3) = tmpp * mask;
      });
  t1 = std::chrono::steady_clock::now();
  dts[7] += t1 - t0;
  Kokkos::realloc(tmp2Plus, 0, 0, 0, 0);
  Kokkos::realloc(tmp2Mins, 0, 0, 0, 0);

  Kokkos::View<double ****> couplingPlus("cp", nq1, maxnb1, nb2, maxnb3Plus),
      couplingMins("cp", nq1, maxnb1, nb2, maxnb3Mins);
  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, nb2, maxnb3Plus}),
      KOKKOS_LAMBDA(int iq1, int ib1, int ib2, int ib3) {
        int mask = ib1 < nb1s(iq1) && ib3 < nb3Pluss(iq1);

        auto tmp = vPlus(iq1, ib1, ib2, ib3);
        couplingPlus(iq1, ib1, ib2, ib3) =
            tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
      });
  t1 = std::chrono::steady_clock::now();
  dts[8] += t1 - t0;

  t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                             {nq1, maxnb1, nb2, maxnb3Mins}),
      KOKKOS_LAMBDA(int iq1, int ib1, int ib2, int ib3) {
        int mask = ib1 < nb1s(iq1) && ib3 < nb3Minss(iq1);

        auto tmp = vMins(iq1, ib1, ib2, ib3);
        couplingMins(iq1, ib1, ib2, ib3) =
            tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
      });
  t1 = std::chrono::steady_clock::now();
  dts[9] += t1 - t0;
  Kokkos::realloc(vPlus, 0, 0, 0, 0);
  Kokkos::realloc(vMins, 0, 0, 0, 0);

  std::vector<Eigen::Tensor<double, 3>> couplingPlus_e(nq1),
      couplingMins_e(nq1);
  auto couplingPlus_h = Kokkos::create_mirror_view(couplingPlus),
       couplingMins_h = Kokkos::create_mirror_view(couplingMins);
  Kokkos::deep_copy(couplingPlus_h, couplingPlus);
  Kokkos::deep_copy(couplingMins_h, couplingMins);
  for (int iq1 = 0; iq1 < nq1; iq1++) {
    int nb1 = nb1s_e[iq1];
    int nb3Plus = nb3Pluss_e[iq1];
    int nb3Mins = nb3Minss_e[iq1];
    couplingPlus_e[iq1] = Eigen::Tensor<double, 3>(nb1, nb2, nb3Plus);
    couplingMins_e[iq1] = Eigen::Tensor<double, 3>(nb1, nb2, nb3Mins);
    for (int ib1 = 0; ib1 < nb1; ib1++) {
      for (int ib2 = 0; ib2 < nb2; ib2++) {
        for (int ib3 = 0; ib3 < nb3Plus; ib3++) {
          couplingPlus_e[iq1](ib1, ib2, ib3) =
              couplingPlus_h(iq1, ib1, ib2, ib3);
        }
        for (int ib3 = 0; ib3 < nb3Mins; ib3++) {
          couplingMins_e[iq1](ib1, ib2, ib3) =
              couplingMins_h(iq1, ib1, ib2, ib3);
        }
      }
    }
  }
  return {couplingPlus_e, couplingMins_e};
}

Interaction3Ph::~Interaction3Ph() {
  if (mpi->mpiHead()) {
    std::cout << "Coupling kernel timing breakdown:"
              << "\n";
    std::cout << "nr2, nr3 phase loop: " << tosec(dts[0]) << "\n";
    std::cout << "D3Cached loop: " << tosec(dts[1]) << "\n";
    std::cout << "nr3 phase loop: " << tosec(dts[2]) << "\n";
    std::cout << "tmp loop: " << tosec(dts[3]) << "\n";
    std::cout << "tmp1 loop: " << tosec(dts[4]) << "\n";
    std::cout << "tmp2 loop: " << tosec(dts[5]) << "\n";
    std::cout << "vp loop: " << tosec(dts[6]) << "\n";
    std::cout << "vm loop: " << tosec(dts[7]) << "\n";
    std::cout << "cp loop: " << tosec(dts[8]) << "\n";
    std::cout << "cm loop: " << tosec(dts[9]) << "\n";
    std::cout << "total kernel time: "
              << tosec(dts[0]) + tosec(dts[1]) + tosec(dts[2]) + tosec(dts[3]) +
                     tosec(dts[4]) + tosec(dts[5]) + tosec(dts[6]) +
                     tosec(dts[7]) + tosec(dts[8]) + tosec(dts[9])
              << std::endl;
  }
}
