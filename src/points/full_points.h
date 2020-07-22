#ifndef FULLPOINTS_H
#define FULLPOINTS_H

#include "points.h"

class FullPoints: public Points {
public:
    /** Default constructor.
     * @param crystal: the crystal object that defines the Brillouin zone.
     * @param mesh: grid size of the Monkhorst-pack.
     * @param offset: the offset of the grid w.r.t. the Gamma point. Offset
     * ranges in values from 0 to 1. 0.5 means half displacement of the grid.
     */
    FullPoints(Crystal &crystal_, const Eigen::Vector3i &mesh_,
            const Eigen::Vector3d &offset_ = Eigen::Vector3d::Zero());

    /** Copy constructor
     */
    FullPoints(const FullPoints &obj); // copy constructor

    /** Copy assignment operator
     */
    FullPoints& operator=(const FullPoints &obj); // assignment operator

    /** Returns a Point object given its integer index.
     * The Point object is used to move around the code the coordinates of the
     * wavevector.
     * @param index: the integer wavevector index ranging in [0,numPoints[
     * @return point: a point object.
     */
    Point getPoint(const long &index);
};

#endif
