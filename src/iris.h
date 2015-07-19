#ifndef IRIS_H
#define IRIS_H

/*!
 * \file iris.h
 * iris feature extraction
 */

/*!
 * Get feature vector from iris image
 *
 * path: path to iris image
 * feature_vect: feature vector (32 floats)
 */
int get_iris_features(const char *path, float *feature_vect);

#endif // IRIS_H
