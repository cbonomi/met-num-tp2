//
// Created by christian on 25/05/18.
//

#ifndef CLASIFICADOR_H
#define CLASIFICADOR_H


#include "../src_catedra/types.h"
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <map>
#include <random>
#include <queue>
#include <tuple>
#include <cmath>
#include <functional>

//#include "util.h"

using namespace std;

//vector<vector<double> > PCATecho (vector<vector<double> > trainX, uint alpha);
vector<vector<double> > multMat(const vector<vector<double> >& mat1, const vector<vector<double> >& mat2);
//void multMatEsc(vector<vector<double> > &mat, double escalar); //para multiplicar una matriz por un escalar. Afecta a la matriz parámetro.
//vector<uint> vectorDeKnns (const vector<vector<double> >& trainX, const vector<clase_t>& labelsX, const vector<vector<double> >& testY, uint k);
//clase_t Knn (const VectorizedEntriesMap& vecNormas, uint k);
bool knn(const VectorizedEntriesMap& train_entries, std::vector<double> bag_of_words, uint k);
//vector<pair<double,clase_t> > vector_de_distancias(const vector<vector<double> >& trainX, const vector<clase_t>& labelsX, const vector<double>& newImg);
vector<vector<double>> getMatrix(VectorizedEntriesMap train_entries);
void setMatrix(VectorizedEntriesMap train_entries, vector<vector<double>> matrix);
vector<vector<double> > PCATecho (vector<vector<double> > &Matriz, uint alpha);
#endif //CLASIFICADOR_H
