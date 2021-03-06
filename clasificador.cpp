//
// Created by christian on 25/05/18.
//

#include "clasificador.h"
#include "../src_catedra/types.h"


using namespace std;

vector<double> calcularMedias(const vector<vector<double> >& imgs) {
    const unsigned long& m = imgs[0].size();
    vector<double> res (m) ;
    double acum;
    for (uint j = 0; j < m; j++) { //itero sobre la cantidad de variables (cantidad de palabras de cada review)
        acum = 0;
        for (uint i = 0; i < imgs.size(); i++){ //itero sobre la cantidad de reviews (cantidad de muestras de cada variable)
            acum += imgs[i][j]; //acumulo todos los valores
        }
        res[j] = acum/imgs.size();//divido por la cantidad de imagenes para obtener la media de cada variable y la guardo en el lugar correspondiente de res
    }
    return res;
}
/*vector<vector<double> > obtenerX(const vector<vector<double> >& imgs, const vector<double>& medias){
    const unsigned long& n = imgs.size();
    const unsigned long& m = imgs[0].size();
    vector<vector<double> > res (n, vector<double>(m));
    for (uint i = 0; i < n; i++){
        for (uint j = 0; j < m; j++){
            res[i][j] = imgs [i][j] - medias[j];//le resto la media correspondiente a cada variable
        }
    }
    return res;
}*/

vector<vector<double> > obtenerX(const vector<vector<double> >& muestras, const vector<double>& medias){
    const unsigned long& m = muestras.size();
    const unsigned long& n = muestras[0].size();
    vector<vector<double> > res (m, vector<double>(n));
    for (uint i = 0; i < m; i++){
        for (uint j = 0; j < n; j++){
            res[i][j] = muestras [i][j] - medias[i];//le resto la media correspondiente a cada variable
        }
    }
    return res;
}




vector<vector<double> > calcularMx (const vector<vector<double> >& X) {
    const size_t& n = X.size();
    const size_t& m = X[0].size();
    vector<vector<double> > res(m, vector<double>(m));
    for (int i = 0; i < m; ++i){
        for (int j = i; j < m; ++j){
            for (int k = 0; k < n; ++k){
                // Using X_T won't trash the cache.
                res[i][j] += X[k][i] * X[k][j];

            }
            res[i][j] = res[i][j]/(n-1);
            if (i!=j){
                res[j][i] = res[i][j];
            }
        }
    }
    return res;
}

vector<vector<double> > calcularMxTecho (const vector<vector<double> >& X) { //como traspongo X aca el m de aca es el n de la funcion de arriba y viceversa
    const size_t& n = X.size();
    const size_t& m = X[0].size();
    vector<vector<double> > res(m, vector<double>(m));
    double covar_ij;
    double& var_i = covar_ij;
    for (uint i = 0; i < m; i++){
        var_i = 0;
        for (uint k = 0; k < n; k++){
            var_i += X[k][i]*X[k][i]; //calculo de la sumatoria de productos para calcular varianza
        }
        res[i][i] = var_i/(m-1);
        for (uint j = i+1; j < m; j++){ //como la matriz es simetrica basta calcular la mitad superior.
            covar_ij = 0;
            for (uint k = 0; k < n; k++){
                covar_ij += X[k][i]*X[k][j];//calculo de la sumatoria de productos para calcular covarianza
            }
            res[i][j] = covar_ij/(m-1);
            res[j][i] = covar_ij/(m-1);
        }
    }
    return res;
}




vector<vector<double> > trasponer(const vector<vector<double> >& mat){
    const unsigned long& n = mat.size();
    const unsigned long& m = mat[0].size();
    vector<vector<double> > res (m, vector<double>(n));
    for (uint i = 0; i<n;i++)
        for (uint j = 0; j<m;j++)
            res[j][i] = mat[i][j];
    return res;

}

double norma1(const vector<double> &v){
    double res = 0;
    for(size_t i = 0; i < v.size(); ++i)
        res += abs(v[i]);
    return res;
}

double norma2(const vector<double> &vec){//no tomo raiz porque no hace falta en nuestro caso
    double acum = 0;
    for (uint i = 0; i < vec.size(); i++) {
        acum+= vec[i]*vec[i];
    }
    return acum;
}

vector<double> mult_matr_por_vect(const vector<vector<double> > &M, const vector<double> &v){
    const size_t& n = v.size();
    vector<double> res(n);
    for(size_t i = 0; i < n; ++i) {
        res[i] = 0;
        for (size_t k = 0; k < n; ++k)
            res[i] += M[i][k]*v[k];
    }
    return res;
}

double producto_interno(const vector<double> &v, const vector<double> &v2){
    double res = 0;
    if(v.size() != v2.size())
        cout << "producto_interno: los vectores no son del mismo tamaño" << endl;
    else{
        for(size_t i = 0; i < v.size(); ++i)
            res += v[i]*v2[i];
    }
    return res;
}

double normalizar1(vector<double>& v){     //Según norma 1. Devuelve la norma 1 que tenía el vector.
    double norma = norma1(v);
    for(size_t i = 0; i < v.size(); ++i)
        v[i] = v[i]/norma;
    return norma;
}

void normalizar2(vector<double>& v){     //Según norma 2
    double norma = sqrt(norma2(v));
    for(size_t i = 0; i < v.size(); ++i)
        v[i] = v[i]/norma;
}

vector<double> restaVec(const vector<double> &vec1, const vector<double> &vec2) {
    vector<double> res (vec1.size());
    for (uint i = 0; i < vec1.size(); i++)
        res[i] = vec1[i]-vec2[i];
    return res;
}


vector<double> multVecEsc(const vector<double> &vec1, const double &esc) {
    vector<double> res (vec1.size());
    for (uint i = 0; i < vec1.size(); i++)
        res[i] = vec1[i]*esc;
    return res;
}

pair<double,vector<double> > metodoPotencia(const vector<vector<double> > &M) {
    const size_t& n = M[0].size();
    pair<double,vector<double> > res2;
    double& autovalor2 = res2.first;
    double autovalor2_temp;
    vector<double>& autovector2 = res2.second;
    autovector2 = vector<double>(n,0);
    autovector2[0] = 1;   //Empieza siendo e1.
    vector<double> autovector2_temp;
    //Cálculo del autovalor:
    double diferencia2 = 1;
    float cantidad_iteraciones2 = 0;
    //auto t3 = chrono::system_clock::now();
    while(diferencia2 >= 0.0001 or cantidad_iteraciones2 < 30){
        autovector2_temp = mult_matr_por_vect(M, autovector2);
        autovalor2_temp = producto_interno(autovector2, autovector2_temp); //autovector está normalizado.
        normalizar2(autovector2_temp);
        autovector2 = mult_matr_por_vect(M, autovector2_temp);
        autovalor2 = producto_interno(autovector2_temp, autovector2); //autovector_temp está normalizado.
        normalizar2(autovector2);
        diferencia2 = abs(autovalor2-autovalor2_temp);
        ++cantidad_iteraciones2;
    }
    //auto t4 = chrono::system_clock::now();
    if(autovalor2 < 0){
        autovector2 = mult_matr_por_vect(M, autovector2);
        normalizar2(autovector2);
        diferencia2 = norma1(restaVec(autovector2, autovector2_temp));
        while(diferencia2 > 0.001){
            autovector2_temp = mult_matr_por_vect(M, autovector2);
            normalizar2(autovector2_temp);
            autovector2 = mult_matr_por_vect(M, autovector2_temp);
            autovector2 = mult_matr_por_vect(M, autovector2);
            normalizar2(autovector2);
            diferencia2 = norma1(restaVec(autovector2, autovector2_temp));
            ++cantidad_iteraciones2;
        }
    }else{
        diferencia2 = norma1(restaVec(autovector2, autovector2_temp));
        while(diferencia2 > 0.001){
            autovector2_temp = mult_matr_por_vect(M, autovector2);
            normalizar2(autovector2_temp);
            autovector2 = mult_matr_por_vect(M, autovector2_temp);
            normalizar2(autovector2);
            diferencia2 = norma1(restaVec(autovector2, autovector2_temp));
            ++cantidad_iteraciones2;
        }
    }
    return res2;
}


void multMatEsc(vector<vector<double> > &mat, double escalar) {//para multiplicar una matriz por un escalar. Afecta a la matriz parámetro.
    for (uint i = 0; i<mat.size();i++)
        for (uint j = 0; j<mat[i].size();j++)
            mat[i][j] *= escalar;
}

vector<vector<double> > multVec(const vector<double> &vec1) {//para generar una matriz a partir de un vector y su traspuesto
    const size_t& n = vec1.size();
    vector<vector<double> > res(n, vector<double>(n));
    for (uint i = 0; i<n;i++)
        for (uint j = 0; j<n;j++)
            res[i][j] = vec1[i]*vec1[j];
    return res;
}


void sumMat(vector<vector<double> > &mat1, const vector<vector<double> > &mat2) {//suma de matrices
    for (uint i = 0; i<mat1.size();i++)
        for (uint j = 0; j<mat1.size();j++)
            mat1[i][j] += mat2[i][j];
}

vector< pair<double,vector<double> > > deflacion(vector<vector<double> > mat, uint alpha) {
    vector< pair<double,vector<double> > > res (alpha);
    for (uint i = 0; i < alpha; i++){
        res[i] = metodoPotencia(mat);
        vector<vector<double> > v_x_vt = multVec(res[i].second);    //v*vt
        multMatEsc(v_x_vt,res[i].first*(-1));                       //-lambda_i*(v*vt)
        sumMat(mat, v_x_vt);
    }
    return res;
}

vector<vector<double> > generarV(const vector<vector<double> > &mat, uint alpha){
    vector<vector<double> > res(mat[0].size(), vector<double>(alpha));
    vector< pair<double,vector<double> > > autovectores = deflacion(mat,alpha);
    for (uint j = 0; j < alpha; ++j)
        for(uint i = 0; i < mat[0].size(); ++i)
            res[i][j] = autovectores[j].second[i];
    return res;
}

bool knn(const VectorizedEntriesMap& train_entries, std::vector<double> bag_of_words, uint k) {

    //Construyo los primeros k vecinos para el heap
    priority_queue<pair<double,bool> > heap;  //Creo un max_heap para los primeros k elementos

    auto entry = train_entries.begin();

    unsigned long delta = 0;
    unsigned long start, end;
    //RDTSC_START(start);

    for (int i=0; i<k;++i){
        double norma = norma2(restaVec(entry->second.bag_of_words, bag_of_words));
        heap.push(make_pair(norma, entry->second.is_positive));
        entry++;
    }

    //RDTSC_STOP(end);
    //delta += end - start;


    while (entry != train_entries.end()) {
        double norma = norma2(restaVec(entry->second.bag_of_words, bag_of_words));
        if(norma < heap.top().first){  //Si el i-ésimo elemento es más chico que el más grande del heap...
            heap.pop();                 //entonces saco al más grande...
            heap.push(make_pair(norma, entry->second.is_positive));    //y meto al nuevo elemento.
        }                               //De esta forma me quedo con los k elementos más chicos.
        entry++;
    }
    


    //Ahora contamos los negativos y los positivos de los k mas cercanos

    int vecinos_positivos = 0;
    int vecinos_negativos = 0;

    while(!heap.empty()){
        if (heap.top().second) {
            vecinos_positivos++;
        } else {
            vecinos_negativos++;
        }
        heap.pop();
    }

    return vecinos_positivos >= vecinos_negativos;
}

vector<vector<double> > multMat(const vector<vector<double> >& mat1, const vector<vector<double> >& mat2) {
    const unsigned long& n = mat1.size();
    const unsigned long& m = mat2[0].size();
    const unsigned long& l = mat2.size();
    vector<vector<double> > res (n, vector<double>(m, 0));
    for (uint i = 0; i < n; i++)
        for (uint j = 0; j < m; j++)
            for (uint k = 0; k < l; k++)
                res[i][j] += mat1[i][k]*mat2[k][j];
    return res;
}

/*/vector<vector<double> > PCA (vector<vector<double> > trainX, uint alpha) {
    const unsigned long& m = trainX[0].size();
    vector<vector<double> > Mx = calcularMx(trainX);
    vector<vector<double> > V = generarV(Mx,alpha);
    return V; //devuelvo la V, recordar multiplicar fuera de la funcion
}*/

vector<vector<double> > PCATecho (vector<vector<double> > &Matriz, uint alpha) {
    const unsigned long& m = Matriz[0].size();
    vector<double> medias = calcularMedias(Matriz);
    vector<vector<double> > Xt = obtenerX(Matriz, medias);
    vector<vector<double> > Mx = calcularMx(Xt);
    vector<vector<double> > V = generarV(Mx,alpha);
    V = trasponer(V);
    for (uint i = 0; i < V.size(); i++){
        normalizar2(V[i]);
    }
    V = trasponer(V);
    return V; //devuelvo la V, recordar multiplicar fuera de la funcion
}

vector<vector<double>> getMatrix(VectorizedEntriesMap train_entries) {
    vector<vector<double>> matrix(0);
    for (const auto entry : train_entries) {
        /* vector<double> row(0);
         for (const auto word_count : entry.second.bag_of_words) {
             row.push_back(word_count);
         }*/
        matrix.push_back(entry.second.bag_of_words);
    }
    return matrix;
}

void setMatrix(VectorizedEntriesMap &data_entries, vector<vector<double>> matrix) {
    int i = 0;
    for (auto it = data_entries.begin(); it != data_entries.end(); ++it){
        data_entries[it->first].bag_of_words = matrix[i];
        i++;
    }
}


/*vector<uint> vectorDeKnns (const vector<vector<double> >& trainX, const vector<clase_t>& labelsX, const vector<vector<double> >& testY, uint k){
    const unsigned long& n = testY.size();
    vector<uint> res (n);
    for (uint i = 0; i < n; i++) {
        res[i] = Knn(trainX,labelsX,testY[i],k);
    }
    return res;
}*/