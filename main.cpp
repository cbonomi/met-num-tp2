#include "../src_catedra/vector_builder.h"
//#include "knn.h"
#include "clasificador.h"



int main() {
    std::string entries_path = "/home/christian/datos/imdb_tokenized.csv";
    ofstream cout("/home/christian/datos/ioutput.txt", std::ios_base::app);

    double valor_inicial = 0.00;
    double incremento_inicial = 0.001;
//    vector<double> frecs = {0.103, 0.108, 0.116, 0.124, 0.132, 0.14, 0.148, 0.156, 0.164, 0.172, 0.18, 0.188, 0.196};

//    for (int i=0; i<=40; i++) {
        //cout << valor_inicial + incremento_inicial*i*4 << endl;
//        double frecuence = frecs[i];//valor_inicial +incremento_inicial * i * 8;

    double frecuence = 0.188;
            auto filter_out = [](const int token, const FrecuencyVocabularyMap &vocabulary, const double frecuence) {
            /**
             *  Lambda para usar como filtro de vocabulario
             *  Retorna `true` si `token` debe eliminarse
             *  Retorna `false` si `token` no debe eliminarse
             **/
            double token_frecuency = vocabulary.at(token);
            return token_frecuency < 0.1 || token_frecuency > frecuence;//0.103;
        };
        VectorizedEntriesMap train_entries;
        VectorizedEntriesMap test_entries;
        build_vectorized_datasets(entries_path, train_entries, test_entries, filter_out, frecuence);
        int N = train_entries.begin()->second.bag_of_words.size();
//           cerr << "frec: " << frecuence << " dim: " << N << endl;
//    }
        cout
                << "N: " << N << std::endl
                << "Dataset de entrenamiento: " << train_entries.size() << " entradas" << std::endl
                << "Dataset de testeo: " << test_entries.size() << " entradas" << std::endl;

        std::cerr
                << "N: " << N << std::endl
                << "Dataset de entrenamiento: " << train_entries.size() << " entradas" << std::endl
                << "Dataset de testeo: " << test_entries.size() << " entradas" << std::endl;

        uint alpha = 40;
        vector<vector<double>> matrix = getMatrix(train_entries);
        vector<vector<double>> V = PCATecho(matrix, alpha);
        // vector<vector<double> > trainSetTemp = multMat(matrix, V);
        setMatrix(train_entries, V);

        std::cerr << "tamano matriz despues de PCA: " << train_entries[0].bag_of_words.size() << "x" << train_entries.size();

        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;
        int amount = 0;


        for (auto it = test_entries.begin(); it != test_entries.end(); it++) {
            std::cerr << "Prediciendo " << amount << " / " << train_entries.size() << '\r';
            bool label = it->second.is_positive;
            bool predi = knn(train_entries, it->second.bag_of_words, 5);
            if (label && predi) tp++;
            else if (label && !predi) fn++;
            else if (!label && predi) fp++;
            else if (!label && !predi) tn++;
            amount++;
        }

        cout
                << "                                    " << std::endl
                << tp << std::endl
                << fp << std::endl
                << tn << std::endl
                << fn << std::endl
                << "Recall: " << (tp / (double) (tp + fp)) << std::endl
                << "Precision: " << (tp / (double) (tp + fn)) << std::endl << std::endl;

        std::cerr
                << "                                    " << std::endl
                << tp << std::endl
                << fp << std::endl
                << tn << std::endl
                << fn << std::endl
                << "Recall: " << (tp / (double) (tp + fp)) << std::endl
                << "Precision: " << (tp / (double) (tp + fn)) << std::endl;
//    }
    return 0;
}
