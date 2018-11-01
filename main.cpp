#include "../src_catedra/vector_builder.h"
#include "clasificador.h"

#define KNN 0
#define PCA 1


int main(int argc, char *argv[]) {
    auto filter_out = [] (const int token, const FrecuencyVocabularyMap & vocabulary) {
        /**
         *  Lambda para usar como filtro de vocabulario
         *  Retorna `true` si `token` debe eliminarse
         *  Retorna `false` si `token` no debe eliminarse
         **/
        double token_frecuency = vocabulary.at(token);
        return token_frecuency < 0.1 || token_frecuency > 0.5;
    };



	std::string entries_path = "../datos/imdb_tokenized.csv";
	bool output_to_file = false;
	string output_file_path = "";
	int k = 5;
	int alpha = 15;
	bool debug = true;
	uint method = PCA;
    for (uint c=1;c<argc;++c){
		string arg = argv[c];
		if (arg=="-m"){
			method = atoi(argv[c+1]);
		}
		if (arg=="-d"){
			entries_path = argv[c+1];
		}
		if (arg=="-o"){
			output_to_file = true;
			output_file_path = argv[c+1];
		}
		if (arg=="-k"){
			k = atoi(argv[c+1]);
		}
		if (arg=="-a" or arg=="-alpha"){
			alpha = atoi(argv[c+1]);
		}
		if (arg=="-debug"){
			debug = true;
		}
	}

    
    VectorizedEntriesMap train_entries;
    VectorizedEntriesMap test_entries;
    build_vectorized_datasets(entries_path, train_entries, test_entries, filter_out);
    int N = train_entries.begin()->second.bag_of_words.size();


    if (method == PCA){
	    vector<vector<double>> matrix = getMatrix(train_entries);
        cerr << "Calculando matriz de covarianza y sus autovectores" << endl;
	    vector<vector<double>> V = PCATecho(matrix, alpha);
	    cerr << "Terminando" << endl;

        cerr << "Aplicando transformaciones caracteristicas" << endl;
	    matrix = multMat(matrix, V);
        setMatrix(train_entries,matrix);

        matrix = getMatrix(test_entries);
        matrix = multMat(matrix,V);
        setMatrix(test_entries,matrix);

	}
    std::cerr
        << "Tamaño de los bags of words: " << test_entries.begin()->second.bag_of_words.size()  << " tokens" << std::endl
        << "Tamaño del dataset de training: " << train_entries.size() << " entradas" << std::endl
        << "Tamaño del dataset de testing: " << test_entries.size() << " entradas" << std::endl;
  

    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    int amount = 0;
    for (auto it = test_entries.begin(); it != test_entries.end(); it++) {
        std::cerr << "Prediciendo " << amount << " / " << test_entries.size() << '\r';
        bool label = it->second.is_positive;
        bool predi = knn(train_entries, it->second.bag_of_words, k);
        if (label && predi) tp++;
        else if (label && !predi) fn++;
        else if (!label && predi) fp++;
        else if (!label && !predi) tn++;
        amount++;
    }

    
    if (debug){
		std::cout
        << tp << std::endl
        << fp << std::endl
        << tn << std::endl
        << fn << std::endl; 		
        return 0;
    }

    std::cerr
            << "                                    " << std::endl
            << tp << std::endl
            << fp << std::endl
            << tn << std::endl
            << fn << std::endl
            << "Recall: " << (tp / (double)(tp + fn)) << std::endl
            << "Precision: " << (tp / (double)(tp + fp)) << std::endl;

    return 0;
}

