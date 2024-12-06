#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <omp.h>
#include <cassert>
#include <iomanip>        
#include <climits>        
#include <limits>  


double generateUniform(double mean, double std);
class NeuralNetwork {
public:
    int n_x, n_h, n_y, n_m;
    std::vector<int> noOfNeurons;
    std::unordered_map<std::string, std::vector<std::vector<double>>> parameters, cache, gradients;
    std::vector<std::vector<double>> X;
    std::vector<std::string> Activations;

    NeuralNetwork(int n_x, int n_m, int n_h, int n_y, std::vector<std::string> Activations) {
        this->n_x = n_x;
        this->n_h = n_h;
        this->n_y = n_y;
        this->n_m = n_m;
        noOfNeurons.resize(n_h);
        this->Activations = Activations;
    }

    void setInputData(std::vector<std::vector<double>> X) {
        this->X=X;
    }
    void printDimensions(const std::vector<std::vector<double>>& matrix, const std::string& name) {
        std::cout << name << " dimensions: " << matrix.size() << "x" << matrix[0].size() << std::endl;
    }
    double activationFunction(double Z, std::string typeActivation = "linear") {
        if (typeActivation == "relu") {
            return Z <= 0 ? 0 : Z;
        } else if (typeActivation == "sigmoid") {
            return (1 / (1 + exp(-Z)));
        } else if (typeActivation == "linear") {
            return Z;
        }
        return Z;  // Default to linear if no valid activation type is passed
    }

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>> &matrix){
        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));
    
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }
    return transposed;
    }

std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B) {
        if (A[0].size() != B.size()) {
            std::cerr << "Error: Matrix dimensions do not match for dot product!" << std::endl;
            exit(EXIT_FAILURE);
        }
        int rows = A.size();
        int cols = B[0].size();
        int innerDim = B.size();
    
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < innerDim; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    return result;
}

void zScoreNormalize(std::vector<std::vector<double>>& data) {
    for (size_t col = 0; col < data[0].size(); ++col) {
        double sum = 0.0;
        double sumSq = 0.0;
        size_t n = data.size();
        for (size_t row = 0; row < n; ++row) {
            sum += data[row][col];
            sumSq += data[row][col] * data[row][col];
        }
        double mean = sum / n;
        double variance = (sumSq / n) - (mean * mean);
        double stdDev = std::sqrt(variance);
        for (size_t row = 0; row < n; ++row) {
            if (stdDev != 0) {
                data[row][col] = (data[row][col] - mean) / stdDev;
            } else {
                data[row][col] = 0; 
            }
        }
    }
}
    
void declareParameters(std::vector<int> noOfNeurons) {
    this->noOfNeurons = noOfNeurons;

    for (int i = 0; i < n_h; i++) {
        std::vector<std::vector<double>> W;  
        std::vector<std::vector<double>> B(noOfNeurons[i], std::vector<double>(1, 0.0)); 
        std::vector<std::vector<double>> Z(noOfNeurons[i], std::vector<double>(n_m));  
        std::vector<std::vector<double>> A(noOfNeurons[i], std::vector<double>(n_m));
        if (i == 0) {
            W.resize(noOfNeurons[i], std::vector<double>(n_x)); 
        } else {
            W.resize(noOfNeurons[i], std::vector<double>(noOfNeurons[i - 1]));  
        }

        parameters["W" + std::to_string(i)] = W;
        parameters["b" + std::to_string(i)] = B;
        if (i != 0) {
            cache["Z" + std::to_string(i)] = Z;
            cache["A"+std::to_string(i)] =  A;
        }

        std::cout << "Layer " << i << ": " << std::endl;
        std::cout << "  Weight dimensions W" << i << ": " << W.size() << " x " << W[0].size() << std::endl;
        std::cout << "  Bias dimensions B" << i << ": " << B.size() << " x " << B[0].size() << std::endl;
        std::cout << "  Activation dimensions Z" << i << ": " << Z.size() << " x " << Z[0].size() << std::endl;
    }
    // Initialize Z0 with input data X
    std::cout << "first datapoint value before normalization " <<X[0][0] << std::endl;
    zScoreNormalize(X);
    std::cout << "First datapoint value after normalization" << X[0][0] << std::endl;

    cache["Z0"] = X; 
}

    double derivativeActivation(double Z, std::string activation = "relu") {
        if (activation == "relu") {
            return Z > 0 ? 1 : 0;
        } else if (activation == "sigmoid") {
            double sigmoid = 1 / (1 + exp(-Z));
            return sigmoid * (1 - sigmoid);
        } else if (activation == "linear") {
            return 1;
        }
        return 0;
    }

void initializeParameters() {
    srand(time(0));
    for (int i = 0; i < n_h; i++) {
        int inputSize = (i == 0) ? n_x : noOfNeurons[i - 1]; 
        int outputSize = noOfNeurons[i];
        std::vector<std::vector<double>> W(outputSize, std::vector<double>(inputSize));
        for (int z = 0; z < outputSize; z++) {
            for (int j = 0; j < inputSize; j++) {
                W[z][j] = ((double) rand() / RAND_MAX) * sqrt(2.0 / inputSize);  // He Initialization
            }
        }
        std::vector<std::vector<double>> B(outputSize, std::vector<double>(1, 0.0));
        parameters["W" + std::to_string(i)] = W;
        parameters["b" + std::to_string(i)] = B;
    }
    cache["Z0"] = X;
}

std::vector<std::vector<double>> forwardProp(const std::string& activation, const std::vector<std::vector<double>>& W,
                                   const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b, int layer) {
    int numRows = W.size();
    int numCols = n_m;
    int numColsA = A.size();  
    
    if (W[0].size() != numColsA) {
        std::cerr << "Error: Matrix dimensions do not match for multiplication!" << std::endl;
        std::cerr << "W dimensions: " << W.size() << "x" << W[0].size() << ", A dimensions: " << A.size() << "x" << A[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> Z(numRows, std::vector<double>(numCols));
    Z = dotProduct(W, A);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                Z[i][j] += b[i][0];
            }
        }
    cache["Z" + std::to_string(layer)] = Z;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            Z[i][j] = activationFunction(Z[i][j], activation);
        }
    }
    cache["A" + std::to_string(layer)] = Z;
    return Z;
}

std::vector<std::vector<double>> computeY() {
    std::vector<std::vector<double>> A;
    for (int i = 0; i < n_h; i++) {
        //cout << "Forward prop begins " << endl;
        std::vector<std::vector<double>> W = parameters["W" + std::to_string(i)];
        std::vector<std::vector<double>> b = parameters["b" + std::to_string(i)];
        std::vector<std::vector<double>> A_prev;
        if(i==0){
            A_prev = X;
        } else {
            A_prev = cache["A" + std::to_string(i-1)];
        }
        std::vector<std::vector<double>> A_prev_T;
        A_prev_T = transpose(A_prev);
        if(i!=0){
            A_prev_T = transpose(A_prev_T);
        }
        A = forwardProp(Activations[i], W, A_prev_T, b, i);
    }
    return A;
}

void backProp(const std::vector<std::vector<double>>& Y_true) {
    int L = n_h-1;  
    int m = Y_true.size(); 
 
    std::vector<std::vector<double>> A_L = cache["A" + std::to_string(L)];
    double mse = 0.0;
    for (int i = 0; i < m; ++i) {
        mse += pow((A_L[0][i] - Y_true[i][0]), 2);
    }
    mse /= m;
    double rmse = sqrt(mse);
    
    std::vector<std::vector<double>> dA_L(m, std::vector<double>(1));
    for (int i = 0; i < m; ++i) {
        dA_L[i][0] = (1.0/rmse) * (2.0/m) * (A_L[0][i]-Y_true[i][0]);
    }

    gradients["dZ" + std::to_string(L)] = dA_L;

    std::vector<std::vector<double>> A_prev_T = transpose(cache["A" + std::to_string(L-1)]);


    std::vector<std::vector<double>> dZ_L = gradients["dZ" + std::to_string(L)];
    std::vector<std::vector<double>> transposedPrevLayerActivation = transpose(cache["A" + std::to_string(L - 1)]);

    std::vector<std::vector<double>> A_prev = cache["A" + std::to_string(L - 1)];
 
    int rows_A_prev = A_prev.size();       
    int cols_A_prev = A_prev[0].size();  
    int rows_dZ_L = dZ_L.size();       
    int cols_dZ_L = dZ_L[0].size();   

    
    std::vector<std::vector<double>> dW_L(rows_A_prev, std::vector<double>(cols_dZ_L, 0));
    std::vector<std::vector<double>> db_L(cols_dZ_L, std::vector<double>(1, 0));
    
    for (int i = 0; i < rows_A_prev; ++i) {
        for (int j = 0; j < cols_dZ_L; ++j) {
            for (int k = 0; k < cols_A_prev; ++k) {
                dW_L[i][j] += A_prev[i][k] * dZ_L[k][j];
            }
        }
    }
    for (int j = 0; j < cols_dZ_L; ++j) { 
        for (int i = 0; i < rows_dZ_L; ++i) { 
            db_L[j][0] += dZ_L[i][j];
        }
        db_L[j][0] /= rows_dZ_L; 
    }
    gradients["dW"+std::to_string(L)] = dW_L;
    gradients["db" + std::to_string(L)] = db_L;

    for (int l = L-1; l >= 0; --l) {
        std::vector<std::vector<double>> dA_next = gradients["dZ" + std::to_string(l+1)];
        std::vector<std::vector<double>> W_next = parameters["W" + std::to_string(l+1)];
        std::string activation = Activations[l];
        int rows = W_next[0].size(); 
        int cols = dA_next[0].size();

        std::vector<std::vector<double>> dA(rows, std::vector<double>(cols, 0));
        if(l!=0){
            dA = dotProduct(transpose(W_next), transpose(dA_next));
        } else {
            dA = dotProduct(transpose(W_next), (dA_next));
        }   
        gradients["dA"+std::to_string(l)]  = dA;
        std::vector<std::vector<double>> Z_prime = cache["Z"+std::to_string(l)];
        for(int i=0;i<Z_prime.size();i++){
            for(int j=0;j<Z_prime[0].size();j++){
                Z_prime[i][j] = derivativeActivation(Z_prime[i][j],activation=Activations[l]);
            }
        }
        std::vector<std::vector<double>> dZ(Z_prime.size(), std::vector<double>(Z_prime[0].size()));
        for(int i=0;i<dZ.size();i++){
            for(int j=0;j<dZ[0].size();j++){
                dZ[i][j] = Z_prime[i][j] * dA[i][j];
            }
        }
        gradients["dZ" + std::to_string(l)] = dZ;

        std::vector<std::vector<double>> A_prev;
        if (l == 0) {
            A_prev = X;
        } else {
            A_prev = cache["A" + std::to_string(l - 1)];
        }


        if(l!=0)
        {
            std::vector<std::vector<double>> dW = dotProduct(dZ, transpose(A_prev));
        for(int i=0;i<dW.size();i++){
            for(int j=0;j<dW[0].size();j++){
                dW[i][j] /= 1/m;
            }
        }
        gradients["dW" + std::to_string(l)] = dW;
        std::vector<std::vector<double>> db(parameters["b" + std::to_string(l)].size(), std::vector<double>(1, 0));
        
        for(int i=0;i<dZ.size();i++){
            double sum = 0;
            for(int j=0;j<dZ[0].size();j++){
                sum+=dZ[i][j];
            }
            sum *= 1/m;
            db[i][0] = sum;
        }
        gradients["db" + std::to_string(l)] = db;
        } 
        else 
        {
            std::vector<std::vector<double>> dW = dotProduct(dZ, A_prev); 
    for (int i = 0; i < dW.size(); ++i) {
        for (int j = 0; j < dW[0].size(); ++j) {
            dW[i][j] /= m;
        }
    }
    gradients["dW" + std::to_string(l)] = dW;

    std::vector<std::vector<double>> db(dZ.size(), std::vector<double>(1, 0));
    for (int i = 0; i < dZ.size(); i++) {
        double sum = 0;
        for (int j = 0; j < dZ[0].size(); j++) {
            sum += dZ[i][j];
        }
        db[i][0] = sum / m;
        }
        gradients["db" + std::to_string(l)] = db;
        }

    }
}
    void updateParams(double learningRate) {
    for (int l = 0; l < n_h; ++l) {
        std::vector<std::vector<double>>& W = parameters["W" + std::to_string(l)];
        std::vector<std::vector<double>>& b = parameters["b" + std::to_string(l)];
        const std::vector<std::vector<double>>& dW = gradients["dW" + std::to_string(l)];
        const std::vector<std::vector<double>>& db = gradients["db" + std::to_string(l)];

        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[i].size(); j++) {
                W[i][j] -= learningRate * dW[i][j];
            }
        }

        for (int i = 0; i < b.size(); i++) {
            b[i][0] -= learningRate * db[i][0];
        }
        parameters["W" + std::to_string(l)] = W;
        parameters["b" + std::to_string(l)] = b;
    }
}
};


std::vector<std::vector<double>> readCSV(std::string filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file!" << std::endl;
        return data;
    }

    std::string line;
    getline(file, line); 

    while (getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cout << "Warning: Non-numeric value found: '" << value << "' in line: " << line << std::endl;
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

double RMSE(const std::vector<std::vector<double>>& yPred, const std::vector<double>& yTrue) {
    double loss = 0;
    for (int i = 0; i < yPred.size(); i++) {
        loss += pow((yPred[i][0] - yTrue[i]), 2);
    }
    return sqrt(loss / yPred.size());
}

double generateUniform(double mean = 0, double std = 1) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}


int main() {
        std::vector<std::vector<double>> data = readCSV("C:\\Users\\hhhhhasdfsadf\\Downloads\\Amazon prices\\AMZN.csv");
    for (int i = 0; i < data.size(); i++) {
        data[i].insert(data[i].begin(), i);
    }

    int train_size = static_cast<int>(0.8 * data.size());
    std::vector<std::vector<double>> X_train(data.begin(), data.begin() + train_size);
    std::vector<std::vector<double>> X_test(data.begin() + train_size, data.end());

    std::vector<double> Y_train, Y_test;
    for (int i = 0; i < X_train.size(); i++) {
        Y_train.push_back(X_train[i].back());  
        X_train[i].pop_back();                 
    }
    for (int i = 0; i < X_test.size(); i++) {
        Y_test.push_back(X_test[i].back());  
        X_test[i].pop_back();                
    }

    int n_x = X_train[0].size(); 
    int n_m = X_train.size();    
    int n_h = 3;                  
    int n_y = 1;                  
    std::vector<std::string> Activations = {"relu","relu","linear"};  
    std::vector<int> noOfNeurons = {64,32,1};
    NeuralNetwork nn1(n_x, n_m, n_h, n_y, Activations);
    std::cout << "Constructed neural network " << std::endl;
    nn1.setInputData(X_train);  
    std::cout << "Input data has been set " << std::endl;
    nn1.declareParameters(noOfNeurons); 
    std::cout << "Parameters have been declared " << std::endl;
    nn1.initializeParameters(); 
    std::cout << "Parameters have been initialized " << std::endl;

    int epochs = 20;
    double learningRate = 0.01;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::vector<std::vector<double>> predictions = nn1.computeY();

        std::vector<std::vector<double>> Y_train_matrix(Y_train.size(), std::vector<double>(1));
        for (int i = 0; i < Y_train.size(); i++) {
            Y_train_matrix[i][0] = Y_train[i];
        }
        double loss = RMSE(predictions, Y_train);

        if (epoch % 1 == 0 || epoch == 1) {
            std::cout << std::fixed << std::setprecision(10)  << "Epoch " << epoch << ", Training Loss (RMSE): " << loss << std::endl << std::setprecision(10);
        }

        nn1.backProp(Y_train_matrix);
        std::cout << "Computed gradients for backpropagation" << std::endl;
        nn1.updateParams(learningRate);
        
        std::cout << " Updated parameters W and b" << std::endl;
    }
}


