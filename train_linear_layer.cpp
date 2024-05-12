#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include "layers.h"



using namespace std;

int main()
{
    srand(time(NULL));
    ifstream fin;
    string line, word;
    vector<vector<float>> data;
    vector<float> row;

    fin.open("../heart_disease.csv");

    // Execute a loop until EOF (End of File)
    getline(fin, line);
    // cout << line;
    while (getline(fin, line)) {
        row.clear();
        stringstream s(line);

        while (getline(s, word, ',')) {
            row.push_back(stoi(word));
        }
        data.push_back(row);
    }
    // Close the file
    fin.close();

    // need to random shuffle to split in validation and train
    auto rng = std::default_random_engine {};
    std::shuffle(data.begin(), data.end(), rng);

    // Split to train and eval
    float trainValSplit = 0.8;
    vector<vector<float>> train = vector<vector<float>>(data.begin(), data.begin() + (int)data.size() * trainValSplit);
    // split to train_x and train_y
    vector<vector<float>> trainX;
    vector<float> norm_mu(13, 0);
    vector<float> norm_sigma(13, 0);
    vector<float> trainXrow;
    vector<float> trainY;
    for (auto row: train)
    {
        trainXrow.clear();
        for (int i = 0; i < row.size() - 1; ++i)
        {
            trainXrow.push_back(row[i]);
            norm_mu[i] += row[i] / train.size();
        }
        trainX.push_back(trainXrow);
        trainY.push_back(row.back());
    }

    // Normalization
    // calculate mu and sigma
    for (auto row: train)
    {
        trainXrow.clear();
        for (int i = 0; i < row.size() - 1; ++i)
        {
            trainXrow.push_back(row[i]);
            norm_sigma[i] += (row[i] - norm_mu[i]) * (row[i] - norm_mu[i]) / train.size();
        }
        trainX.push_back(trainXrow);
        trainY.push_back(row.back());
    }
    for (int i = 0; i < norm_sigma.size(); ++i)
    {
        norm_sigma[i] = sqrt(norm_sigma[i]);
        cout << norm_mu[i] << ", " << norm_sigma[i] << " ";
    }
    cout << "\n";
    cout << "TrainX size = " << trainX.size() << " TrainY size = " << trainY.size() << "\n";

    // same for validation
    vector<vector<float>> val = vector<vector<float>>(data.begin() + (int)data.size() * trainValSplit, data.end());
    vector<vector<float>> valX;
    vector<float> valXrow;
    vector<float> valY;
    for (auto row: val)
    {
        valXrow.clear();
        for (int i = 0; i < row.size() - 1; ++i)
        {
            valXrow.push_back(row[i]);
        }
        valX.push_back(valXrow);
        valY.push_back(row.back());
    }
    cout << "valX size = " << valX.size() << " valY size = " << valY.size() << "\n";


    //Normalize inputs
    for (int i = 0; i < trainX.size(); ++i)
    {
        for (int j = 0; j < trainX[0].size(); ++j)
        {
            trainX[i][j] = (trainX[i][j] - norm_mu[j]) / norm_sigma[j];
        }
    }
    for (int i = 0; i < valX.size(); ++i)
    {
        for (int j = 0; j < valX[0].size(); ++j)
        {
            valX[i][j] = (valX[i][j] - norm_mu[j]) / norm_sigma[j];
        }
    }


    // for (auto it1: data) {
    //     for (auto it2: it1) {
    //         cout << it2 << " ";
    //     }
    //     cout << "\n";
    // }

    // Initialize dnn
    LinearLayer<2> L1(13, 5);
    LinearLayer<2> L2(5, 1);
    Sigmoid<2> S, S2;
    CrossEntropyLoss<2> LossFunc;
    Matrix3d<2> x, y1, y2, y3, y4, target, loss, grad_loss, grad_sigm, grad_l2, grad_sigm2;
    int batch_size = 16;

    // Measure accuracy before train
    float true_preds = 0;
    for (int i = 0; i < trainX.size(); i += batch_size)
    {
        x = Matrix3d<2>(vector<vector<float>>(trainX.begin() + i, min(trainX.end(), trainX.begin() + (i + batch_size))));
        target = Matrix3d<2>(vector<float>(trainY.begin() + i, min(trainY.end(), trainY.begin() + (i + batch_size))), {min(batch_size, (int)trainX.size() - i), 1});

        // grads not accumulated so in next call they will be rewrited
        // target.printMatrix();
        // y2 = S.forward(L1.forward(x));
        y2 = S2.forward(L2.forward(S.forward(L1.forward(x))));
        // cout << target.N << " " << target.M << "\n";
        // cout << "Preds = \n";
        // y2.printMatrix();
        // cout << "Targets = \n";
        // target.printMatrix();
        for (int j = 0; j < target.shape[0]; ++j)
        {
            true_preds += abs(y2(0, j) - target(0, j)) < 0.5;
        }
        // cout << "true_preds = " << true_preds << "\n";
    }
    cout << "Train accuracy before train = " << true_preds / trainX.size() << "\n";

    true_preds = 0;
    for (int i = 0; i < valX.size(); i += batch_size)
    {
        x = Matrix3d<2>(vector<vector<float>>(valX.begin() + i, min(valX.end(), valX.begin() + (i + batch_size))));
        target = Matrix3d<2>(vector<float>(valY.begin() + i, min(valY.end(), valY.begin() + (i + batch_size))), {min(batch_size, (int)valX.size() - i), 1});

        // grads not accumulated so in next call they will be rewrited
        // target.printMatrix();
        // y2 = S.forward(L1.forward(x));
        y2 = S2.forward(L2.forward(S.forward(L1.forward(x))));
        // cout << target.N << " " << target.M << "\n";
        // cout << "Preds = \n";
        // y2.printMatrix();
        // cout << "Targets = \n";
        // target.printMatrix();
        for (int j = 0; j < target.shape[0]; ++j)
        {
            true_preds += abs(y2(0, j) - target(0, j)) < 0.5;
        }
        // cout << "true_preds = " << true_preds << "\n";
    }
    cout << "Val accuracy before train = " << true_preds / valX.size() << "\n";

    // training loop
    for (int epoch = 0; epoch < 100; ++epoch)
    {
        // cout << "Epoch = " << epoch << "\n";
        for (int i = 0; i < trainX.size(); i += batch_size)
        {
            x = Matrix3d<2>(vector<vector<float>>(trainX.begin() + i, min(trainX.end(), trainX.begin() + (i + batch_size))));
            target = Matrix3d<2>(vector<float>(trainY.begin() + i, min(trainY.end(), trainY.begin() + (i + batch_size))), {min(batch_size, (int)trainX.size() - i), 1});
            // x.printMatrix();
            y1 = L1.forward(x);
            // y1.printMatrix();
            y2 = S.forward(y1);
            y3 = L2.forward(y2);
            y4 = S2.forward(y3);
            // y2.printMatrix();
            // loss = LossFunc.forward(y2, target);
            loss = LossFunc.forward(y4, target);
            // cout << "Loss = " << loss.mean() << "\n";
            // cout << "Weights =  \n";
            // L1.print_weights();
            // cout << "biases =  \n";
            // L1.print_bias();
            // loss.printMatrix();
            // cout << "\n";

            // // gradient step
            // grad_loss = LossFunc.backward();
            // // grad_loss.printMatrix();
            // grad_sigm = S.backward(grad_loss);
            // // grad_sigm.printMatrix();
            // L1.backward(grad_sigm);

            // L1.gradient_step(-3e-5);

            // // gradient step 2 layers
            grad_loss = LossFunc.backward();
            // grad_loss.printMatrix();
            grad_sigm2 = S2.backward(grad_loss);
            // grad_sigm.printMatrix();
            grad_l2 = L2.backward(grad_sigm2);
            L2.gradient_step(-3e-4);

            grad_sigm = S.backward(grad_l2);
            L1.backward(grad_sigm);
            L1.gradient_step(-3e-4);
        }
    }

    // Measure accuracy after train
    true_preds = 0;
    for (int i = 0; i < trainX.size(); i += batch_size)
    {
        x = Matrix3d<2>(vector<vector<float>>(trainX.begin() + i, min(trainX.end(), trainX.begin() + (i + batch_size))));
        target = Matrix3d<2>(vector<float>(trainY.begin() + i, min(trainY.end(), trainY.begin() + (i + batch_size))), {min(batch_size, (int)trainX.size() - i), 1});

        // grads not accumulated so in next call they will be rewrited
        // target.printMatrix();
        // y2 = S.forward(L1.forward(x));
        y2 = S2.forward(L2.forward(S.forward(L1.forward(x))));
        // cout << target.N << " " << target.M << "\n";
        for (int j = 0; j < target.shape[0]; ++j)
        {
            true_preds += abs(y2(0, j) - target(0, j)) < 0.5;
        }
    }
    cout << "Train accuracy after train = " << true_preds / trainX.size() << "\n";

    true_preds = 0;
    for (int i = 0; i < valX.size(); i += batch_size)
    {
        x = Matrix3d<2>(vector<vector<float>>(valX.begin() + i, min(valX.end(), valX.begin() + (i + batch_size))));
        target = Matrix3d<2>(vector<float>(valY.begin() + i, min(valY.end(), valY.begin() + (i + batch_size))), {min(batch_size, (int)valX.size() - i), 1});

        // grads not accumulated so in next call they will be rewrited
        // y2 = S.forward(L1.forward(x));
        y2 = S2.forward(L2.forward(S.forward(L1.forward(x))));
        for (int j = 0; j < target.shape[0]; ++j)
        {
            true_preds += abs(y2(0, j) - target(0, j)) < 0.5;
        }
        // cout << "true_preds = " << true_preds << "\n";
    }
    cout << "Val accuracy after train = " << true_preds / valX.size() << "\n";

    return 0;
}