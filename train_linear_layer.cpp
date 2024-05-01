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
    ifstream fin;
    string line, word;
    vector<vector<float>> data;
    vector<float> row;

    fin.open("heart_disease.csv");

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
    vector<float> trainXrow;
    vector<float> trainY;
    for (auto row: train)
    {
        trainXrow.clear();
        for (int i = 0; i < row.size() - 1; ++i)
        {
            trainXrow.push_back(row[i]);
        }
        trainX.push_back(trainXrow);
        trainY.push_back(row.back());
    }

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

    // for (auto it1: data) {
    //     for (auto it2: it1) {
    //         cout << it2 << " ";
    //     }
    //     cout << "\n";
    // }

    // Initialize dnn
    LinearLayer L1(13, 1);
    Matrix x, y;

    // training loop
    short batch_size = 16;

    for (int i = 0; i < trainX.size(); i += batch_size)
    {
        x = Matrix(vector<vector<float>>(trainX.begin() + i, min(trainX.end(), trainX.begin() + (i + batch_size))));
        // x.printMatrix();
        y = L1.forward(x);
        y.printMatrix();
        cout << "\n";
    }

    return 0;
}