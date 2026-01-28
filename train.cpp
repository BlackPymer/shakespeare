#include <iostream>
#include <set>
#include <vector>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include "common.cpp"

std::ifstream fin;

int main()
{
    // dataset opening
    fin.open("input.txt");
    std::vector<std::vector<float>> batches;
    char tmp;
    load_tokens("input.txt");
    while (fin.get(tmp))
    {
        std::vector<float> batch = to_token(tmp);
        batches.push_back(batch);
    }
    fin.close();

    // nn init
    std::vector<std::vector<float>> weights1(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> weights2(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> bias1(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> bias2(1, std::vector<float>(65, 0));

    load_vector(weights1, W1_FILENAME);
    load_vector(weights2, W2_FILENAME);
    load_vector(bias1, B1_FILENAME);
    load_vector(bias2, B2_FILENAME);

    init_params_uniform(weights1);
    init_params_uniform(weights2);

    init_params_uniform(bias1);
    init_params_uniform(bias2);

    // forward
    std::vector<SymbolOutputs> outputs(backwardRate);
    for (int i = 0; i < size(batches) - 1; ++i)
    {
        forward(weights1, bias1, weights2, bias2, batches, i, outputs);
        if (i % backwardRate == backwardRate - 1)
            backward(i % 10000 == backwardRate - 1, batches, weights1, weights2, bias1, bias2, outputs, i);
    }
    save_vector(weights1, W1_FILENAME);
    save_vector(weights2, W2_FILENAME);
    save_vector(bias1, B1_FILENAME);
    save_vector(bias2, B2_FILENAME);
    return 0;
}