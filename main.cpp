#include <iostream>
#include <set>
#include <vector>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <cmath>
std::ifstream fin;

std::map<char, char> TOKENS;

void init_params_uniform(std::vector<std::vector<float>> &w,
                         float low = -1.0f, float high = 1.0f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);

    for (auto &row : w)
        for (auto &x : row)
            x = dist(gen);
}

std::vector<std::vector<float>> apply_sigmoid(const std::vector<std::vector<float>> &A)
{
    std::vector<std::vector<float>> R = A;
    for (auto &row : R)
        for (auto &x : row)
            x = 1.0f / (1.0f + std::exp(-x));
    return R;
}

std::vector<std::vector<float>> apply_tanh(const std::vector<std::vector<float>> &A)
{
    std::vector<std::vector<float>> R = A;
    for (auto &row : R)
        for (auto &x : row)
            x = std::tanh(x);
    return R;
}

std::vector<std::vector<float>> matadd(
    const std::vector<std::vector<float>> &A,
    const std::vector<std::vector<float>> &B)
{
    if (A.size() != B.size())
        throw std::invalid_argument("Bad dimensions for matadd");
    if (A.empty())
        return {};

    const size_t m = A.size();
    const size_t n = A[0].size();
    for (size_t i = 0; i < m; ++i)
    {
        if (A[i].size() != n || B[i].size() != n)
            throw std::invalid_argument("Bad dimensions for matadd");
    }

    std::vector<std::vector<float>> C(m, std::vector<float>(n));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];

    return C;
}

std::vector<std::vector<float>> matmul(
    const std::vector<std::vector<float>> &A,
    const std::vector<std::vector<float>> &B)
{
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty())
        throw std::invalid_argument("Empty matrix");

    const size_t m = A.size();
    const size_t n = A[0].size();
    const size_t n2 = B.size();
    const size_t p = B[0].size();

    for (const auto &row : A)
        if (row.size() != n)
            throw std::invalid_argument("A is not rectangular");
    for (const auto &row : B)
        if (row.size() != p)
            throw std::invalid_argument("B is not rectangular");

    if (n != n2)
        throw std::invalid_argument("Bad dimensions for matmul");

    std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0f));

    for (size_t i = 0; i < m; ++i)
        for (size_t k = 0; k < n; ++k)
        {
            const float aik = A[i][k];
            for (size_t j = 0; j < p; ++j)
                C[i][j] += aik * B[k][j];
        }

    return C;
}

int main()
{
    // dataset opening
    fin.open("input.txt");
    std::set<char> symbols;
    char tmp;
    while (fin.get(tmp))
        symbols.insert(tmp);
    std::cout << symbols.size() << '\n';
    float i = 0;
    for (const auto &sym : symbols)
    {
        TOKENS[sym] = i++;
        std::cout << sym << ' ';
    }
    fin.close();
    fin.open("input.txt");
    std::vector<std::vector<float>> batches;
    while (fin.get(tmp))
    {
        std::vector<float> batch(65, 0);
        batch[TOKENS[tmp]] = 1;
    }
    fin.close();

    // nn init
    std::vector<std::vector<float>> weights1(65, std::vector<float>(65));
    std::vector<std::vector<float>> weights2(65, std::vector<float>(65));

    std::vector<std::vector<float>> bias1(1, std::vector<float>(65));
    std::vector<std::vector<float>> bias2(1, std::vector<float>(65));

    std::vector<std::vector<float>> b(1, std::vector<float>(65, 0));

    init_params_uniform(weights1);
    init_params_uniform(weights2);

    init_params_uniform(bias1);
    init_params_uniform(bias2);

    // forward
    for (int i = 0; i < size(batches) - 1; ++i)
    {
        std::vector<std::vector<float>> x = {batches[i]};
        std::vector<std::vector<float>> y = {batches[i + 1]};
        auto N = matmul(x, weights1);
        auto sum = matadd(N, bias1);
        b = apply_tanh(sum);
    }
    return 0;
}