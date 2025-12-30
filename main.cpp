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

#pragma region constants

#define backwardRate 50

#pragma endregion

std::ifstream fin;

std::map<char, char> TOKENS;

struct SymbolOutputs
{
    std::vector<std::vector<float>> linearOutput;
    std::vector<std::vector<float>> thOutput;
    std::vector<std::vector<float>> linearOutput2;
    std::vector<std::vector<float>> softmaxOutput;
};

#pragma region Forward functions
std::vector<std::vector<float>> softmax(
    const std::vector<std::vector<float>> &xs)
{
    std::vector<std::vector<float>> out;
    out.reserve(xs.size());

    for (const auto &row : xs)
    {
        if (row.empty())
        {
            out.emplace_back();
            continue;
        }

        const float m = *std::max_element(row.begin(), row.end());

        std::vector<float> probs(row.size());
        float sum = 0.0f;

        for (size_t i = 0; i < row.size(); ++i)
        {
            const float e = std::exp(row[i] - m);
            probs[i] = e;
            sum += e;
        }

        if (sum == 0.0f || !std::isfinite(sum))
        {
            const float uniform = 1.0f / static_cast<float>(row.size());
            std::fill(probs.begin(), probs.end(), uniform);
        }
        else
        {
            const float inv = 1.0f / sum;
            for (float &v : probs)
                v *= inv;
        }

        out.emplace_back(std::move(probs));
    }

    return out;
}

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

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix)
{
    std::vector<std::vector<float>> res(matrix[0].size(), std::vector<float>(matrix.size()));
    for (int i = 0; i < matrix[0].size(); ++i)
    {
        for (int j = 0; j < matrix.size(); ++j)
        {
            res[i][j] = matrix[j][i];
        }
    }
    return res;
}
#pragma endregion

#pragma region Backward functions
std::vector<std::vector<float>> sigmoid_derivative(
    const std::vector<std::vector<float>> &Y)
{
    std::vector<std::vector<float>> R = Y;
    for (auto &row : R)
        for (auto &y : row)
            y = y * (1.0f - y);
    return R;
}

std::vector<std::vector<float>> tanh_derivative(
    const std::vector<std::vector<float>> &Y)
{
    std::vector<std::vector<float>> R = Y;
    for (auto &row : R)
        for (auto &y : row)
            y = 1.0f - y * y;
    return R;
}

std::vector<std::vector<std::vector<float>>> softmax_derivative(
    const std::vector<std::vector<float>> &Y)
{
    std::vector<std::vector<std::vector<float>>> J;
    J.reserve(Y.size());

    for (const auto &y : Y)
    {
        const size_t n = y.size();
        std::vector<std::vector<float>> Jy(n, std::vector<float>(n, 0.0f));

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                Jy[i][j] = (i == j) ? (y[i] * (1.0f - y[i])) : (-y[i] * y[j]);
            }
        }
        J.emplace_back(std::move(Jy));
    }
    return J;
}

std::vector<std::vector<float>> softmax_cross_entropy_grad(
    const std::vector<std::vector<float>> &P,
    const std::vector<std::vector<float>> &Y)
{
    if (P.size() != Y.size())
        throw std::invalid_argument("P/Y batch mismatch");
    if (P.empty())
        return {};

    std::vector<std::vector<float>> dZ = P;

    for (size_t i = 0; i < dZ.size(); ++i)
    {
        if (dZ[i].size() != Y[i].size())
            throw std::invalid_argument("P/Y class mismatch");

        for (size_t j = 0; j < dZ[i].size(); ++j)
        {
            dZ[i][j] -= Y[i][j];
        }
    }
    return dZ;
}

float softmax_cross_entropy_loss_onehot(
    const std::vector<std::vector<float>> &P,
    const std::vector<std::vector<float>> &Y,
    float eps = 1e-12f)
{
    if (P.size() != Y.size())
        throw std::invalid_argument("P/Y batch mismatch");
    if (P.empty())
        return 0.0f;

    float loss_sum = 0.0f;
    for (size_t i = 0; i < P.size(); ++i)
    {
        if (P[i].size() != Y[i].size())
            throw std::invalid_argument("P/Y class mismatch");

        float li = 0.0f;
        for (size_t j = 0; j < P[i].size(); ++j)
        {
            const float pj = std::max(P[i][j], eps);
            li += -Y[i][j] * std::log(pj);
        }
        loss_sum += li;
    }
    return loss_sum / static_cast<float>(P.size());
}
#pragma endregion

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

    init_params_uniform(weights1);
    init_params_uniform(weights2);

    init_params_uniform(bias1);
    init_params_uniform(bias2);

    // forward

    std::vector<SymbolOutputs> outputs(backwardRate);
    float avgLoss = 0;
    for (int i = 0; i < size(batches) - 1; ++i)
    {
        std::vector<std::vector<float>> x = {batches[i]};
        SymbolOutputs output;
        auto N = matmul(x, weights1);
        output.linearOutput = matadd(N, bias1);
        output.thOutput = apply_tanh(output.linearOutput);

        output.linearOutput2 = matadd(matmul(output.thOutput, weights2), bias2);
        output.softmaxOutput = softmax(output.linearOutput2);

        outputs[i % backwardRate] = output;
        if (i % backwardRate == backwardRate - 1)
        {
            std::vector<std::vector<float>> b;
            std::vector<std::vector<float>> dw2;
            std::vector<std::vector<float>> dw1;
            for (int j = i; j > i - backwardRate; --j)
            {
                float loss = softmax_cross_entropy_loss_onehot(output.softmaxOutput, {batches[j + 1]});
                avgLoss += loss / (size(batches) / backwardRate);

                auto softmax_grad = softmax_cross_entropy_grad(output.softmaxOutput, {batches[j + 1]});
                dw2 = matadd(matmul(transpose(output.linearOutput2), softmax_grad), dw2);
            }
        }
    }
    return 0;
}