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

#define backwardRate 10
#define learningRate 0.01f

const char *W1_FILENAME = "w1.bin";
const char *W2_FILENAME = "w2.bin";
const char *B1_FILENAME = "b1.bin";
const char *B2_FILENAME = "b2.bin";
const char *W3_FILENAME = "w3.bin";
const char *B3_FILENAME = "b3.bin";
const char *HUYURUS1_FILENAME = "huyurus1.bin";
const char *HUYURUS2_FILENAME = "huyurus2.bin";
const char *HUYURUS3_FILENAME = "huyurus3.bin";

std::map<char, char> TOKENS;

struct SymbolOutputs
{
    std::vector<std::vector<float>> rt;
    std::vector<std::vector<float>> zt;
    std::vector<std::vector<float>> ht;
    std::vector<std::vector<float>> output;
    std::vector<std::vector<float>> softmaxOutput;
};
bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}
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

std::vector<std::vector<float>> mult(const std::vector<std::vector<float>> &A, float num)
{
    std::vector<std::vector<float>> res = A;
    for (auto &row : res)
    {
        for (float &el : row)
        {
            el *= num;
        }
    }
    return res;
}

std::vector<std::vector<float>> matadd(
    const std::vector<std::vector<float>> &A,
    const std::vector<std::vector<float>> &B)
{
    if (A.size() != B.size())
    {
        std::cout << A.size() << " not equals to " << B.size() << "   0\n";
        throw std::invalid_argument("Bad dimensions for matadd");
    }
    if (A.empty())
        return {};

    const size_t m = A.size();
    const size_t n = A[0].size();
    for (size_t i = 0; i < m; ++i)
    {
        if (A[i].size() != n || B[i].size() != n)
        {
            std::cout << A.size() << " not equals to " << B.size() << "   1\n";
            throw std::invalid_argument("Bad dimensions for matadd");
        }
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
    if (matrix.empty() || matrix[0].empty())
    {
        return {};
    }
    std::vector<std::vector<float>> res(matrix[0].size(), std::vector<float>(matrix.size()));
    for (size_t i = 0; i < matrix[0].size(); ++i)
    {
        for (size_t j = 0; j < matrix.size(); ++j)
        {
            res[i][j] = matrix[j][i];
        }
    }
    return res;
}

std::vector<std::vector<float>> hadamard(
    const std::vector<std::vector<float>> &A,
    const std::vector<std::vector<float>> &B)
{
    if (A.size() != B.size())
        throw std::invalid_argument("hadamard: bad batch size");
    std::vector<std::vector<float>> R = A;
    for (size_t i = 0; i < R.size(); ++i)
    {
        if (R[i].size() != B[i].size())
            throw std::invalid_argument("hadamard: bad row size");
        for (size_t j = 0; j < R[i].size(); ++j)
            R[i][j] *= B[i][j];
    }
    return R;
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

std::vector<std::vector<float>> diag(const std::vector<std::vector<float>> &v)
{
    if (v.empty() || v[0].empty())
        return {};

    const size_t n = v[0].size();
    std::vector<std::vector<float>> D(n, std::vector<float>(n, 0.0f));

    for (size_t i = 0; i < n; ++i)
    {
        D[i][i] = v[i][i];
    }
    return D;
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

bool load_vector(std::vector<std::vector<float>> &vec, const std::string &filename)
{
    if (!file_exists(filename))
    {
        std::cerr << "File '" << filename << "' not found, keeping zero vector" << std::endl;
        return false;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open '" << filename << "' for reading" << std::endl;
        return false;
    }

    size_t rows = 0;
    size_t cols = 0;

    if (!file.read(reinterpret_cast<char *>(&rows), sizeof(rows)))
    {
        std::cerr << "Failed to read rows from '" << filename << "'" << std::endl;
        return false;
    }

    vec.clear();
    vec.reserve(rows);

    for (size_t i = 0; i < rows; i++)
    {
        if (!file.read(reinterpret_cast<char *>(&cols), sizeof(cols)))
        {
            std::cerr << "Failed to read cols from '" << filename << "' at row " << i << std::endl;
            return false;
        }

        std::vector<float> row(cols);
        if (!file.read(reinterpret_cast<char *>(row.data()), cols * sizeof(float)))
        {
            std::cerr << "Failed to read data from '" << filename << "' at row " << i << std::endl;
            return false;
        }

        vec.push_back(std::move(row));
    }

    std::cout << "Successfully loaded '" << filename << "': "
              << rows << "x" << (vec.empty() ? 0 : vec[0].size()) << std::endl;
    return true;
}

bool save_vector(const std::vector<std::vector<float>> &vec, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open '" << filename << "' for writing" << std::endl;
        return false;
    }

    size_t rows = vec.size();
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

    for (const auto &row : vec)
    {
        size_t cols = row.size();
        file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        file.write(reinterpret_cast<const char *>(row.data()), cols * sizeof(float));
    }

    std::cout << "Successfully saved '" << filename << "': "
              << rows << "x" << (vec.empty() ? 0 : vec[0].size()) << std::endl;
    return file.good();
}

void forward(
    const std::vector<std::vector<float>> &weights1,
    const std::vector<std::vector<float>> &bias1,
    const std::vector<std::vector<float>> &weights2,
    const std::vector<std::vector<float>> &bias2,
    const std::vector<std::vector<float>> &weights3,
    const std::vector<std::vector<float>> &bias3,
    const std::vector<std::vector<float>> &huyurus1,
    const std::vector<std::vector<float>> &huyurus2,
    const std::vector<std::vector<float>> &huyurus3,
    const std::vector<std::vector<float>> &batches,
    int i,
    std::vector<SymbolOutputs> &outputs)
{

    std::vector<std::vector<float>> x;
    x.push_back(batches[i]);
    SymbolOutputs output;
    std::vector<std::vector<float>> prev_output;
    if (i == 0)
        prev_output = std::vector<std::vector<float>>(1, std::vector<float>(65, 0));
    else
        prev_output = outputs[(i - 1) % backwardRate].output;
    output.rt = apply_sigmoid(matadd(matadd(matmul(x, weights1), bias1), matmul(prev_output, huyurus1)));
    output.zt = apply_sigmoid(matadd(matadd(matmul(x, weights2), bias2), matmul(prev_output, huyurus2)));
    output.ht = apply_tanh(matadd(matadd(matmul(x, weights3), bias3), matmul(hadamard(output.rt, prev_output), huyurus3)));

    std::vector<std::vector<float>> ones(1, std::vector<float>(output.zt[0].size(), 1.0f));

    output.output = matadd(matmul(matadd(mult(output.zt, -1.0f), ones), prev_output), mult(hadamard(output.ht, output.zt), 1.0f));
    output.softmaxOutput = softmax(output.output);
    outputs[i % backwardRate] = output;
}
void backward(bool shouldPrint, const std::vector<std::vector<float>> &batches,
              std::vector<std::vector<float>> &weights1,
              std::vector<std::vector<float>> &weights2,
              std::vector<std::vector<float>> &weights3,
              std::vector<std::vector<float>> &bias1,
              std::vector<std::vector<float>> &bias2,
              std::vector<std::vector<float>> &bias3,
              std::vector<std::vector<float>> &huyurus1,
              std::vector<std::vector<float>> &huyurus2,
              std::vector<std::vector<float>> &huyurus3,
              std::vector<SymbolOutputs> &outputs, int i)
{
    float avgLoss = 0;

    std::vector<std::vector<float>> b(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dw2(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dw1(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> db2(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> db1(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dw3(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> db3(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dhuyurus1(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dhuyurus2(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> dhuyurus3(65, std::vector<float>(65, 0));
    SymbolOutputs output;
    for (int j = i; j > i - backwardRate; --j)
    {
        std::vector<std::vector<float>> prev_output;
        if (j == 0)
            prev_output = std::vector<std::vector<float>>(1, std::vector<float>(65, 0));
        else
            prev_output = outputs[(i - 1) % backwardRate].output;
        int output_idx = j % backwardRate;
        output = outputs[output_idx];
        float loss = softmax_cross_entropy_loss_onehot(output.softmaxOutput, {batches[j + 1]});
        avgLoss += loss / (backwardRate);
        std::vector<std::vector<float>> doutput = matadd(softmax_cross_entropy_grad(output.softmaxOutput, {batches[j + 1]}), b);

        std::vector<std::vector<float>> ones(1, std::vector<float>(output.ht[0].size(), 1.0f));

        std::vector<std::vector<float>> doutput_condidate = hadamard(hadamard(output.zt, doutput), matadd(ones, mult(hadamard(output.ht, output.ht), -1.0f)));
        db3 = matadd(db3, doutput_condidate);
        dw3 = matadd(dw3, matmul(transpose({batches[j]}), doutput_condidate));
        dhuyurus3 = matadd(dhuyurus3, matmul(transpose(hadamard(output.rt, prev_output)), doutput_condidate));
    }
    weights1 = matadd(weights1, mult(dw1, -learningRate));
    weights2 = matadd(weights2, mult(dw2, -learningRate));
    weights3 = matadd(weights3, mult(dw3, -learningRate));
    huyurus1 = matadd(huyurus1, mult(dhuyurus1, -learningRate));
    huyurus2 = matadd(huyurus2, mult(dhuyurus2, -learningRate));
    huyurus3 = matadd(huyurus3, mult(dhuyurus3, -learningRate));
    bias1 = matadd(bias1, mult(db1, -learningRate));
    bias2 = matadd(bias2, mult(db2, -learningRate));
    bias3 = matadd(bias3, mult(db3, -learningRate));
    if (shouldPrint)
        std::cout << i << '\t' << avgLoss << '\n';
}
void load_tokens(const std::string &filename)
{

    std::ifstream fin;
    fin.open("input.txt");
    std::set<char> symbols;
    char tmp;
    while (fin.get(tmp))
        symbols.insert(tmp);
    std::cout << symbols.size() << '\n';
    {
        float i = 0;
        for (const auto &sym : symbols)
        {
            TOKENS[sym] = i++;
            std::cout << sym << ' ';
        }
        fin.close();
    }
}
std::vector<float> to_token(char symbol)
{
    std::vector<float> vec(65, 0);
    vec[TOKENS[symbol]] = 1.0f;
    return vec;
}