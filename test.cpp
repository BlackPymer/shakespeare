#include "common.cpp"
#include <vector>
#include <fstream>
#include <iostream>

char to_char(const std::vector<float> &token)
{
    if (token.empty())
        return '?';

    // Находим индекс максимального элемента
    int max_idx = std::max_element(token.begin(), token.end()) - token.begin();

    // Ищем символ с таким индексом
    for (const auto &pair : TOKENS)
    {
        if (static_cast<int>(pair.second) == max_idx)
        {
            return pair.first;
        }
    }
    return '?';
}
int main()
{
    // Загружаем токены ПЕРВЫМ делом
    load_tokens("input.txt");
    std::cout << "TOKENS size: " << TOKENS.size() << std::endl;

    // Инициализируем веса
    std::vector<std::vector<float>> weights1(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> weights2(65, std::vector<float>(65, 0));
    std::vector<std::vector<float>> bias1(1, std::vector<float>(65, 0));
    std::vector<std::vector<float>> bias2(1, std::vector<float>(65, 0));

    // Загружаем или инициализируем случайными значениями
    if (!load_vector(weights1, W1_FILENAME))
    {
        std::cout << "Initializing W1 with random values" << std::endl;
        init_params_uniform(weights1);
    }
    if (!load_vector(weights2, W2_FILENAME))
    {
        std::cout << "Initializing W2 with random values" << std::endl;
        init_params_uniform(weights2);
    }
    if (!load_vector(bias1, B1_FILENAME))
    {
        std::cout << "Initializing B1 with random values" << std::endl;
        init_params_uniform(bias1);
    }
    if (!load_vector(bias2, B2_FILENAME))
    {
        std::cout << "Initializing B2 with random values" << std::endl;
        init_params_uniform(bias2);
    }

    // bias можно оставить нулевыми

    // Получаем начальное слово
    std::string start_word;
    std::cout << "Enter start word: ";
    std::cin >> start_word;

    // Преобразуем в токены
    std::vector<std::vector<float>> batches;
    for (const auto &ch : start_word)
    {
        // Проверяем, что символ есть в TOKENS
        if (TOKENS.find(ch) != TOKENS.end())
        {
            batches.push_back(to_token(ch));
            std::cout << "Added token for char: " << ch << std::endl;
        }
        else
        {
            std::cout << "Warning: character '" << ch << "' not in vocabulary, skipping" << std::endl;
        }
    }

    if (batches.empty())
    {
        std::cerr << "No valid characters in start word!" << std::endl;
        return 1;
    }

    // Инициализируем outputs
    std::vector<SymbolOutputs> outputs(backwardRate);
    for (auto &out : outputs)
    {
        out.linearOutput = {};
        out.thOutput = {};
        out.linearOutput2 = {};
        out.softmaxOutput = {};
    }

    // Пропускаем forward для начальных символов
    std::cout << "Processing initial sequence..." << std::endl;
    for (int i = 0; i < batches.size(); ++i)
    {
        forward(weights1, bias1, weights2, bias2, batches, i, outputs);
        std::cout << "Processed char " << i << std::endl;
    }

    // Генерация новых символов
    std::cout << "\nGenerated text: " << start_word;
    std::flush(std::cout);

    for (int i = batches.size(); i < 2000; ++i)
    {
        int last_idx = (i - 1) % backwardRate;

        if (outputs[last_idx].softmaxOutput.empty() ||
            outputs[last_idx].softmaxOutput[0].empty())
        {
            std::cerr << "\nError: empty softmax output at step " << i << std::endl;
            break;
        }

        const auto &softmax_output = outputs[last_idx].softmaxOutput[0];

        // Отладочный вывод softmax
        /*
        if (i < batches.size() + 10)
        {
            std::cout << "\nSoftmax output: ";
            for (int k = 0; k < 5; ++k)
            {
                std::cout << softmax_output[k] << " ";
            }
            std::cout << std::endl;
        }
        */
        char predicted_char = to_char(softmax_output);
        std::cout << predicted_char;
        // std::flush(std::cout);

        batches.push_back(to_token(predicted_char));
        forward(weights1, bias1, weights2, bias2, batches, batches.size() - 1, outputs);
    }

    std::cout << std::endl;
    return 0;
}