#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>

#define EMPTY_STRING ""
#define ZERO_PROB    "0.000000"
#define ID_LABEL     "id"
#define PRED_LABEL   "prediction"
#define PROB_LABEL   "logprob"
#define WHITESPACE   " \n\r\t\f\v"

std::string ltrim(const std::string& s)
{
	size_t start = s.find_first_not_of(WHITESPACE);
	return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string& s)
{
	size_t end = s.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string& s)
{
	return rtrim(ltrim(s));
}

void write_csv(std::string id, std::vector<std::pair<std::string, std::string>> data, std::string filename, std::string mode)
{
    std::ofstream targetFile;
    if (mode == "a")
        targetFile.open(filename, std::ios_base::app);
    else if (mode == "w")
    {
        // Requires Header initialization.
        targetFile.open(filename);
        targetFile << ID_LABEL;
        if (data.size() > 0)
            targetFile << ",";
        for(size_t j = 0; j < data.size(); ++j)
        {
            targetFile << "\"" << data.at(j).first << "\"";
            if (j != data.size() - 1)
                targetFile << ",";
        }
        targetFile << "\n";
    }
    else
    {
        std::cout << "Problem with mode. Assuming it is in append mode." << std::endl;
        targetFile.open(filename, std::ios_base::app);
    }

    targetFile << id;
    if (data.size() > 0)
        targetFile << ",";

    for(size_t i = 0; i < data.size(); ++i)
    {
        targetFile << "\"" << data.at(i).second << "\"";
        if (i != data.size() - 1)
            targetFile << ",";
    }
    targetFile << "\n";
    targetFile.close();
}

std::vector<std::string> split_string(const std::string& str,
                                      const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}


std::vector<std::pair<std::string, std::string>> parse_string(std::string stringToParse)
{
    std::string prob;
    std::string caption;
    std::vector<std::pair<std::string, std::string>> data;
    // Parse lines
    auto guessStrings = split_string(stringToParse, "\n");

    // Parse caption and probability
    for (size_t i = 0; i < guessStrings.size(); i++)
    {
        auto length = guessStrings.at(i).length();
        prob = guessStrings.at(i).substr(length - 9, 8);
        caption = rtrim(guessStrings.at(i).substr(5, length - 17));
        if (caption.substr(caption.length() - 1) == ".")
            caption = rtrim(caption.substr(0, caption.length() - 1));
        data.push_back({PRED_LABEL + std::to_string(i), caption});
        data.push_back({PROB_LABEL + std::to_string(i), prob});
    }
    return data;
}

int main(int argc, char const *argv[])
{
    if (argc != 5)
    {
        std::cout << "Please provide an id string, an output string, export file name and mode." << std::endl;
        return -1;
    }
    else
    {
        std::string resultID      = argv[1];
        std::string stringToParse = argv[2];
        std::string filename      = argv[3];
        std::string mode          = argv[4];

        std::cout << "Parsing started for " << resultID << "." << std::endl;
        auto data = parse_string(stringToParse);
        std::cout << "Parsing completed for " << resultID << "." << std::endl;
        std::cout << "Parsed data is being exported for " << resultID << "." << std::endl;
        write_csv(resultID, data, filename, mode);
        std::cout << "Parsed data is exported for " << resultID << "." << std::endl;
    }
    return 0;
}
