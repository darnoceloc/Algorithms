#include <bits/stdc++.h>

void checkMagazine(std::vector<std::string> magazine, std::vector<std::string> note) {
    std::unordered_map<std::string, int> words;
    for (auto &it: magazine){
        words[it]++;
    }
    for (auto &it: note) {
        if (words[it] > 0){
            words[it]--;
            continue;
        }
        if (words[it] == 0){
            std::cout << "No" << std::endl;
            return;
        }
    }
    std::cout << "Yes" << std::endl;
    return;
}