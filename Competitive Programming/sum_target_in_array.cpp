#include <iostream>
#include <vector>
#include <unordered_map>

bool two_sum(std::vector<int>& arr, int target) {
    //your code here
    std::unordered_map<int, std::vector<int>> sums;
    for(auto itr = arr.begin(); itr != arr.end(); ++itr){
        std::vector<int> pairs;
        for(auto it2 = ++arr.begin(); it2 != arr.end(); ++it2){
            pairs.push_back(*itr + *it2);
        }
        sums[*itr] = pairs;
        pairs.clear();
    }
    for(auto it1 = sums.begin(); it1 != sums.end(); ++it1){
        for(auto it2 = it1->second.begin(); it2 != it1->second.end(); ++it2){
            if(*it2 == target){
                return true;
            }
        }
    }
    return false;
}