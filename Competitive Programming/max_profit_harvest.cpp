#include<iostream>
#include<vector>
#include <algorithm>
#include <math.h>

long maxProfit(int k, std::vector<int> profit) {
    if(k > profit.size()/2){
        return 0;
    }
    std::vector<long> result;
    long temp = 0;
    for(int i = 0; i + k -1 < profit.size()/2; i++){
        int j  = 0;
        int it = i;
        while(j < k){
            temp += profit[it];
            temp += profit[it + profit.size()/2];
            ++j;
            ++it;
        }
        std::cout << temp << '\n';
        result.push_back(temp);
        temp = 0;
    }
    return *std::max_element(result.begin(), result.end());
}

int main(){
    std::vector<int> prof = {2, 5, 8, 9, 0, 1, -3, -7, -4, 6};
    std::cout << maxProfit(5, prof) << '\n';
    return 0;
}