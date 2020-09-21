#include <iostream>
#include <vector>
#include <set>
#include <math.h>
#include <limits>
#include <string>
#include <fstream>

std::vector<std::string> split_string(std::string);

// Complete the sockMerchant function below.
int sockMerchant(int n, std::vector<int> ar) {
    std::set<int> unique;
    for (int x : ar) { 
        unique.insert(x); 
    }
    int result;
    for(auto it = unique.begin(); it != unique.end(); ++it){
        int count = 0;
        for(int j = 0; j < n; j++){
            if(*it == ar[j]){
                count++;
            }
        }
        if(count > 1){
            result += floor(count/2);
        }
    }
    return result;
}

int main() {
    std::ofstream fout(getenv("OUTPUT_PATH"));

    int n;
    std::cin >> n;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string ar_temp_temp;
    std::getline(std::cin, ar_temp_temp);

    std::vector<std::string> ar_temp = split_string(ar_temp_temp);

    std::vector<int> ar(n);

    for (int i = 0; i < n; i++) {
        int ar_item = stoi(ar_temp[i]);

        ar[i] = ar_item;
    }

    int result = sockMerchant(n, ar);

    fout << result << "\n";

    fout.close();

    return 0;
}