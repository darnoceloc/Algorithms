#include <iostream>
#include <vector>
#include <math.h>
#include <limits>
#include <string>
#include <fstream>

std::vector<std::string> split_string(std::string);

// Complete the jumpingOnClouds function below.
int jumpingOnClouds(std::vector<int> c) {
    int jumps = 0;
    for (int i = 0; i < c.size(); i++){
        if(c[i+2] == 0 && i+2 < c.size()){
            jumps += 1;
            i += 1;
            continue;
        }
        if(c[i+1] == 0 && i+1 < c.size()){
            jumps += 1;
            continue;
        } 
    }
    return jumps;
}

int main()
{
    std::ofstream fout(getenv("OUTPUT_PATH"));

    int n;
    std::cin >> n;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string c_temp_temp;
    std::getline(std::cin, c_temp_temp);

    std::vector<std::string> c_temp = split_string(c_temp_temp);

    std::vector<int> c(n);

    for (int i = 0; i < n; i++) {
        int c_item = stoi(c_temp[i]);

        c[i] = c_item;
    }

    int result = jumpingOnClouds(c);

    fout << result << "\n";

    fout.close();

    return 0;
}