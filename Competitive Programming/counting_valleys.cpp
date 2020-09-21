#include <iostream>
#include <string>
#include <limits>
#include <fstream>

// Complete the countingValleys function below.
int countingValleys(int n, std::string s) {
    int elevation = 0;
    int valleys = 0;
    for(int i = 0; i < n; i++){
        if(elevation == 0 && s[i] == 'D'){
            valleys += 1;
        }
        if (s[i] == 'U'){
            elevation += 1;
        }
        if (s[i] == 'D'){
            elevation += -1;
        }  
    }
    return valleys;
}

int main(){
    std::ofstream fout(getenv("OUTPUT_PATH"));

    int n;
    std::cin >> n;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string s;
    std::getline(std::cin, s);

    int result = countingValleys(n, s);

    fout << result << "\n";

    fout.close();

    return 0;
}