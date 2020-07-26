#include <iostream>
#include <string>
#include <stack>
#include <algorithm>

const int MAX_LENGTH = 20;

bool isInLanguage(char* theString){
    
    std::string w = std::string(theString);
    std::cout << w[0] << '\n';

    static std::stack<char> charStack;

    if (w.length() % 2 != 0){
        std::cout << 1 << '\n';
        std::cout <<"False, pattern entered does not match the language " << '\n';
        return false;
    }
    
    charStack.push(w[0]);
    int i = 1;
    while(i < w.length()){
        if (charStack.size() >= 0){
            if (w[i] == 'A' ||'B'){            
                if (w[0] == 'A') {
                    if (w[i] == 'A'){
                        charStack.push(w[i]);
                        ++i;
                    }
                    if (w[i] == 'B'){
                        charStack.pop();
                        ++i;
                    }
                }
                if (w[0] == 'B') {
                    if (w[i] == 'B'){
                        charStack.push(w[i]);
                        ++i;
                    }
                    if (w[i] == 'A'){
                        charStack.pop();
                        ++i;
                    }
                }
            }
        }
        // Finally, the stack should be empty
        if (charStack.size() == 0){
            std::cout << 5 << '\n';
            std::cout <<"True, pattern entered matches the language " << '\n';
            return true;
        }
         if (charStack.size() != 0){
            std::cout << 6 << '\n';
            std::cout <<"False, pattern entered does not match the language " << '\n';
            return false;
        }
    }
}


int main() {
    char * input = new char[MAX_LENGTH];
    std::cout << "Input any string: " << std::endl;
    std::cin >> input;
    isInLanguage(input);
    return 0;
}
