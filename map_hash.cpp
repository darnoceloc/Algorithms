#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <stack>


std::map<int, std::stack<std::string>> createMap(std::string arr[], int len) {
    std::stack<std::string> strstack;
         for(int i = 0; i < len; ++i){
           strstack.push(arr[i]);
        }
    std::map<int, std::stack<std::string>> mymap;
    for(int i = 0; i < len; ++i){
        mymap[arr[i].length()].push(strstack.top());
        strstack.pop();
    }
    return mymap;
}

bool isPresent(std::map<int,std::stack<std::string>> obj, std::string value) {
    std::map<int, std::stack<std::string>>::iterator it = obj.begin();
    for(it; it!= obj.end(); ++it) {
        if(it->second.top() == value){
            std::cout << std::boolalpha << true << '\n';
            return true;
        }
        else{
            it->second.pop();}
            ++it;
        }
    std::cout << "value not found" << '\n';
    std::cout << std::boolalpha << false << '\n';
    return false;
}


int main() {
    std::string val;
    int size;
    std::string values;
    std::string* str = new std::string[20];
    std::cout << "Enter the elements of map" << std::endl;
    std::getline(std::cin, values);
    std::istringstream iss(values); 
     while (iss >> val) {
        str[size] = val;
        ++size;
     }
    std::map<int, std::stack<std::string>> mainmap = createMap(str, size);
    std::string check;
    std::cout << "Enter string to search in map" << std::endl;
    std::cin >> check;
    isPresent(mainmap, check);
    return 0;
}