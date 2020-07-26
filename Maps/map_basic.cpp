#include <iostream>
#include <string>
#include <map>
#include <sstream>

std::map<int, std::string> createMap(std::string arr[], const int& len) {
    std::map<int, std::string> mymap;
    for(int i = 0; i < len; ++i){
        mymap[arr[i].length()] = arr[i];
    }
    return mymap;
}

void printMap(std::map<int, std::string>& obj) {
    std::cout << "mymap contains:\n";
    std::map<int,std::string>::iterator it = obj.begin();
    for(it; it != obj.end() ; ++it){
        std::cout << it->second << " => " << it->second.size() << '\n';
    } 
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
    std::map<int, std::string> mainmap = createMap(str, size);
    printMap(mainmap);
    return 0;
}
