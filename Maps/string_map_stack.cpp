#include <iostream>
#include <stack>
#include <map>
#include <string>
using namespace std;


map<int,stack<string>> createMap(string arr[], int len){
    for (int i = 1 ;i < len; i++) { 
        string temp = arr[i]; 
        int j = i - 1; 
        while (j >= 0 && temp.length() < arr[j].length()){ 
            arr[j+1] = arr[j]; 
            j--; 
        } 
        arr[j+1] = temp; 
    }
    map<int, stack<string>> key_value;
    stack<string> real; 
    for (int i = 0; i < len; i++) {
        real.push(arr[i]);
        for (int k = i+1; k < len; k++){
            if(arr[i].length() == arr[k].length()){
                real.push(arr[k]);
            }
            else if(arr[i].length() != arr[k].length()){
                continue;
            }
        }
        key_value[arr[i].length()] = real;
        while (!real.empty()){
            real.pop();
        }
    }
    return key_value;
}

bool isPresent(map<int,stack<string>> obj, string value){
    for(auto itr = obj.begin(); itr != obj.end(); ++itr){
        while(!itr->second.empty()){
            if(itr->second.top() == value){
                return true;
            }
            itr->second.pop();
        }
    }
    return false;
}
