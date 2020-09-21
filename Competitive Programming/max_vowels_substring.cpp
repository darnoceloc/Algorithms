#include <iostream>
#include <vector>
#include <string> 
#include <algorithm> 

int maxVowels(std::string s, int k) {
    int num = 0;
    for(int x = 0; x < s.size(); x++){
        if(s[x] == 'a' || s[x] == 'e' || s[x] == 'i' || s[x] == 'o' || s[x] == 'u'){
            num++;
        }
    }
    if(num == 0){
        return 0;
    }
    std::vector<int> counts;
    for(int i = 0; i <= s.size()-k+1; i++){
        std::string subs = s.substr(i, k);
        int count = 0;
        for(int j = 0; j < k; j++){
            if(subs[j] == 'a' || subs[j] == 'e' || subs[j] == 'i' || subs[j] == 'o' || subs[j] == 'u'){
                count++;
            }
        }
        counts.push_back(count);
    }
    return *max_element(counts.begin(), counts.end());    
    return 0;
    
}

std::string maxVowel(std::string s, int k) {
    int num = 0;
    for(int x = 0; x < s.size(); x++){
        if(s[x] == 'a' || s[x] == 'e' || s[x] == 'i' || s[x] == 'o' || s[x] == 'u'){
            num++;
        }
    }
    if(num == 0){
        return "None found!";
    }
    std::vector<int> counts;
    std::vector<std::string> sublist;
    for(int i = 0; i <= s.size()-k+1; i++){
        std::string subs = s.substr(i, k);
        int count = 0;
        for(int j = 0; j < k; j++){
            if(subs[j] == 'a' || subs[j] == 'e' || subs[j] == 'i' || subs[j] == 'o' || subs[j] == 'u'){
                count++;
            }
        }
        counts.push_back(count);
        sublist.push_back(subs);
    }
    auto pos = distance(counts.begin(), max_element(counts.begin(), counts.end()));
    return sublist[pos];    
    return 0;
    
}

int main(){
    std::string s = "auociioeldefuoek";
    std::string w = "arhethmime";
    std::string p = "ceebbaceeffo";
    std::cout << maxVowel(s, 4) << std::endl;
    std::cout << maxVowel(w, 4) << std::endl;
    std::cout << maxVowel(p, 4) << std::endl;
    return 0;
}