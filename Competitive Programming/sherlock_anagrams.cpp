#include <bits/stdc++.h>

using namespace std;

int sherlockAndAnagrams(string s) {
    int N = s.length(); 
  
    map<vector<int>, int> mp; 
  
    for (int i=0; i<N; i++) { 
        vector<int> freq(26, 0); 
  
        for (int j=i; j<N; j++) { 
            freq[s[j]-'a']++; 
            mp[freq]++; 
        } 
    } 
  
    int result = 0; 
    for (auto it=mp.begin(); it!=mp.end(); it++) { 
        int freq = it->second; 
        result += ((freq) * (freq-1))/2; 
    } 
    return result;  

}