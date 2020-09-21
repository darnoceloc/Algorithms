
#include <iostream>
#include <bits/stdc++.h>
using namespace std;  
  
vector<vector<int>> subsets_test(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> subs;
    for(int i = 0; i < nums.size(); i++){
        subs.push_back(nums[i]);
        res.push_back(subs);
        subs.clear(); 
        if(i <= nums.size()-1){
            int k = i+1;
            while(k < nums.size()){
                int x = 0;
                while(x < nums.size()-k){
                    subs.push_back(nums[i]);
                    for(int j = k; j < k+1+x; j++){
                        subs.push_back(nums[j]);
                    }
                    res.push_back(subs);
                    subs.clear();
                    x++;
                }
                k++;
            }
        }
    }
    res.push_back({});
    return res;
}

 vector<vector<int>> subsets_real2(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> v;
    res.push_back(v);
    for(int i = 0; i < nums.size(); i++){
        v.push_back(nums[i]);
        vector<vector<int>> temp = res;
        for(int j = 0; j < temp.size(); j++){
            temp[j].push_back(v[0]);
        }
        res.insert(res.end(),temp.begin(),temp.end());
        v.clear();
    }
    return res;
}



vector<vector<int>> subsets_real1(vector<int>& nums) {
    vector<vector<int>> res;
    res.push_back({});
    for(int i = 0; i < nums.size(); i++){
        vector<vector<int>> temp = res;
        for(int j = 0; j < temp.size(); j++){
            temp[j].push_back(nums[i]);
        }
        res.insert(res.end(),temp.begin(),temp.end());
    }
    return res;
}