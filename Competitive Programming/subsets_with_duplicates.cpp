#include <iostream>
#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> v;
    res.push_back(v);
    for(int i = 0; i < nums.size(); i++){
        v.push_back(nums[i]);
        vector<vector<int>> temp = res;
        for(int j = 0; j < temp.size(); j++){
            temp[j].push_back(v[0]);
            sort(temp[j].begin(), temp[j].end());
        }
        res.insert(res.end(),temp.begin(),temp.end());
        v.clear();
    }
    sort(res.begin(), res.end());
    res.erase(unique(res.begin(), res.end()), res.end());
    return res;
}
