
#include <bits/stdc++.h>
using namespace std;

bool comp(const pair<int,int>& a, const pair<int,int>& b){
        return a.first > b.first;
}

int lilysHomework(vector<int> arr) {
    vector<pair<int,int>> maps(arr.size());
    for(int i = 0; i < maps.size(); i++){
        maps[i].first = arr[i];
        maps[i].second = i;
    }
    sort(maps.begin(), maps.end());
    int res = 0;
    for(int j = 0; j < maps.size(); j++){
        if(maps[j].second == j){
            continue;
        }
        else{
            swap(maps[j].first,maps[maps[j].second].first);
            swap(maps[j].second,maps[maps[j].second].second);
        }
        if(maps[j].second != j){
            --j;
        }
        res++;
    }
    
    sort(maps.begin(), maps.end(), comp);
    int res2 = 0;
    for(int j = maps.size()-1; j >= 0; j--){
        if(maps[j].second == j){
            continue;
        }
        else{
            swap(maps[j].first,maps[maps[j].second].first);
            swap(maps[j].second,maps[maps[j].second].second);
        }
        if(maps[j].second != j){
            ++j;
        }
        res2++;
    }
    return (res < res2)?res:res2;
}