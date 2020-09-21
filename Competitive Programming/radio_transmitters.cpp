#include <bits/stdc++.h>

using namespace std;

int hackerlandRadioTransmitters(vector<int> x, int k) {
    // sort(x.begin(), x.end());
    // if(x[x.size()-1] -x[0] <= k){
    //     return 1;
    // }
    // vector<double> x2(x.begin(), x.end());
    // auto it = unique(x2.begin(), x2.end()); 
    // x2.resize(distance(x2.begin(), it)); 
    // double res = 0;
    // for(int i = 0; i < x2.size(); i+=k+1){
    //     if(i < x2.size() - k){
    //         res += ceil((x2[i+1] - x2[i])/k);
    //     }
    //     else{
    //         if(k == 1){
    //             res += ceil((x2[x2.size()-1] - x2[i])/k);
    //         }
    //         else{
    //             res += ceil((x2[x2.size()-1] - x2[i])/k);
    //         }
    //     }
    // }
    // return res;
    priority_queue<int , vector<int> , greater<int>> pq;
    for(int i : x){
        pq.push(i);
    }
    int res = 1 , l = pq.top() , r = pq.top();
    pq.pop();
    while(!pq.empty()){
        while(pq.top() <= l+k && !pq.empty()){
            r = pq.top();
            pq.pop();
        }
        while(pq.top() <= r+k && !pq.empty())
            pq.pop();
        l = (!pq.empty())?pq.top():0;
        res = (!pq.empty())?res+1:res;
    }
    return res;
}
