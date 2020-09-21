#include <bits/stdc++.h>
using namespace std;


vector<int> freqQuery(vector<vector<int>> queries) {
    unordered_map<int, long long int> elems;
    unordered_map<long long int, int> freqs;
    vector<int> res;
    for(int i = 0; i < queries.size(); i++){
        if(queries[i][0] == 1){
            elems[queries[i][1]]++;
            freqs[elems[queries[i][1]]] = queries[i][1];
            continue;
        }
        if(queries[i][0] == 2){
            if(elems.find(queries[i][1]) != elems.end()){
                elems[queries[i][1]]--;
                freqs[elems[queries[i][1]]] = queries[i][1];
                continue;
            }
            else{
                continue;
            }
        }
        if(queries[i][0] == 3){
            if(elems.empty() || freqs.empty()){
                res.push_back(0);
                continue;
            }
            if(freqs.find(queries[i][1]) != freqs.end()){
                res.push_back(1);
                continue;
            }
            else{
                res.push_back(0);
                continue;
            }
        }
    }
    return res;
}

int main(){
    vector<vector<int>> queries;
    int num_queries;
    cin >> num_queries;
    for(int i = 0; i < num_queries; i++){
        cin >> queries[i][0];
        cin >> queries[i][1];
    }

    vector<int> ans = freqQuery(queries);

    for (int i = 0; i < ans.size(); i++) {
        cout << ans[i];

        if (i != ans.size() - 1) {
            cout << "\n";
        }
    }

    cout << "\n";

    return 0;
}