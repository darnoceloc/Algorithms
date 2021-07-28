#include <iostream>
#include <vector> 
//#include <pair>

using namespace std;

pair<vector<pair<int, int>>, int> maxProfit(vector <int> prices) {
        int i = 0, buy, sell, ibuy, isell, profit = 0, N = prices.size() - 1;
        vector<pair<int, int>> store;
        pair<vector<pair<int, int>>, int> res;
        while (i < N) {
            while (i < N && prices[i + 1] <= prices[i]){ i++;}
            buy = prices[i];
	    ibuy = i;

            while (i < N && prices[i + 1] > prices[i]) {i++;}
            sell = prices[i];
            isell = i;
            cout << "sub profit = " << sell - buy << endl;
            profit += sell - buy;
            cout << "total so far = " << profit << endl;
            pair<int, int> temp = {ibuy, isell};
	    store.push_back(temp);
        }
        res = {store, profit};
        return res;
}

int main(){
	vector<int> stocks = {2, -3, 7, 0, -5, 10, 15};
        auto profits = maxProfit(stocks);
        for(int i = 0 ; i < profits.first.size(); i++){
        	cout << "best days " << profits.first[i].first << " " << profits.first[i].second << endl;
        }
        cout << "total " << profits.second << endl;
        return 0;
}
