class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() <= 1){
            return 0;
        }
        if(prices.size() == 2){
            if(prices[0] < prices[1]){
                return prices[1] - prices[0];
            }
            else{
                return 0;
            }
        }
        int buy1 = numeric_limits<int>::max(); int buy2 = numeric_limits<int>::max();
        int sell1 = 0, sell2 = 0;
        for (int i = 0; i < prices.size(); i++) {
            buy1 = min(buy1, prices[i]);
            cout << buy1 << endl;
            sell1 = max(sell1, prices[i] - buy1);
            cout << sell1 << endl;
            buy2 = min(buy2, prices[i] - sell1);
            cout << buy2 << endl;
            sell2 = max(sell2, prices[i] - buy2);
            cout << sell2 << endl;
        }
        return sell2;
    }
};
