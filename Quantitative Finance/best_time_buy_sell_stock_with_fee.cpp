class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int buy = INT_MIN;
        int sell = 0;
        for(auto price: prices){
            buy = max(sell - price, buy);
            sell = max(price + buy - fee, sell);
        }
        return sell;
    }
};
