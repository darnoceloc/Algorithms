class Solution {
public:
    string licenseKeyFormatting(string s, int k) {
        stack<char> temp;
        string result;
        for(int i = s.length() - 1; i >= 0; i--){
            if(s[i] != '-'){
                temp.push(toupper(s[i]));
            }
        }
        int N = temp.size();
        if(fmod(N, k)){
            for(int j = 0; j < fmod(N, k); j++){
                result += temp.top();
                temp.pop();
            }
        }
        while(temp.size()){
            if(result.size()){
                result += '-';
            }
            int ctr = 0;
            while(ctr < k){
                result += temp.top();
                temp.pop();
                ctr++;
            }
        }
        return result;
    }
};

class Solution {
public:
    string licenseKeyFormatting(string s, int k) {
        string result;
        int ctr = 0;
        for(int i = s.length() - 1; i >= 0; i--){
            if(s[i] != '-'){
                if(result.size() > 0 && fmod(result.size()-ctr, k) == 0){
                    result += '-';
                    ctr++;
                }
                result += toupper(s[i]);
            }
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
