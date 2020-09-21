#include <bits/stdc++.h>

using namespace std;

// Complete the repeatedString function below.
long repeatedString(string s, long n) {
    if(s.length() == 1  && n == 1){
        return 1;
    }
    if(n < (long) s.length()){
        int count = 0;
        string sub_diff = s.substr(0, n);
        for(int j : sub_diff){
            if(j == 'a'){
                count += 1;
            }
        }
        return count;
    } 
    if(n >= (long) s.length()){
        long mult = floor(n/s.length());
        long rem = n - mult*s.length();
        long count = 0;
        for(int i : s){
            if(i == 'a'){
                count += 1;
            }
        }
        count *= mult;
        string sub_rem;
        if(rem > 0) {
            sub_rem = s.substr(0, rem);
        }
        for(int j : sub_rem){
            if(j == 'a'){
                count += 1;
            }
        }
        return count;
    }
    return 0;

}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    string s;
    getline(cin, s);

    long n;
    cin >> n;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    long result = repeatedString(s, n);

    fout << result << "\n";

    fout.close();

    return 0;
}
