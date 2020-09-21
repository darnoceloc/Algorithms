#include <bits/stdc++.h>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);


#include <string>
/*
 * Complete the 'simpleCipher' function below.
 *
 * The function is expected to return a STRING.
 * The function accepts following parameters:
 *  1. STRING encrypted
 *  2. INTEGER k
 */

string simpleCipher(string encrypted, int k) {
    string result = "";
    for(int i = 0; i < encrypted.length(); i++){
        if(k <= 26){
            if(encrypted[i] - k < 65){
                result += encrypted[i] - k + 26;
            }
            if(encrypted[i] -k >= 65 && encrypted[i] -k <= 90){
                result += encrypted[i] - k;
            }
        }
        if(k > 26){
            if(encrypted[i] - k%26 < 65){
                result += encrypted[i] - k%26 + 26;
            }
            if(encrypted[i] -k%26 >= 65 && encrypted[i] - k%26 <= 90){
                result += encrypted[i] - k%26;
            }
        }
    }
    return result;
}