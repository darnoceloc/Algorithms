#include<iostream>
#include<string>
#include<vector>

using namespace std;

//Function to compute temporary integer array for prefix/suffix matches
vector<int> temp_array(string& pattern){
    vector<int> tmp(pattern.length());
    int index = 0;
    tmp[0] = 0;
    for(int i=1; i<pattern.length();){
        if(pattern[i]==pattern[index]){
            tmp[i]= tmp[index]+1;
            index++;
            i++;
        }
        else{
            if(index != 0){
            index = tmp[index-1];}
            else{tmp[i]=0;
            i++;}
        }
    }
    return tmp; 
}

//Function to implement KMP Algorithm for pattern match
bool KMP_alg(string& input_string, string& pattern){
    vector<int> tmp_kmp = temp_array(pattern);
    int i=0;
    int j=0;
    while(i < input_string.length() && j < pattern.length()){
        if(input_string[i] == pattern[j]){
            i++;
            j++;}
        else{
            if(j!=0){
            j = tmp_kmp[j-1];}
            else{
            i++;}
        }
    }
    if(j == pattern.length()){
        cout << "Match begins at char. " << i-pattern.length()+1 << " of input" << endl;
        cout << "Match ends at char. " << i << " of input" << endl;
        return true;}
    else if(i == input_string.length()) {
        cout <<"No match found" << endl;
        return false;}
    else {
        cout << i <<" "<< j << endl;
    }
}

int main(){
    string input = "darnoc77eloc";
    string pattern = "darnoc";
    bool result = KMP_alg(input, pattern);
    cout << result << endl;
    return 0;
}   
