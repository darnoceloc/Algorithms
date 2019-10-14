#include<iostream>
#include<string>
#include<vector>

using namespace std;

//Simple (brute force/naive) match algorithm
bool has_substring(const string& input_string, const string& pattern){
    int j=0;
    int i=0;
    int k = 0;
    while(i<input_string.length() && j<pattern.length()){
        if(input_string[i] == pattern[j]){
        i++;
        j++;}
        else{j=0; k++ ; i=k;}
    }
    if(j==pattern.length()){
        return true;
    }
    return false;
}

//Function to compute temporary integer array for prefix/suffix matches
vector<int> temp_array(const string& pattern){
    vector<int> tmp(pattern.length());
    int index = 0;
    for(int i=1; i<pattern.length();){
        if(pattern[i] == pattern[index]){
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
bool KMP_alg(const string& input_string, const string& pattern){
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
        //cout << "Match begins at char. " << i-pattern.length()+1 << " of input" << endl;
        //cout << "Match ends at char. " << i << " of input" << endl;
        return true;}
    else {
        //cout <<"No match found" << endl;
        return false;}
}

int main(){
    string input = "darnoc76elocbergdarnoc78darnoc77";
    string pattern = "darnoc77";
    int start = clock();
    bool result = KMP_alg(input, pattern);
    int end = clock();
    cout << "KMP algorithm took " << ((float)end - start)/CLOCKS_PER_SEC << " seconds." << endl;
    int start1 = clock();
    bool check = has_substring(input, pattern);
    int end1 = clock();
    cout << "Naive algorithm took "  << ((float)end1 - start1)/CLOCKS_PER_SEC << " seconds." << endl;
    cout << result << endl;
    cout << check << endl;
    return 0;
}   
