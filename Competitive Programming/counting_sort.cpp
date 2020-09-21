
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

vector<int> countingSort(vector<int> arr, int n) {  
    // Map to store the frequency of the array elements 
    map<int, int> freqMap; 
    for (auto i = arr.begin(); i != arr.end(); i++) { 
        freqMap[*i]++; 
    } 
    int i = 0; 
    // For every element of the map 
    for (auto it : freqMap) { 
        // Value of the element 
        int val = it.first; 
        // Its frequency 
        int freq = it.second; 
        for (int j = 0; j < freq; j++) 
            arr[i++] = val; 
    } 
     // Print the sorted array 
    for (auto i = arr.begin(); i != arr.end(); i++) { 
        cout << *i << " "; 
    }
    return arr;   
} 

int activityNotifications(vector<int> expenditure, int d) {
    int notifications = 0;
    int N = expenditure.size();
    for(int i = 0; i < expenditure.size() - d; i++){
        vector<int> sub(expenditure.begin()+i, expenditure.begin()+i+d);
        countingSort(sub, sub.size());
        if(d%2 == 0){
            if(expenditure[i+d+1] >= sub[d/2]+sub[(d/2)-1]){
                notifications += 1;
            }
        }
        else if(d%2 != 0){
            if(expenditure[i+d+1] >= 2*sub[d/2]){
                notifications += 1;
            }
        }
    }
    return notifications;
}