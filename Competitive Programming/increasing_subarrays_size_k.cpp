int fn(vector<int>arr,int k){
    int n=arr.size();
    if(k == 0 || k>n) return 0;
    int ans=0;
    vector<int> preCompute(n,-1);
    int i = 0;
    int curr = 0;
    while(i < n-1){
        if(arr[i+1] > arr[i]) curr++;
        else{
            preCompute[i-curr] = curr + 1;
            curr=0;
        }
        i++;
    }
    preCompute[i-curr] = curr+1;
    for(int i=0;i<n;i++){
        if(preCompute[i] !=- 1 && preCompute[i] >= k) ans+= preCompute[i]-k+1;
    }
	return ans;
}


int cntSubArrays(vector<int> arr, int k) {
    int n = arr.size();
    int res = 0;
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n and arr[j] > arr[j - 1])
            j++;
        res += max(j - i - k + 1, 0);
        i = j;
    }
    return res;
}
