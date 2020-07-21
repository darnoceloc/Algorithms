void heapify_down(vector<int>& arr, int size, int root) {
     for (int k = 1; k < size; ++k) {

        if (arr[k] < arr[(k - 1) / 2])  { 
            int j = k; 

            while (arr[j] < arr[(j - 1) / 2])  { 
                std::swap(arr[j], arr[(j - 1) / 2]); 
                j = (j - 1) / 2; 
            }
        }
    }
    for (int k = size - 1; k > 0; --k) {
        std::swap(arr[0], arr[k]);           

        int j = 0, index; 

        do { 
            index = (2 * j + 1);  

            if (arr[index] > arr[index + 1] && index < (k - 1)) 
                ++index; 

            if (arr[j] > arr[index] && index < k) 
                std::swap(arr[j], arr[index]); 

            j = index; 

        } while (index < k); 
    } 
    return;
}
