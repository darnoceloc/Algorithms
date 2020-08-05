#include<iostream>
#include<vector>


void dfs(std::vector<std::vector<int>>& P, int v, bool visited[]) {
    // Mark the current node as visited and print it 
    visited[v] = true; 
    std::cout << v << " "; 
  
    // Recur for all the vertices adjacent to this vertex 
    int j = 0;
    for(auto i = P[v].begin(); i != P[v].end(); ++i){
        if(*i == 0) {
            continue;
        }
        if(!visited[j]){
            dfs(P, j, visited);
            std::cout << std::endl;
            std::cout << j << " dfs ";
        }
        ++j;
    } 
}

int NFR(std::vector<std::vector<int>>& P) {
    // Mark all the vertices as not visited 
    bool *visited = new bool[P.size()]; 
    for(int v = 0; v < P.size(); v++){
        visited[v] = false; 
    }
    int count = 0;
    for (int v = 0; v < P.size(); v++) { 
        if (visited[v] == false) { 
            // print all reachable vertices from v 
            dfs(P, v, visited); 
            std::cout << "\n";
            ++count; 
        } 
    } 
    delete[] visited;
    return count; 
}

int main(){
    std::vector<std::vector<int>> P = {{1,1,0},{1,1,1},{0,1,1}};
    int result = NFR(P);
    std::cout << result << std::endl;
    return 0;
}