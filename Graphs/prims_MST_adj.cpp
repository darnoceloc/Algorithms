#define MAXNUMVERTICES 10
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

class Graph {
    private:
        int theGraph[MAXNUMVERTICES][MAXNUMVERTICES] = {{INT_MAX}};
        int key[MAXNUMVERTICES] = {0};
        int parent[MAXNUMVERTICES] = {0};
        bool mstSet[MAXNUMVERTICES] = {0};
        set<int> numEdge;
    public:
        void insertEdge(int from, int to, int weight);
        void primMST();
        vector<int> getAdjacent(int vertex);
        int minKey();
        void printGraph();
        void printMST(); 

};
void Graph::insertEdge(int to, int from, int weight) {
    theGraph[to][from] = weight;
    theGraph[from][to] = weight;
    numEdge.insert(from);
    numEdge.insert(to);
}
int Graph::minKey() {  
    // Initialize min value  
    int min = INT_MAX;
    int min_index = 1;  
    for (auto v = numEdge.begin(); v != numEdge.end(); v++){  
        if (mstSet[*v] == false && key[*v] < min){
            min = key[*v];
            min_index = *v;
        }
    }  
    return min_index;  
}
void Graph::printGraph(){
    for(auto it = numEdge.begin(); it != numEdge.end(); ++it) {
        cout << *it;
        for(int j = 1; j <= numEdge.size(); j++) {
            cout << " " << theGraph[*it][j];
        }
        cout << endl;
    }
    return;
} 
  
void Graph::printMST()  {  
    cout << "Edge \tWeight\n";  
    for (auto itr = ++numEdge.begin(); itr != numEdge.end(); itr++){
        cout << parent[*(itr)]<<" - "<<*itr<<" \t"<< theGraph[*(itr)][parent[*(itr)]]<<" \n";
    }
    return;  
}   
 
void Graph::primMST() {   
    for (int i = 1; i <= numEdge.size(); i++){  
        this->key[i] = INT_MAX;
        this->mstSet[i] = false; 
    } 
    key[1] = 0;
    parent[1] = -1; 
    for (int count = 1; count <= this->numEdge.size(); count++) {   
        int u = this->minKey(); 
        this->mstSet[u] = true;   
        for (int v = 1; v <= this->numEdge.size(); v++){  
            if (theGraph[u][v] && mstSet[v] == false && theGraph[u][v] < key[v]) { 
                parent[v] = u;
                key[v] = theGraph[u][v];
            }  
        }
    }  
    printMST();
    return;  
}   
int main() {
    Graph* myGraph = new Graph();
    int numEdges, inVert, outVert, wt = 0;
    cin >> numEdges;
    for (int i = 0; i < numEdges; i++) {
        std::cin >> inVert;
        std::cin >> outVert;
        std::cin >> wt;
        myGraph->insertEdge(inVert, outVert, wt);
    }
    myGraph->primMST();
    return 0;
}
