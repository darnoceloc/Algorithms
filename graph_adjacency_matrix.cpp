#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
using namespace std;


class Graph {
    private:
    int graph[10][10] = {{0}};
    vector<int> numEdge = {0};
    int numVert = 0;
    int red = 0;
    public:
    void insertEdge(int from, int to, int weight){
        if (weight == 0) {
            return; // Can't store weight of 0
        }
        if (from == to && !isVertex(from)) {
            graph[from][to] = weight;
            numEdge[numVert] = from;
            ++numVert;
            return;
        }
        if (from == to && isVertex(from)) {
            graph[from][to] = weight;
            return; 
        }
        if (from != to) {
            if(!isVertex(from) && !isVertex(to)){
                graph[from][to] = weight;
                numEdge[numVert] = from;
                numVert++;
                numEdge[numVert] = to;
                return;
            }
            if(isVertex(from) && !isVertex(to)){
                graph[from][to] = weight;
                ++numVert;
                numEdge[numVert] = to;
                return;
            }
            if(!isVertex(from) && isVertex(to)){
                graph[from][to] = weight;
                ++numVert;
                numEdge[numVert] = from;
                return;
            }
            if(isVertex(from) && isVertex(to)){
                graph[from][to] = weight;
                return;
            }
        }
    }

    bool isVertex(int vertex){
        for (int i = 1; i <= numVert; i++){
            if (graph[vertex][vertex+i] != 0 || graph[vertex-i][vertex] != 0){
                return true;
            }
        }
        return false;
    }
    
    bool isVertex2(int vertex){
        vector<int> adj = getAdjacent(vertex);
        return !adj.empty();
    }

    bool isEdge(int from, int to){
        return graph[from][to] != 0;
    }

    int getWeight(int from, int to){
        return graph[from][to];
    }

    vector<int> getAdjacent(int vertex){
        int i;
        int count = 0;
        vector<int> temp;
        for (i=0; i <= numVert+1; i++){
            if (isEdge(vertex, i)) {
                temp.push_back(i);
            }
        }
        return temp;
    } 

    void printGraph(){
        int i, j;
        for(i = 0; i <= numVert; i++) {
            cout << numEdge[i];
            for(j = 0; j <= numVert; j++) {
                cout << " " << graph[numEdge[i]][numEdge[j]];
            }
            cout << endl;
        }
    }
};

int main() {
    //DO NOT CHANGE THIS FUNCTION. CHANGE YOUR IMPLEMENTATION CODE TO MAKE IT WORK

    int noOfLines, operation, vertex, to, fro, weight, source, j;
    vector<int> arr;
    int arrSize;
    Graph g;
    cin>>noOfLines;
    for(int i=0;i<noOfLines;i++)
    {
        cin>>operation;
        switch(operation)
        {
            case 1: 
                cin>>fro;
                cin>>to;
                cin>>weight;
                g.insertEdge(fro,to,weight);
                break;
            case 2: 
                cin>>fro;
                cin>>to;
                cout<<g.isEdge(fro,to)<<"\n";
                break;
            case 3: 
                cin>>fro;
                cin>>to;
                cout<<g.getWeight(fro,to)<<"\n";
                break;
            case 4: 
                cin>>vertex;
                arr=g.getAdjacent(vertex);
                arrSize = arr.size();
                j=0;
                while(j<arrSize)
                {
                    cout<<arr[j]<<" ";
                    j++;
                }
                cout<<"\n";
                break;
            case 5: 
                g.printGraph();
                cout<<"\n";
                break;
        }
    }
    return 0;
}