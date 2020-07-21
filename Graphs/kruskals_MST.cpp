#define MAXNUMVERTICES 10
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;


struct edge{int src,des,wgt;};

class Graph {
    public:
        int sumOfMST(vector<edge> edges);
};

class UnionFind {
    int *parent, *ranks, _size;
public:
    UnionFind(){
    }
    UnionFind(int size){
        parent = new int[size]; ranks = new int[size];
        for(int element = 0 ; element < size ; element++){
            parent[element] = element , ranks[element] = 0 ;
        }
        _size = size;
    }
    void resize(int size){
        parent = new int[size]; ranks = new int[size];
        for(int element = 0 ; element < size ; element++){
            parent[element] = element , ranks[element] = 0 ;
        }
        _size = size;
    }
    int find(int element){
        if(parent[element] == element){
            return element;
        }
        else{
            return parent[element] = find(parent[element]);          // Path Compression algorithm
        }
    }
    bool connected(int x,int y){
        if(find(x) == find(y)){
            return true;
        }
        else{
            return false;
        }
    }
    void merge(int x,int y){
        x = find(x);
        y = find(y);
        if(x != y){                                                   // Union by Rank algorithm
            if(ranks[x] > ranks[y]){
                parent[y] = x;
            }
            else if(ranks[x] < ranks[y]){
                parent[x] = y;
            }
            else{
                parent[x] = y; ranks[y] ++ ;
            }
            _size--;
        }
    }
    void clear(){
        delete [] parent; delete [] ranks;
    }
    int size(){
        return _size;
    }
};

bool comparator(const edge &a,const edge &b){
    return a.wgt < b.wgt;
}

int Graph::sumOfMST(vector<edge> edges) {
    int sumMST = 0;
    UnionFind uf(this->vertices.size());
    vector<edge> spanningTree;
    sort(edges.begin(),edges.end(),comparator);
    spanningTree.push_back(edges[0]);
    uf.merge(edges[0].src, edges[0].des);
    for(int i=1; i<edges.size(); i++){
        if(!uf.connected(edges[i].src,edges[i].des)){
            uf.merge(edges[i].src,edges[i].des);
            spanningTree.push_back(edges[i]);
        }
    }
    for(int i = 0; i < spanningTree.size(); i++){
        sumMST += spanningTree[i].wgt;
    }
    return sumMST;
}

int main() {
    Graph *myGraph = new Graph();
    vector<edge> edges;
    int numEdges, inVert, outVert, wt;
    std::cin >> numEdges;
    for (int i=0; i<numEdges; i++) {
        std::cin >> inVert;
        std::cin >> outVert;
        std::cin >> wt;
        edges.resize(numEdges);
        edges[i].src = inVert;
        edges[i].des = outVert;
        edges[i].wgt = wt;
    }
    int res =  myGraph->sumOfMST(edges);
    cout << res;
}
