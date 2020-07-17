#define MAXNUMVERTICES 100
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <climits>
using namespace std;


typedef map<int,int> Ints;
typedef map<int, Ints> nestInts;

struct edge{
    int src,des,weight = 0; 
    edge(){};
    edge(int s,int d,int w): src(s),des(d),weight(w){}
};

struct compare {
    bool operator()(const edge &a,const edge &b){ 
        if(a.weight == 0 && b.weight != 0){
            return a.weight < b.weight;
        }
        if(a.weight != 0 && b.weight == 0){
            return a.weight > b.weight;
        }
        if(a.weight == 0 && b.weight == 0){
            return a.weight == b.weight;
        }
        else{
            return a.weight < b.weight;
        } 
    }
}comp;

class Graph {
    private:
        int theGraph[MAXNUMVERTICES][MAXNUMVERTICES] = {{0}};
        set<int> vertices = {0};      

    public:
        void insertEdge(int from, int to, int weight);
        void insertVertex(int x);
        void primMST(nestInts mstGraph);
};

void Graph::insertEdge(int to, int from, int weight) {
     if (weight = 0){
         return;
     }
	 this->theGraph[to][from] = weight;
     this->theGraph[from][to] = weight;
     return;
}

void Graph::insertVertex(int x) {
    this->vertices.insert(x);
    return;
}

void Graph::primMST(nestInts mstGraph) {
    vector<edge> spanningTree;
    vector<edge> minEdges;
    for(auto it = mstGraph.cbegin(); it != mstGraph.cend(); it++){
        for(auto it2 = mstGraph[it->first].cbegin(); it2 != mstGraph[it->first].cend(); it2++){
            minEdges.push_back(edge(it->first, it2->first, it2->second));
        }
        auto it3 = min_element(minEdges.cbegin(), minEdges.cend(), comp);
        spanningTree.push_back(edge(it3->src, it3->des, it3->weight));
        minEdges.clear();
    }
    
    sort(spanningTree.begin(), spanningTree.end(), compare());
    vector<edge> finalTree;
    while(true){
        if(spanningTree.front().src < spanningTree.front().des){
            finalTree.push_back(edge(spanningTree.front().src, spanningTree.front().des, spanningTree.front().weight));
            spanningTree.erase(spanningTree.begin());
            break;
        }
        if(spanningTree.front().src > spanningTree.front().des){
            finalTree.push_back(edge(spanningTree.front().des, spanningTree.front().src, spanningTree.front().weight));
            spanningTree.erase(spanningTree.begin());
            break;
        }
    }
    while(spanningTree.size() > 0){
        auto it4 = find_if(finalTree.begin(), finalTree.end(), [&spanningTree](const edge& obj) {return obj.des == spanningTree.front().src;});
        auto it5 = find_if(finalTree.begin(), finalTree.end(), [&spanningTree](const edge& obj) {return obj.src == spanningTree.front().src;});
        auto it6 = find_if(finalTree.begin(), finalTree.end(), [&spanningTree](const edge& obj) {return obj.des == spanningTree.front().des;});
        auto it7 = find_if(finalTree.begin(), finalTree.end(), [&spanningTree](const edge& obj) {return obj.src == spanningTree.front().des;});

        if(spanningTree.front().src == it4->des && spanningTree.front().des == it7->src){
            spanningTree.erase(spanningTree.begin());  
        }
        if(spanningTree.front().src == it5->src || spanningTree.front().src == it4->des){
            finalTree.push_back(edge(spanningTree.front().src, spanningTree.front().des, spanningTree.front().weight));
            spanningTree.erase(spanningTree.begin());      
        }
        if(spanningTree.front().des == it7->src || spanningTree.front().des == it6->des){
            finalTree.push_back(edge(spanningTree.front().des, spanningTree.front().src, spanningTree.front().weight));
            spanningTree.erase(spanningTree.begin());             
        }
    }
    for(auto x : finalTree){
        cout << x.src << " " << x.des << endl;
    }
    return;
}

int main() {
    Graph *myGraph = new Graph();
    nestInts graph;
    int numEdges, inVert, outVert, wt;
    std::cin >> numEdges;
    edge minEdge;
    for (int i=0; i<numEdges; i++) {
        std::cin >> inVert;
        std::cin >> outVert;
        std::cin >> wt;
        if(wt == 0){
            continue;
        }
        myGraph->insertEdge(inVert, outVert, wt);
        myGraph->insertVertex(inVert);
        myGraph->insertVertex(outVert);
        graph[inVert][outVert] = wt;
        graph[outVert][inVert] = wt;
    }
    myGraph->primMST(graph);
    return 0;
}