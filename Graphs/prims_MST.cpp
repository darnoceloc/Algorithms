#define MAXNUMVERTICES 100
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <functional>
#include <climits>
using namespace std;


typedef map<int, map<int,int>> nestInts;

struct edge{
    int src,des,weight = 0; 
    edge(){};
    edge(int s,int d,int w): src(s),des(d),weight(w){}
};

struct comp_weight {
    bool operator()(const edge &a, const edge &b){ 
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
            return a.weight > b.weight;
        } 
    }
}comp_wt;

struct comp_vert {
    bool operator()(const edge &a, const edge &b){ 
        if(a.src == b.src && a.des != b.des){
            return true;
        }
        if(a.src == b.des && a.des != b.src){
            return true;
        }
        if(a.des == b.src && a.src != b.des){
            return true;
        }
        if(a.des == b.des && a.src != b.src){
            return true;
        }
        else{
            return false;
        }
    }
}comp_vt;

struct comp_match {
    bool operator()(const edge &a, const edge &b){ 
        if(a.des == b.src && a.src == b.des){
            return true;
        }
        if(a.des == b.des && a.src == b.src){
            return true;
        }
        else{
            return false;
        }
    }
}comp_mt;

class Graph { 
    private:
        nestInts graph;
        vector<edge> spanningTree;
        vector<edge> finalTree;
        vector<edge>::iterator iter;
        int edge_count = 0;    
    public:
        void insertEdge(int from, int to, int weight);
        void primMST();
        vector<edge>::iterator check_vert(vector<edge>& span, vector<edge>& final);
        vector<edge>::iterator check_match(vector<edge>& e1, vector<edge>& e2);
        vector<edge>::iterator find_vt(vector<edge>::iterator ita, vector<edge>& span, vector<edge>& final);
        int get_size();
        int get_density();
};

void Graph::insertEdge(int from, int to, int weight){
	 this->graph[to][from] = weight;
     this->graph[from][to] = weight;
     ++edge_count;
}

vector<edge>::iterator Graph::find_vt(vector<edge>::iterator ita, vector<edge>& span,  vector<edge>& final){
    this->iter = find_first_of(++++ita, span.end(), final.begin(), final.end(), comp_vt);
    cout << iter->src << " vt1 " << iter->des << endl;
    auto itb = find_if(final.begin(), final.end(), [this](edge& obj) {return obj.src == this->iter->src || obj.des == this->iter->src;});
    auto itc = find_if(final.begin(), final.end(), [this](edge& obj) {return obj.src == this->iter->des || obj.des == this->iter->des;});
    if (itb == final.end() || itc == final.end()){
        return this->iter;
    }
    else if(itb != final.end() && itc != final.end()){
        this->find_vt(this->iter, span, final);
    }
    else{
        return this->iter;
    }
}

vector<edge>::iterator Graph::check_vert(vector<edge>& span, vector<edge>& final){
    this->iter = find_first_of(span.begin(), span.end(), final.begin(), final.end(), comp_vt);
    cout << iter->src << " cvt1 " << iter->des << endl;
    auto it2 = find_if(final.begin(), final.end(), [this](edge& obj) {return obj.src == this->iter->src || obj.des == this->iter->src;});
    auto it3 = find_if(final.begin(), final.end(), [this](edge& obj) {return obj.src == this->iter->des || obj.des == this->iter->des;});
    if (it2 == final.end() || it3 == final.end()){
        cout << iter->src << iter->des << endl;
        return this->iter;
    }
    else if(it2 != final.end() && it3 != final.end()){
        this->find_vt(this->iter, span, final);
    }
    else{
        return this->iter;
    }
}

vector<edge>::iterator Graph::check_match(vector<edge>& e1, vector<edge>& e2){
    auto it_ma = find_first_of(e2.begin(), e2.end(), e1.begin(), e1.end(), comp_mt);
    return it_ma;
}

int Graph::get_size(){
    return this->graph.size();
}

int Graph::get_density(){
    return this->edge_count;
}

void Graph::primMST() {
    priority_queue<edge, vector<edge>, comp_weight> pq;
    for(auto it = this->graph.cbegin(); it != this->graph.cend(); it++){
        for(auto it2 = this->graph[it->first].cbegin(); it2 != this->graph[it->first].cend(); it2++){
            pq.push(edge(it->first, it2->first, it2->second));
        }
    }
    while (!pq.empty()) {
        spanningTree.push_back(pq.top());
        pq.pop();
    }
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
   
    while(this->finalTree.size() < this->get_size()-1){
        auto itr_mt = check_match(finalTree, spanningTree);
        spanningTree.erase(itr_mt);
        auto itr_vt = check_vert(spanningTree, finalTree);
        finalTree.push_back(edge(itr_vt->src, itr_vt->des, itr_vt->weight));
        spanningTree.erase(itr_vt);
    }
    for(auto x : finalTree){
        cout << x.src << " " << x.des << endl;
    }
    return;
}

int main() {
    Graph *myGraph = new Graph();
    int numEdges, inVert, outVert, wt;
    std::cin >> numEdges;
    edge minEdge;
    for (int i=0; i<numEdges; i++) {
        std::cin >> inVert;
        std::cin >> outVert;
        std::cin >> wt;
        myGraph->insertEdge(inVert, outVert, wt);
    }
    myGraph->primMST();
    return 0;
}