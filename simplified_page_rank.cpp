#include <iostream>
#include<sstream>
#include <string>
#include<map>
#include<vector>
#include <iomanip>
using namespace std;

class AdjacencyList {

    private:
    map<string, map<string, float>> graph;
    map<string, float> rank;
    vector<string> adj_temp;
    map<string, map<string, float>> Mat;
    map<string, float> temp_rank;

    public:
    void insertEdge(string from, string to){
        graph[from][to] = 1;
        map<string, map<string, float>>::iterator it;
        it = graph.find(to);
        if(it == graph.end()){
            map<string, float> dest;
            graph[to] = dest; 
        }
    }

    vector<string> getAdjacent(string vertex){
        vector<string> temp;
        for (auto k = graph.cbegin(); k != graph.cend(); ++k){
            if (graph[k->first][vertex] == 1) {
                temp.push_back(k->first);
            }
        }
        return temp;
    }
     
    bool isEdge(string from, string to){
          return graph[from][to] == 1;
    }

    int outdegree (string vertex){
        int degree = 0;
        for (auto k = graph.cbegin(); k != graph.cend(); ++k){
            if (graph[vertex][k->first] == 1) {
                ++degree;
            }
        }
        return degree;
    }
    
    void PageRank(int& n){
        float gsize = graph.size();
        for(auto j = graph.cbegin(); j != graph.cend(); ++j) {
            rank[j->first] = 1/gsize;
            adj_temp = getAdjacent(j->first);
            for(auto i = adj_temp.cbegin(); i != adj_temp.cend(); ++i) { 
                if(isEdge(*i, j->first)){
                    float outd = outdegree(*i);
                    Mat[j->first][*i] = 1/outd;
                }
            }
        }
        for(int s = 0; s < n-1; s++){
            for(auto k = Mat.cbegin(); k != Mat.cend(); ++k) {
                for(auto l = Mat[k->first].cbegin(); l != Mat[k->first].cend(); ++l) {
                    temp_rank[k->first] += Mat[k->first][l->first]*rank[l->first];
                }
            }
            rank = temp_rank;
            temp_rank.clear();
        }
        for(auto m = graph.cbegin(); m != graph.cend(); ++m) {
            cout << m->first << " " << fixed << setprecision(2) << rank[m->first] << endl; 
        }
    }
};

int main() {
    int no_of_lines, power_iterations;
    AdjacencyList al;
    string from, to;
    cin >> no_of_lines;
    cin >> power_iterations;
    for(int i=0; i< no_of_lines; i++){
        cin>>from;
        cin>>to;
        al.insertEdge(from, to);
    }
    al.PageRank(power_iterations);
}