#include<iostream>
#include<string>
#include<sstream>
#include<map>
#include<vector>
#include<algorithm>
using namespace std;


class Graph
{
    private:
    map<int, map<int, int>> graph;
    vector<int> numEdge;
    int vert = 0;
    public:
    void insertEdge(int from, int to, int weight){
        graph[from][to] = weight;
        numEdge.push_back(from);
        numEdge.push_back(to);
    }

    bool isEdge(int from, int to){
          return graph[from][to] != 0;
    }

    int getWeight(int from, int to){
          return graph[from][to];
    }

    vector<int> getAdjacent(int vertex){
        int i;
        vector<int> temp;
        for (i=0; i <= graph.size(); i++){
            if (graph[vertex][i] != 0) {
                temp.push_back(i);
            }
        }
        return temp;
    } 

    void printGraph(){
        sort(numEdge.begin(), numEdge.end());
        numEdge.erase(unique(numEdge.begin(),numEdge.end()),numEdge.end());
        for(auto j = numEdge.cbegin(); j != numEdge.cend(); ++j) {
            cout << *j;
            for(auto test = numEdge.cbegin(); test!= numEdge.cend(); ++test) {
                if(isEdge(*j, *test)){
                    cout << " " << *test;
                }
            }
            cout << endl;
        }
    }
};

int main()
{
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