#include<iostream>
#include<vector>

struct Edge {
	int src, dest;
	Edge(int _src, int _dest) {src = _src; dest = _dest;}
};

class Graph {
  public:
	int numVertices;
	std::vector<std::vector<int>> adjList;  // adjacency list

	Graph(const std::vector<Edge>& edges, int vertices) {
		numVertices = vertices;

		adjList.resize(vertices);

		for (auto &edge : edges) {
			adjList[edge.src].push_back(edge.dest);
		}
	}
};

bool DFS(const Graph& graph, int v, std::vector<bool> &visited, int parent) {
	// mark current node as discovered
	visited[v] = true;
	// do for every edge (v -> w)
	for (int w : graph.adjList[v])
	{
		// w is not discovered
		if (!visited[w])
		{
			if (DFS(graph, w, visited, v))
				return true;
		}
		// w is discovered and w is not a parent
		else if (w != parent)
		{
			// we found a back-edge (cycle)
			return true;
		}
	}
	// No back-edges found in the graph
	return false;
}

bool anyCycle(const Graph& graph) {
    if(graph.numVertices < 3){
            return false;
    }
    std::vector<bool> visited(graph.adjList.size(),false);
    for (auto x : graph.adjList){
        for(auto y: x){
            visited[y] = false;
            }
        } 
        // Call the recursive helper function to detect cycle in
        // different DFS trees
        for (auto w : graph.adjList){
            for(auto z : w){
                if (!visited[z]){ // Don't recur for u if already visited
                    if (DFS(graph, graph.adjList[z][-1], visited, graph.adjList[z][0])){
                        return true;
                    }
                }
            }
        }
    return false;
}










