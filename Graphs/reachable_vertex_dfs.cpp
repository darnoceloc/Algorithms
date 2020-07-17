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


bool dfs_helper(const Graph& graph, int src, int dest, std::vector<bool> visited) {
	// your code
      if(src == dest) {
        return true;
    }
        // mark current node as discovered
	visited[src] = true;

	// do for every edge (src -> i)
	for (int i: graph.adjList[src])
	{
		// u is not discovered
		if (!visited[i])
		{	// return true if destination is found
			if (dfs_helper(graph, i, dest, visited))
				return true;
		}
	}
	// return false if destination vertex is not reachable from src
	return false;
}


bool dfs(const Graph& graph, int src, int dest) {
    std::vector<bool> visited(graph.adjList.size(),false);
    return dfs_helper(graph, src, dest, visited);
}


