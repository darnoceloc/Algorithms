#include<iostream>
#include<vector>
#include<queue>


struct Edge 
{
	int src, dest;
	Edge(int _src, int _dest) {src = _src; dest = _dest;}
};

class Graph 
{
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

void bfs(const Graph& graph, int src) {
  // your code
    std::vector<bool>visited(graph.adjList.size(),false);
	std::queue<int> q;

	// mark source vertex as discovered
	visited[src] = true;

	// push source vertex into the queue
	q.push(src);

	// run till queue is not empty
	while (!q.empty())
	{
		// pop front node from queue and print it
		src = q.front();
		q.pop();
		std::cout << src << " ";

		// do for every edge (v -> u)
		for (int u : graph.adjList[src]){
			if (!visited[u])
			{
				// mark it discovered and push it into queue
				visited[u] = true;
				q.push(u);
			}
	    }
    }
}
