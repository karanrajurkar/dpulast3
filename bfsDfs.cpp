#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;
const int MAXN = 100005;
vector<int> adj[MAXN];
bool visited[MAXN];
void parallel_bfs(int start) {
 queue<int> q;
 q.push(start);
 visited[start] = true;
 while (!q.empty()) {
 int level_size = q.size();
 vector<int> next_level;
 #pragma omp parallel for
 for (int i = 0; i < level_size; i++) {
 int v;
 #pragma omp critical
 {
 if (!q.empty()) {
 v = q.front();
 q.pop();
 } else v = -1;
 }
 if (v == -1) continue;
 #pragma omp critical
 cout << v << " ";
 for (int u : adj[v]) {
 bool seen;
 #pragma omp atomic read
 seen = visited[u];
 if (!seen) {
 #pragma omp critical
 {
 if (!visited[u]) {
 visited[u] = true;
 next_level.push_back(u);
 }
 }
 }
 }
 }
 for (int u : next_level)
 q.push(u);
 }
}
void dfs_task(int v) {
 #pragma omp critical
 {
 if (visited[v]) return;
 visited[v] = true;
 cout << v << " ";
 }
 #pragma omp parallel for
 for (int i = 0; i < adj[v].size(); i++) {
 int u = adj[v][i];
 #pragma omp task firstprivate(u)
 dfs_task(u);
 }
}
void parallel_dfs(int start) {
 #pragma omp parallel
 {
 #pragma omp single
 dfs_task(start);
 }
}
void reset_visited(int n) {
 #pragma omp parallel for
 for (int i = 1; i <= n; i++)
 visited[i] = false;
}
int main() {
 int n, m;
 cout << "Enter number of nodes and edges: ";
 cin >> n >> m;
 if (m > n * (n - 1) / 2) {
 cout << "Error: Too many edges.\n";
 return 1;
 }
 cout << "Enter edges:\n";
 for (int i = 0; i < m; i++) {
 int u, v;
 cin >> u >> v;
 adj[u].push_back(v);
 adj[v].push_back(u);
 }
 while (true) {
 cout << "\nChoose an option:\n1. Parallel BFS\n2. Parallel DFS\n3. Exit\nEnter your choice: ";
 int choice;
 cin >> choice;
 if (choice == 3) break;
 cout << "Enter starting node: ";
 int start;
 cin >> start;
 reset_visited(n);
 if (choice == 1) {
 cout << "Running Parallel BFS...\nVisited nodes: ";
 parallel_bfs(start);
 } else if (choice == 2) {
 cout << "Running Parallel DFS...\nVisited nodes: ";
 parallel_dfs(start);
 }
 for (int i = 1; i <= n; i++) {
 if (!visited[i] && !adj[i].empty()) {
 cout << "\nGraph has disconnected components. Resuming from node: " << i << endl;
 if (choice == 1) parallel_bfs(i);
 else parallel_dfs(i);
 }
 }
 cout << endl;
 }
 return 0;
}



//Enter number of node and edgse: 6 7
//Enter edges:
// 1 2
// 1 3
// 2 4
// 2 5
// 3 6
// 4 6
// 5 6
//select 1
//starting node 1
//select 2
//staring node 1