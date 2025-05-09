#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid input size.\n";
        return 1;
    }

    vector<int> arr(n);
    cout << "Enter " << n << " space-separated integers:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int minVal = numeric_limits<int>::max();
    int maxVal = numeric_limits<int>::min();
    long long sum = 0;
    double avg = 0.0;

    // ✅ Correct OpenMP reduction using std::min and std::max
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        minVal = min(minVal, arr[i]);  // Correct min
        maxVal = max(maxVal, arr[i]);  // Correct max
        sum += arr[i];                 // Correct sum
    }

    avg = static_cast<double>(sum) / n;

    cout << "\nResults (Using Parallel Reduction):\n";
    cout << "Minimum: " << minVal << endl;
    cout << "Maximum: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;

    return 0;
}