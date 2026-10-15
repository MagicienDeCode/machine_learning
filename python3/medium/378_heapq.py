import heapq
"""
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        pq = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                heapq.heappush(pq,-matrix[i][j])
                if len(pq) > k: heapq.heappop(pq)
        return -heapq.heappop(pq)
"""

pq = []
heapq.heappush(pq, 3)
heapq.heappush(pq, 10)
heapq.heappush(pq, 5)

while len(pq):
    print(heapq.heappop(pq))

pq2 = [1,9,9,5]
heapq.heapify(pq2)
while len(pq2):
    print(heapq.heappop(pq2))