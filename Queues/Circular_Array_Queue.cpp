#include <iostream>
#define SIZE 5 /* Size of Circular Queue */

class CircularArrayQueue {
   private:
  int items[SIZE], front, rear, count;

   public:
	CircularArrayQueue();
	bool isEmpty() const;
	void enqueue(int x);
	void dequeue();
	int peekFront() const;
};

CircularArrayQueue::CircularArrayQueue() {
    front = 0;
    rear = SIZE -1;
    count = 0;
  }

  bool CircularArrayQueue::isEmpty() const{
      return count == 0;
  }
  // Adding an element
  void CircularArrayQueue::enqueue(int element) {
      if (count < SIZE) {
        rear = (rear + 1) % SIZE;
        items[rear] = element;
        count++;
      }
    }

  // Removing an element
  void CircularArrayQueue::dequeue() {
    if (!isEmpty()) {
      front = (front + 1) % SIZE;
      count--;
      return;
    }
  }

  int CircularArrayQueue::peekFront() const {
    // Function to display status of Circular Queue
    if (!isEmpty()) {
      return items[front];
    }
  }
