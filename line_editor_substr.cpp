#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <math.h>
using namespace std;

// Linked List Class 
class LinkedList 
{
	int size;
	

public:
	LinkedList() 
	{
		size = 0;
		head = nullptr;
		tail = nullptr;
	}

	// Nested Node Class
	struct Node 
	{
		string value;
		int lineNum;
		Node * next = nullptr;
	};
 

	Node * head;
	Node * tail;

	// Search
	void search(string& phrase) 
	{
		string sub;
		if (phrase.length() > 80) {
             for (size_t i = 0; i < phrase.size(); i += 80){
                sub = phrase.substr(i, 80);
				Node * iter = head;
				vector<Node *> result;
				while (iter != nullptr) {
			// Search for phrase throughout the node's data
					if (iter->value.find(phrase) != string::npos)
					{
						result.push_back(iter);
					}
					iter = iter->next;
				}
				if (result.empty()) cout << "not found" << endl;
				else
				{
					for (Node * node : result)
					{
						cout << node->lineNum << " " << node->value << endl;
					}
				}
			 }
		}
		else {
			Node * iter = head;
			vector<Node *> result;
			while (iter != nullptr) {
		// Search for phrase throughout the node's data
				if (iter->value.find(phrase) != string::npos)
				{
					result.push_back(iter);
				}
				iter = iter->next;
			}
			if (result.empty()) cout << "not found" << endl;
			else
			{
				for (Node * node : result)
				{
					cout << node->lineNum << " " << node->value << endl;
				}
			}
		}
	}

	// Print function
	void print() 
	{
		Node * iter = head;
		while (iter != nullptr)
		{
			cout << iter->lineNum << " " << iter->value << endl;
			iter = iter->next;
		}
	}


	// Insert end
	void insertEnd(string& data) 
	{
		string sub2;
		Node* temp = new Node();
		if (data.length() > 80) {
             for (size_t i = 0; i < data.size(); i += 80){
                sub2 = data.substr(i, 80);
                temp->value = sub2;
                temp->lineNum = size + 1;
                temp->next = nullptr;
                // Empty List Case
                if (size == 0) 
                {
                    head = temp;
                    tail = temp;
                } 
                else 
                {
                    tail->next = temp;
                    if (size == 1) head->next = temp;
                    tail = temp;
                }
                size++;
             }
            return;
          }
		else {
			temp->value = data;
			temp->lineNum = size + 1;
			temp->next = nullptr;

			// Empty List Case
			if (size == 0) 
			{
				head = temp;
				tail = temp;
			} 
			else 
			{
				tail->next = temp;
				if (size == 1) head->next = temp;
				tail = temp;
			}
			size++;
		}
	}
	// Update Line Numbers after Addition/Deletion
	void updateLineNumbers(Node * ptr, bool decrement = true)
	{
		while (ptr != nullptr)
		{
			if (decrement) ptr->lineNum -= 1;
			else ptr->lineNum += 1;
			ptr = ptr->next;
		}
	}

	// Insert in the middle
	void insert(int& num, string& str)
	{
		string sub1;
		if (str.length() > 80) {
            for (size_t i = 0; i < str.size(); i += 80){
                sub1 = str.substr(i, 80);
                if (num <= 0 || num > size + 1) return;
                if (num == size + 1)
                {
                    insertEnd(sub1);
					num++;
					continue;
                }
				Node* temp = new Node();
                temp->lineNum = num;
                temp->value = sub1;

                Node * iter = head;
                Node * prev = iter;
                while (iter != nullptr && iter->lineNum != num)
                {
                    prev = iter;
                    iter = iter->next;
                }
                size++;
                prev->next = temp;
                temp->next = iter;
                updateLineNumbers(iter, false);
            }
		}
	 else {
		if (num <= 0 || num > size + 1) return;
			if (num == size + 1)
			{
				insertEnd(str);
				return;
			}
			Node* temp = new Node();
			temp->lineNum = num;
			temp->value = str;

			Node * iter = head;
			Node * prev = iter;
			while (iter != nullptr && iter->lineNum != num)
			{
				prev = iter;
				iter = iter->next;
			}
			size++;
			prev->next = temp;
			temp->next = iter;
			updateLineNumbers(iter, false);
		}
	}
	// Edit function  
	void editNode(int& num, string& editStr) 
	{
		int initial_pos = num;
		string subedit;
		if (editStr.length() > 80) {
			for (size_t i = 0; i < editStr.size(); i += 80){
				subedit = editStr.substr(i, 80);
				num++;
				insert(num, subedit);
			}
			deleteNode(initial_pos);
			return;
		}
		else {
			if (num <= 0 || num > size) return;
			Node * iter = head;
			while (iter != nullptr && iter->lineNum != num) iter = iter->next;
			iter->value = editStr;
			return;
		}
	}

	// Delete function
	void deleteNode(int& num)
	{
		if (num <= 0 || num > size) return;
		Node * iter = head;
		Node * prev = nullptr;
		while (iter != nullptr && iter->lineNum != num)
		{
			prev = iter;
			iter = iter->next;
		}
		Node* temp;
		// Case of Head
		if (num == 1)
		{
			temp = head;
			head = head->next;
			updateLineNumbers(head);
			delete temp;
		}
		// Case of Tail
		else if (num == size) 
		{
			temp = iter;
			tail = prev;
			tail->next = nullptr;
			delete temp;
        }
	
		else
		{
			// Case of Middle
			prev->next = iter->next;
			updateLineNumbers(iter->next);
			delete iter;
		}
		size--;		
	}		
};


int main()
{
    //your code to invoke line editor here
    LinkedList lineEditor;
	string operation;

	// Get input line by line and break each by spaces
	getline(cin, operation, '\n');
	while (operation != "quit")
	{
		istringstream component(operation);
		string str;
		int num;
		getline(component, str, ' ');

		if (str == "insertEnd")
		{
			getline(component, str, '\"');
			getline(component, str, '\"');
			lineEditor.insertEnd(str);
		} 
		else if (str == "print")
		{
			lineEditor.print();
		} 
		else if (str == "search")
		{
			getline(component, str, '\"');
			getline(component, str, '\"');
			lineEditor.search(str);
		} 
		else if (str == "edit")
		{
			getline(component, str, ' ');
			num = std::stoi(str);
			getline(component, str, '\"');
			getline(component, str, '\"');
			lineEditor.editNode(num, str);
		} 
		else if (str == "delete")
		{
			getline(component, str, ' ');
			num = std::stoi(str);
			lineEditor.deleteNode(num);
		}
		else if (str == "insert")
		{
			getline(component, str, ' ');
			num = std::stoi(str);
			getline(component, str, '\"');
			getline(component, str, '\"');
			lineEditor.insert(num, str);
		}


		// Get next line
		getline(cin, operation, '\n');

	}

	return 0;
}