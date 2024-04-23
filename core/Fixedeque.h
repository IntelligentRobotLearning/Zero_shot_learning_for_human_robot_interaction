#include <deque>
#include <iostream>

// deque with max length 
template <class T> class FixedQueue{
    std::deque<T> _deque;
    int MaxLen; 
    
public:
    FixedQueue(){}; 
    FixedQueue(int len){
        MaxLen = len; 
    }
    
    ~FixedQueue(){};
    
    void setMaxLen(int i){
        MaxLen = i;
    }
    void push_back(const T& value) {
        if (_deque.size() == MaxLen) {
           _deque.pop_front();
        }
        _deque.push_back(value);
    }
    T get(int i){
        return _deque[i];
    }
    int size(){
        return _deque.size();
    }

};
