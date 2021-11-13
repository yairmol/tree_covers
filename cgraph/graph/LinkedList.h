//
// Created by Yair Molakandov on 17/07/2021.
//

#ifndef LEARNCPP_LINKEDLIST_H
#define LEARNCPP_LINKEDLIST_H


template<typename T>
struct LL {
    T value;
    struct LL<T> *next;
};


template<typename T>
LL<T> *insert(LL<T> *current, T value) {
  return new LL<T>{value, current};
}

template<typename T>
int length(LL<T>* l) {
  int len {0};
  while (l != nullptr){
    l = l->next;
    len++;
  }
  return len;
}

template<typename T>
struct queue {
    LL<int>* head;
    LL<int>* tail;
};

template<typename T>
queue<T>* init_queue(T e){
  queue<T>* q {new queue<T>{new LL<T>{e, nullptr}, nullptr}};
  q->tail = q->head;
  return q;
}

template<typename T>
bool is_empty(queue<T>* q){
  return q->head == nullptr;
}

template<typename T>
void enqueue(queue<T>* q, T e) {
  q->tail->next = new LL<T>{e, nullptr};
  q->tail = q->tail->next;
  if (is_empty(q)){
    q->head = q->tail;
  }
}

template<typename T>
T dequeue(queue<T>* q) {
  T ret {q->head->value};
  q->head = q->head->next;
  return ret;
}

#endif //LEARNCPP_LINKEDLIST_H
