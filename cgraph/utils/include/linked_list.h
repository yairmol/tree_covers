#ifndef LEARNCPP_LINKED_LIST_H
#define LEARNCPP_LINKED_LIST_H

struct Link {
    int value;
    struct Link *next;
};

/**
 * a linked list struct. when creating it, call the init_linked_list method
 * otherwise the behavior is undefined
 */
struct LinkedList{
    struct Link* head;
    struct Link* tail;
};

// Constructors and Destructorss

/** 
 * initialize the given linked list.
 * must be called on a linked list when it is created
 */
void init_linked_list(struct LinkedList* ll);

/**
 * allocate a linked list on the heap and initialize it
 */
struct LinkedList* LinkedList();

/**
 * free the memory of link including succesor links
 */
void free_link(struct Link* l);

/**
 * free the linked list memory
 */
void free_linked_list(struct LinkedList* ll);

// Insertion and Deletion

// insert an item the front of the linked list
void insert(struct LinkedList* ll, int value);

// insert an item to the back of the list
void push_back(struct LinkedList* ll, int value);

// removes a value if it exists and return 0, return -1 otherwise
int linked_list_remove(struct LinkedList* ll, int value);

// Copy

// creates a copy of the linked list
struct LinkedList copy_linked_list(struct LinkedList* ll);

// returns a linked list which is the reverse of the given linked list
struct LinkedList reverse_linked_list(struct LinkedList* ll);

// returns a new linked list which is the catenation of two linked lists
struct LinkedList caten_linked_list(struct LinkedList* ll1, struct LinkedList* ll2);

// prints a linked list contents nicely
void print_linked_list(struct LinkedList* ll);

#endif //LEARNCPP_LINKED_LIST_H
