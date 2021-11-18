#ifndef SET_H
#define SET_H

#include <iostream>

template <typename T>
struct set {
    T* table;
    int size;
    int max_size;
    T junkval;

    set(T junkval, int initial_size = 4): size(0), max_size(initial_size), junkval(junkval){
        if (initial_size <= 0){
            table = nullptr;
            return;
        }
        table = (T*)calloc(initial_size, sizeof(T));
        for (size_t i = 0; i < max_size; i++){
            table[i] = junkval;
        }
    }

    void set_free(){
        free(table);
    }

    class iterator {
    private:
        struct set<T>& s;
        int current;
    public:
        iterator(struct set<T>& s, int start): s(s), current(start) {}
        iterator(const iterator& it): s(it.s), current(it.current) {}
        ~iterator(){}

        iterator& operator=(const iterator& it){
            s = it.s;
            current = it.current;
            return *this;
        }
        bool operator==(const iterator& it) const{
            // std::cout << "here==" << std::endl;
            return it.current == current;
        }
        bool operator!=(const iterator& it) const{
            // std::cout << "here!=" << std::endl;
            return it.current != current;
        }

        iterator& operator++(){
            // std::cout << "here++" << std::endl;
            while (current < s.max_size) {
                current++;
                if (s.table[current] != junkval){
                    break;
                }
            }
            return *this;
        }

        T& operator*() const{
            // std::cout << "here*" << std::endl;
            return s.table[current];
        }
    };

    iterator begin() {
        // std::cout << "herebegin" << std::endl;
        int first{0};
        while (first < max_size){
            if (table[first] != junkval){
                break;
            }
            first++;
        }
        return iterator{*this, first};
    }
    iterator end(){
        // std::cout << "hereend" << std::endl;
        return iterator{*this, max_size};
    }

    set<T>& operator=(struct set<T>&& other){
        table = other.table;
        size = other.size;
        max_size = other.max_size;
        other.table = nullptr;
        return *this;
    }

    set<T>& operator=(struct set<T>& other){
        size = other.size;
        max_size = other.max_size;
        table = other.table;
        return *this;
    }
};

template <typename T>
void rehash(struct set<T>& s){
    T* old_table = s.table;
    int old_size = s.max_size;
    s.max_size = s.max_size * 2;
    s.size = 0;
    s.table = (T*)calloc(s.max_size, sizeof(T));
    for (int i = 0; i < s.max_size; i++){
        s.table[i] = s.junkval;
    }
    // init(snew, s.max_size * 2);
    for (int i = 0; i < old_size; i++){
        if (old_table[i] != s.junkval){
            insert(s, old_table[i]);
        }
    }
    delete old_table;
}

template <typename T>
/**
 * @brief return 0 on success, 1 if item already exists and -1 if fails
 * 
 * @param s 
 * @param e 
 */
int insert(struct set<T>& s, T& e){
    if (s.size >= s.max_size / 2){
        rehash(s);
    }
    int loc = e % s.max_size;
    while (1) {
        if (s.table[loc] != s.junkval && s.table[loc] != e){
            loc = (loc + 1) % s.max_size;
            continue;
        }
        if (s.table[loc] != e){
            s.table[loc] = e;
            s.size++;
            return 0;
        }
        return 1;
    }
}

template <typename T>
void insert(struct set<T>& s, T&& e){
    if (s.size >= s.max_size / 2){
        rehash(s);
    }
    int loc = e % s.max_size;
    while (1) {
        if (s.table[loc] != s.junkval){
            loc = (loc + 1) % s.max_size;
            continue;
        }
        if (s.table[loc] != e){
            s.table[loc] = e;
            s.size++;
            return;
        }
    }
}

template <typename T>
bool mem(struct set<T>& s, T e){
    int loc = e % s.max_size;
    while (1) {
        // our table in set is always at least half empty
        // so we surely gonna bump into junkval
        if (s.table[loc] == s.junkval){
            return false;
        }
        if (s.table[loc] != e){
            loc = (loc + 1) % s.max_size;
            continue;
        }
        return true;
    }
}

template <typename T>
void remove(struct set<T>& s, T e){
    return;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, struct set<T>& s){
    os << "{";
    bool first = true;
    for (int i{0}; i < s.max_size; i++){
        if (s.table[i] != s.junkval){
            if (first) {
                os << s.table[i];
                first = false;
            } else {
                os << ", " << s.table[i];
            }
        }
    }
    os << "}" << std::endl;
    return os;
}

#endif