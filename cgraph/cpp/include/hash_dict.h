#ifndef HASHDICT_H
#define HASHDICT_H

#include <ostream>

template <typename K, typename V>
struct kvpair {
    K k;
    V v;
};

template <typename K, typename V>
struct dict {
    struct kvpair<K, V>* table;
    int size;
    int max_size;
    K kjunkval;
    V vjunkval;
    
    dict(K kjunkval, V vjunkval, int initial_size = 4): max_size(initial_size), size(0), kjunkval(kjunkval), vjunkval(vjunkval) {
        if (initial_size < 0){
            table = nullptr;
            return;
        }
        table = (kvpair<K, V>*)calloc(initial_size, sizeof(kvpair<K, V>));
        for (size_t i = 0; i < max_size; i++){
            table[i].k = kjunkval;
        }
        
    }

    void dict_free() {
        free(table);
    }

    V& operator[](K k);
    class iterator {
    private:
        struct dict<K, V>& d;
        int current;
    public:
        iterator(struct dict<K, V>& d, int start): d(d), current(start) {}
        iterator(const iterator& it): d(it.d), current(it.current) {}

        iterator& operator=(const iterator& it){
            d = it.d;
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
            while (current < d.max_size) {
                current++;
                if (d.table[current].k != d.kjunkval){
                    break;
                }
            }
            return *this;
        }

        kvpair<K, V>& operator*() const{
            // std::cout << "here*" << std::endl;
            return d.table[current];
        }
    };

    iterator begin() {
        // std::cout << "herebegin" << std::endl;
        int first{0};
        while (first < max_size){
            if (table[first].k != kjunkval){
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
};


template <typename K, typename V>
void rehash(struct dict<K, V>& d){
    kvpair<K, V>* old_table = d.table;
    d.table = (kvpair<K, V>*)calloc(d.max_size * 2, sizeof(kvpair<K, V>));
    int old_size = d.max_size;
    d.max_size *= 2;
    d.size = 0;
    for (size_t i = 0; i < d.max_size; i++){
        d.table[i].k = d.kjunkval;
    }
    for (int i = 0; i < old_size; i++){
        if (old_table[i].k != d.kjunkval){
            insert(d, old_table[i].k, old_table[i].v);
        }
    }
    delete old_table;
}

template <typename K, typename V>
void insert(struct dict<K, V>& d, K k, V& v){
    if (d.size >= d.max_size / 2){
        rehash(d);
    }
    int loc = k % d.max_size;
    while (1) {
        if (d.table[loc].k != d.kjunkval && d.table[loc].k != k){
            loc = (loc + 1) % d.max_size;
            continue;
        }
        if (d.table[loc].k != k){
            d.size++;
        }
        d.table[loc].k = k;
        d.table[loc].v = v;
        return;
    }
}


template <typename K, typename V>
void insert(struct dict<K, V>& d, K k, V&& v){
    if (d.size >= d.max_size / 2){
        rehash(d);
    }
    int loc = k % d.max_size;
    while (1) {
        if (d.table[loc].k != d.kjunkval && d.table[loc].k != k){
            loc = (loc + 1) % d.max_size;
            continue;
        }
        if (d.table[loc].k != k){
            d.size++;
        }
        d.table[loc].k = k;
        d.table[loc].v = v;
        return;
    }
}

template <typename K, typename V>
void remove(struct dict<K, V>& d, K k){
    return;
}

template <typename K, typename V>
bool mem(struct dict<K, V>& d, K k){
    int loc = k % d.max_size;
    while (1) {
        // our table in set is always at least half empty
        // so we surely gonna bump into junkval
        if (d.table[loc].k == d.kjunkval){
            return false;
        }
        if (d.table[loc].k != k){
            loc = (loc + 1) % d.max_size;
            continue;
        }
        return true;
    }
}

template <typename K, typename V>
V& dict<K,V>::operator[](K k){
    int loc = k % max_size;
    while(1){
        if (table[loc].k == k){
            return table[loc].v;
        }
        if (table[loc].k == kjunkval) {
            table[loc].v = vjunkval;
            return table[loc].v;
        }
        loc = (loc + 1) % max_size;
    }
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, struct dict<K, V>& d){
    os << "{";
    bool first = true;
    for (int i{0}; i < d.max_size; i++){
        if (d.table[i].k != d.kjunkval){
            if (first) {
                first = false;
            } else {
                os << ", ";
            }
            os << d.table[i].k << ": " << d.table[i].v;
        }
    }
    os << "}" << std::endl;
    return os;
}


#endif