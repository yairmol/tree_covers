#include "hash_dict.h"


template <typename K, typename V>
void rehash(struct dict<K, V>& d){
    struct dict<K, V> dnew;
    init(dnew, d.max_size * 2);
    for (int i = 0; i < d.max_size; i++){
        if (d.table[i].k != (K)-1){
            insert(dnew, d.table[i].k, d.table[i].v);
        }
    }
    delete d.table;
    d = dnew;
}


template <typename K, typename V>
void insert(struct dict<K, V>& d, K k, V v){
    if (d.size >= d.max_size / 2){
        rehash(d);
    }
    int loc = k % d.max_size;
    while (1) {
        if (d.table[loc].k != -1){
            loc = (loc + 1) % d.max_size;
            continue;
        }
        d.table[loc].k = k;
        d.table[loc].v = v;
        break;
    }
    s.size++;
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
        // so we surely gonna bump into -1
        if (d.table[loc].k == -1){
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
std::ostream& operator<<(std::ostream& os, struct dict<K, V>& d){
    os << "{";
    bool first = true;
    for (int i{0}; i < d.max_size; i++){
        if (d.table[i].k != -1){
            if (first) {
                os << d.table[i].k << ": " << d.table[i].v;
                first = false;
            } else {
                os << ", " << d.table[i].k << ": " << d.table[i].v;
            }
        }
    }
    os << "}" << std::endl;
    return os;
}