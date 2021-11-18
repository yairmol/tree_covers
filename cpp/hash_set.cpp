#include "hash_set.h"


void rehash(struct set& s){
    int* old_table = s.table;
    int old_size = s.max_size;
    s.max_size = s.max_size * 2;
    s.size = 0;
    s.table = new int[s.max_size];
    for (int i = 0; i < s.max_size; i++){
        s.table[i] = -1;
    }
    // init(snew, s.max_size * 2);
    for (int i = 0; i < old_size; i++){
        if (old_table[i] != -1){
            insert(s, old_table[i]);
        }
    }
    delete old_table;
}


void insert(struct set& s, int e){
    if (s.size >= s.max_size / 2){
        rehash(s);
    }
    int loc = e % s.max_size;
    while (1) {
        if (s.table[loc] != -1){
            loc = (loc + 1) % s.max_size;
            continue;
        }
        s.table[loc] = e;
        break;
    }
    s.size++;
}


bool mem(struct set& s, int e){
    int loc = e % s.max_size;
    while (1) {
        // our table in set is always at least half empty
        // so we surely gonna bump into -1
        if (s.table[loc] == -1){
            return false;
        }
        if (s.table[loc] != e){
            loc = (loc + 1) % s.max_size;
            continue;
        }
        return true;
    }
}


void remove(struct set& s, int e){
    return;
}


std::ostream& operator<<(std::ostream& os, struct set& s) {
    os << "{";
    bool first = true;
    for (int i{0}; i < s.max_size; i++){
        if (s.table[i] != -1){
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