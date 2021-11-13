struct Graph** two_tree_embedding(int k);

typedef int* (*find_separator_t)(struct Graph*);

struct Graph* tree_cover(struct Graph*, find_separator_t find_separator);