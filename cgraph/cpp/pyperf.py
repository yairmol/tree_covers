from time import time

def set_perf():
    size = 100000000
    # s = set()
    start = time()
    s = set(range(size))
    total = time() - start
    print("set built in", total)
    start = time()
    for i in range(size):
        i in s
    total = time() - start
    print("checked all items in set in", total, "seconds")
    start = time()
    for i in range(size, size + 1000):
        i in s
    total = time() - start
    print("checked 1000 items not in set in", total, "seconds")


def main():
    set_perf()


if __name__ == "__main__":
    main()