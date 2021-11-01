#include <iostream>

#define N (1 << 12)
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        printf("cuda error %s at file %s in line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__device__
int get_value(){
    uint u = threadIdx.x;
    return u * 2;
}

__global__
void set_value(int *a){
    uint u = threadIdx.x;
    a[u] = get_value();
}

__global__
void matmul(const int *a, const int *b, int *c){
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < N; k++) {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

void print_matrix(int* M, int row_offset, int column_offset, int limit){
    for (int i = row_offset; i < row_offset + limit; i++) {
        for (int j = column_offset; j < column_offset + limit; j++) {
            printf("%d ", M[i * N + j]);
        }
        // printf("%d ", a[i]);
        printf("\n");
    }
}

int main(){
    int *a, *b, *c;
    int *deva, *devb, *devc;
    int i, j;
    int size = sizeof(int) * N * N;
    cudaEvent_t start, end;
    float time = 0;
    a = (int *)(calloc(N * N, sizeof(int)));
    b = (int *)(malloc(size));
    c = (int *)(malloc(size));
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            a[i * N + j] = b[i * N + j] = 1;
            c[i * N + j] = 0;
        }
    }

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    HANDLE_ERROR(cudaMalloc((void **)&deva, size));
    HANDLE_ERROR(cudaMalloc((void **)&devb, size));
    HANDLE_ERROR(cudaMalloc((void **)&devc, size));

    HANDLE_ERROR(cudaMemcpy(deva, a, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devb, b, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devc, c, size, cudaMemcpyHostToDevice));

    dim3 blocksize(N/32, N/32);
    dim3 threadsize(32, 32);
    cudaEventRecord(start);
    matmul<<<blocksize, threadsize>>>(deva, devb, devc);
    // set_value<<<1, N>>>(deva);
    cudaEventRecord(end);
    cudaMemcpy(c, devc, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(a, deva, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&time, start, end);
    printf("execution time of %d x %d matrix is %lf seconds\n", N, N, time/1000);
    print_matrix(c, 4, 7, 6);
    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    free(a);
    free(b);
    free(c);
    return 0;
}