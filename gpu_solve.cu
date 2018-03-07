#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 16

void save_to_file(double *AB, const int a_size) {
    FILE *f = fopen("out.txt", "w+");

    fprintf(f, "%d\n", a_size);

    for(int i = 0; i < a_size*(a_size + 1); i++) {
        if((i + 1) % (a_size+1) == 0) {
            fprintf(f, "\n");
            continue;
        }
        fprintf(f, "%lf ", AB[i]);
    }

    for(int i = 0; i < a_size; i++)
        fprintf(f, "%lf ", AB[a_size + i*(a_size + 1)]);
    fprintf(f, "\n");

    fclose(f);
}

double *malloc_matrix(const int a, const int b) {
    return (double*)malloc(sizeof(double *)*a*b);
}

double *load_from_file(int *a_size, char *name) {
    FILE *f = fopen(name, "r");

    int size;
    fscanf(f, "%d", &size);
    double *matrix_ab = malloc_matrix(size, size + 1);

    for(int i = 0; i < size*(size + 1); i++) {
        if((i+1) % (size + 1) == 0)
            continue;
        fscanf(f, "%lf", &matrix_ab[i]);
    }

    for(int i = 0; i < size; i++) {
        fscanf(f, "%lf", &matrix_ab[size + i*(size + 1)]);
    }

    fclose(f);

    *a_size = size;

    return matrix_ab;
}


void print_matrix(double *matrix, const int a, const int b) {
    for(int i = 0; i < a*b; i++) {
        printf("%lf\t", matrix[i]);
        if((i+1) % b == 0)
            printf("\n");
    }
}

void print_output(double *AB, const int a_size) {
    printf("%d\n", a_size);

    for(int i = 0; i < a_size*(a_size + 1); i++) {
        if((i + 1) % (a_size+1) == 0) {
            printf("\n");
            continue;
        }
        printf("%lf\t", AB[i]);
    }

    for(int i = 0; i < a_size; i++)
        printf("%lf\t", AB[a_size + i*(a_size + 1)]);
    printf("\n");
}

int load_size() {
    int size;
    scanf("%d", &size);
    return size;
}

void load_ab_matrix(double *a, const int size) {
    for(int i = 0; i < size*(size + 1); i++) {
        if((i+1) % (size + 1) == 0)
            continue;
        scanf("%lf", &a[i]);
    }

    for(int i = 0; i < size; i++) {
        scanf("%lf", &a[size + i*(size + 1)]);
    }
}

double  *load_input(int *size) {
    *size = load_size();
    double *matrix_ab = malloc_matrix(*size, *size + 1);
    load_ab_matrix(matrix_ab, *size);
    return matrix_ab;
}


__global__ void replace_zero_gpu(double *AB, int rows, int columns, int column) {
    if(fabs(AB[column*columns + column]) <= 1e-4) {
        int row = column;
        for(; row < rows; row++) {
            if(fabs(AB[row*columns + column]) > 1e-4)
                break;
        }
        int threadId = blockDim.x*blockIdx.x + threadIdx.x;
        if(threadId + column >= columns)
            return;

        int zero = column*columns + column + threadId;
        int chosen = row*columns + column + threadId;
        AB[zero] += AB[chosen];
    }
}


__global__ void column_elimination_gpu(double *AB, int rows, int columns, int column) {
    int threadId = blockDim.x*blockIdx.x + threadIdx.x;
    if(threadId >= (rows - 1 - column)*(columns - column))
        return;

    int el_row = column + threadId/(columns - column) + 1;
    int el_col = column + threadId%(columns - column);
    int el = el_col + el_row*columns;
    int upper_el = el_col + column*columns;

    int main_el = column + column*columns;
    int main2_el = column + el_row*columns;
    double f = AB[main2_el]/AB[main_el];

    AB[el] -= f*AB[upper_el];
}

__global__ void multiple_column(double *AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;
    AB[(threadId * columns) + row] *= AB[columns*(row + 1) - 1];
}

__global__ void reverse_row_elimination(double *AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;
    int cols = columns - 2 - row;

    int start_index = row*columns + row + 1;

    int j = cols%2;
    for(int i = cols/2; i > 0; i/=2) {
        if(threadId >= i)
            return;

        AB[start_index + threadId] += (AB[start_index + threadId + i + j]);
        AB[start_index + threadId + i + j] = 0;
        if(j == 1)
            i++;
        j = i%2;
        __syncthreads();
    }

    int x_el = (row + 1)*columns - 1;
    int diag_el = row*columns + row;

    if(diag_el + 1 != x_el) {
        AB[x_el] -= AB[diag_el + 1];
        AB[diag_el + 1] = 0.0;
    }

    AB[x_el] /= AB[diag_el];
    AB[diag_el] = 1.0;
}

__global__ void sum_row(double *AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;

    int j = columns%2;
    for(int i = columns/2; i > 0; i/=2) {
        if(threadId >= i)
            return;

        AB[threadId] += AB[threadId + i + j];
        __syncthreads();
        if(j == 1)
            i++;
        j = i%2;
    }
}


void start_gaussian_elimination_gpu(double *AB, int rows, int cols) {
    double *AB_gpu;

    cudaMalloc(&AB_gpu, sizeof(double)*rows*cols);
    cudaMemcpy(AB_gpu, (void*)AB, sizeof(double)*rows*cols, cudaMemcpyHostToDevice);

    int block_size;

    for(int column = 0; column < cols - 1; column++) {
        block_size = (cols - column - 1)/THREADS_PER_BLOCK + 1;
        replace_zero_gpu<<<block_size, THREADS_PER_BLOCK>>>(AB_gpu, rows, cols, column);
        cudaThreadSynchronize();

        block_size = ((rows - column)*(cols - column) - 1)/THREADS_PER_BLOCK + 1;
        column_elimination_gpu<<<block_size, THREADS_PER_BLOCK>>>(AB_gpu, rows, cols, column);
        cudaThreadSynchronize();
    }

    for(int row = rows - 1; row >= 0; row--) {
        reverse_row_elimination<<<1, cols>>>(AB_gpu, rows, cols, row);
        multiple_column<<<1, row>>>(AB_gpu, rows, cols, row);

        cudaThreadSynchronize();
    }

    cudaMemcpy(AB, (void*)AB_gpu, sizeof(double)*rows*cols, cudaMemcpyDeviceToHost);

    cudaFree(AB_gpu);
}


int main(int argc, char ** argv) {
    int size;
    double *AB = load_from_file(&size, argv[1]);

    print_output(AB, size);

    start_gaussian_elimination_gpu(AB, size, size + 1);

    printf("\n\n");

    print_output(AB, size);

    save_to_file(AB, size);

    return 0;
}
