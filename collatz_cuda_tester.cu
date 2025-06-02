#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>



__device__ bool collatzConverge(uint64_t x) {
    // Executa a sequência Collatz em registradores locais até:
    //  a) x == 1    → retorna true (convergiu)
    //  b) x < start → retorna true (já caiu em intervalo testado)
    //  c) ou até um limite máximo de iterações → retorna false (suspenso)
    // Para detectar ciclo, opcionalmente podia guardar últimos K valores, mas isso complica.
    uint64_t orig = x;
    int maxIter = 10000;  // limite arbitrário para cada número
    for (int i = 0; i < maxIter; i++) {
        if ((x & 1) == 0) {
            x = x >> 1;
        } else {
            x = 3*x + 1;
        }
        if (x == 1 || x < orig) {
            return true;  // “convergiu” para 1 ou entrou em intervalo já testado
        }
        // Note: se x crescer > 2^63, overflow vai acontecer, mas ignoramos
        // Se quiser detectar overflow, checar antes de 3*x+1:
        // if (x > (UINT64_MAX-1)/3) return false; 
    }
    return false;  // excedeu maxIter sem convergir → tratamos como “não investigado totalmente”
}

__global__ void checkCollatzKernel(uint64_t startN, uint64_t range, int *flag) {
    // Cada thread testa um único valor de n:
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= range) return;
    uint64_t n = startN + idx;  // mapeia “idx” → valor natural
    if (!collatzConverge(n)) {
        // Usuario finalizador: se encontrou candidato “suspeito”, grava índice
        atomicExch(flag, 1);
    }
}


int main() {
    uint64_t faixaSize = 50'000'000;   // testamos 50M de cada vez (ajustável)
    uint64_t currentStart = 1;
    int *d_flag;
    int h_flag;

    cudaMalloc(&d_flag, sizeof(int));

    while (true) {
        // (1) Resetar flag
        h_flag = 0;
        cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

        // (2) Lançar kernel para essa faixa [currentStart .. currentStart + faixaSize - 1]
        int TPB = 256;
        int numBlocks = (faixaSize + TPB - 1) / TPB;
        checkCollatzKernel<<<numBlocks, TPB>>>(currentStart, faixaSize, d_flag);
        cudaDeviceSynchronize();

        // (3) Ler flag de volta
        cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_flag != 0) {
            printf("Encontrado candidato a contraexemplo na faixa [%llu .. %llu]\n",
                   currentStart, currentStart + faixaSize - 1);
            break;
        }

        // (4) Avançar para próxima faixa
        currentStart += faixaSize;
        printf("Faixa [%llu .. %llu] processada—continuando...\n",
               currentStart - faixaSize, currentStart - 1);

        // (Opcional) critério de parada extra para não rodar eternamente:
        // if (currentStart > LIMITE_SUPERIOR) break;
    }

    cudaFree(d_flag);
    return 0;
}
