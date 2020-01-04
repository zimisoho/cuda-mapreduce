#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define GRID_SIZE 100
#define BLOCK_SIZE 100

#define MAX_INPUT_COUNT 10
#define MAX_OUTPUT_COUNT 10
#define MAX_WORD_COUNT 20
#define NUM_KEYS 100 
typedef struct tagWord {
	char szWord[30];
}Word;
typedef struct tagKeyPair {
	char szWord[30];
	int nCount;
}KeyPair;

typedef struct tagInputData {
	Word pWord[20];
	int nWordCount;
}InputData;
typedef KeyPair OutputData;

typedef struct tagKeyValueData {
	KeyPair pKeyPair[NUM_KEYS];	
	int nCount;
}KeyValueData;

int g_nKeyData = 0;

__device__ void mapper(InputData *input, KeyValueData *keyData)
{
	int nWordCount = input->nWordCount;
	keyData->nCount = nWordCount;
	for (int i = 0; i<nWordCount; i++) {
		KeyPair* keyPair = &keyData->pKeyPair[i];
		int j = 0;		

		char* p = input->pWord[i].szWord;
		while(*p != 0)
		{ 
			keyPair->szWord[j] = *p;
			p++;
			j++;
		}		
		keyPair->nCount = 1;
	}
}


__device__ bool compare(char* sz1, char* sz2) {
	char* p2 = sz2;
	int n = 0;
	while (*p2 != 0) {
		if (sz1[n] != *p2)
			return false;
		p2++;
		n++;
	}
	return true;
}

__device__ void reducer(KeyValueData *keyData, OutputData *output, int nInputCount, int* nOutputCount)
{
	int nIndex = 0;
	for (int index = 0; index < nInputCount; index++) {
		KeyValueData* pKeyData = &keyData[index];
		int nCount = pKeyData->nCount;
		for (int i = 0; i<nCount; i++) {

			bool bExist = false;
			int j = 0;

			KeyPair keyPair = pKeyData->pKeyPair[i];
			for (int j = 0; j <= nIndex; j++)
			{
				if (compare(output[j].szWord, keyPair.szWord))
				{
					output[j].nCount++;
					bExist = true;
					break;
				}
			}
			if (bExist)
				continue;

			j = 0;

			char* p = keyPair.szWord;
			//printf("Word2:%s\n", keyPair.szWord);
			while (*p != 0)
			{
				output[nIndex].szWord[j] = *p;
				p++;
				j++;
			}
			output[nIndex].nCount = 1;
			nIndex++;
		}
	}
	*nOutputCount = nIndex;
}
__global__ void mapKernel(InputData *input, int nInputCount, KeyValueData *pairs)
{
	int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
	int gridStride = gridDim.x * blockDim.x;
	for (int i = indexWithinTheGrid; i < nInputCount; i += gridStride)
	{
		mapper(&input[i], &pairs[i]);
	}
}

__global__ void reduceKernel(KeyValueData *pairs, int nInputCount, OutputData *output, int* nOutputCount) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < 1; i += blockDim.x * gridDim.x) {
		reducer(pairs, output, nInputCount, nOutputCount);
	}
}
void cudaMap(InputData *input, int nInputCount, KeyValueData *pairs) {
	mapKernel << <GRID_SIZE, BLOCK_SIZE >> >(input, nInputCount, pairs);

}

void cudaReduce(KeyValueData *pairs, int nInputCount, OutputData *output, int* nOutputCount) {
	reduceKernel << <GRID_SIZE, BLOCK_SIZE >> >(pairs, nInputCount, output, nOutputCount);

}
void runMapReduce(InputData *input, int nInputCount, OutputData *output, int* nOutputCount) {
	InputData   *dev_input;
	OutputData  *dev_output;
	KeyValueData *dev_pairs;
	int			*dev_count;
	size_t input_size = nInputCount * sizeof(InputData);
	size_t output_size = MAX_OUTPUT_COUNT * sizeof(OutputData);
	size_t pairs_size = nInputCount * sizeof(KeyValueData);
	

	cudaMalloc(&dev_input, input_size);
	cudaMalloc(&dev_pairs, pairs_size);
	cudaMalloc(&dev_count, sizeof(int));

	cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

	cudaMap(dev_input, nInputCount, dev_pairs);

	cudaFree(dev_input);

	cudaMalloc(&dev_output, output_size);
	cudaMemset(dev_output, 0, output_size);
	cudaReduce(dev_pairs, nInputCount, dev_output, dev_count);

	cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nOutputCount, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_count);
	cudaFree(dev_pairs);
	cudaFree(dev_output);
}

int main(int argc, char const *argv[])
{
	printf("Input Data:\n");
	FILE* pFile = fopen("test.txt", "rt");
	char c;
	// Input Text and Splitting
	InputData* pInputData = (InputData*)malloc(MAX_INPUT_COUNT*sizeof(InputData));
	OutputData* pOutputData = (OutputData*)malloc(MAX_OUTPUT_COUNT * sizeof(OutputData));
	int nInputCount = 0;
	int nOutputCount = 0;
	if (pFile) {
		char szLine[100];
		while(!feof(pFile))
		{
			memset(szLine, 0, 100);
			fgets(szLine, 100, pFile);
			printf("%s", szLine);
			pInputData[nInputCount].nWordCount = 0;
			int nWordIndex = 0;
			int nIndex = 0;
			char szWord[20];
			if(strlen(szLine)<2)
				break;
			for (int i = 0; i < strlen(szLine); i++) {				
				if (szLine[i] == ' ' || szLine[i] == 0x0d || szLine[i] == 0x0A)
				{
					szWord[nIndex] = 0;
					nIndex = 0;
					strcpy(pInputData[nInputCount].pWord[nWordIndex].szWord, szWord);
					pInputData[nInputCount].nWordCount++;					
					nWordIndex++;
					if (szLine[i] == 0x0d)
						break;
				}
				else {
					szWord[nIndex] = szLine[i];
					nIndex++;
				}
			}
			nInputCount++;
		}
		fclose(pFile);
	}
	// Splitting End
	int nTotalCount = 0;
	runMapReduce(pInputData, nInputCount, pOutputData, &nOutputCount);
	printf("--------------------------\n");
	
	for (int i = 0; i < nOutputCount; i++) {
		printf("%s\t%d\n", pOutputData[i].szWord, pOutputData[i].nCount);
		nTotalCount += pOutputData[i].nCount;
	}
	// Output WordCount	
	printf("--------------------------\n");
	printf("Total Count:%d\n", nTotalCount);
	free(pOutputData);
	free(pInputData);
	return 0;
}
