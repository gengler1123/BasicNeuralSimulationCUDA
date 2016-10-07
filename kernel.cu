
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#include <ctime>
#include <iostream>

__global__ void NeuronTimestep(
	int numNeur,
	int numExcit,
	float *d_v,
	float *d_u,
	float *d_I,
	bool *d_cf,
	float *d_driven)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numExcit)
	{
		d_cf[i] = false;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (0.7 * (d_v[i] + 60)*(d_v[i] + 40) - d_u[i] + d_I[i] + d_driven[i]) / 100;
			float du = (0.03 * (-2*(d_v[i] + 60) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 35)
			{
				d_cf[i] = true;
				d_v[i] = -50;
				d_u[i] += 100;
				break;
			}
			d_I[i] = 0;
		}


	}
	else if (i < numNeur)
	{

		d_cf[i] = false;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (1.2 * (d_v[i] + 75)*(d_v[i] + 45) - d_u[i] + d_I[i] + d_driven[i]) / 150;
			float du = (0.01 * (5 * (d_v[i] + 75) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 50)
			{
				d_cf[i] = true;
				d_v[i] = -56;
				d_u[i] += 130;
				break;
			}
		}

		d_I[i] = 0;
	}

}


__global__ void CommunicationPhase(
	int numEdge,
	bool *d_cf,
	int *d_source,
	int *d_target,
	float *d_weight,
	float *d_I)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numEdge)
	{
		if (d_cf[d_source[i]])
		{
			atomicAdd(&d_I[d_target[i]], d_weight[i]);
		}
	}
}


int main()
{
	int numNeurons = 1000;
	int numExcit = 800;
	int T = 2000;
	int equilizationTime = 100;
	int transientTime = 300;

	/* CUDA Parameters */
	int numThreads = 512;

	/* Neurons */
	float *h_v, *d_v, *h_u, *d_u, *h_I, *d_I, *h_driven, *d_driven;
	bool *d_cf, *h_cf;

	h_v = new float[numNeurons];
	h_u = new float[numNeurons];
	h_I = new float[numNeurons];
	h_cf = new bool[numNeurons];
	h_driven = new float[numNeurons];

	bool **SpikeTrainYard = new bool*[T];
	float **VoltageTrace = new float *[T];
	for (int i = 0; i < numNeurons; i++)
	{
		h_v[i] = -60;
		h_u[i] = 0;
		h_I[i] = 0;
		h_cf[i] = false;
		if (i < 100)
		{
			h_driven[i] = 75;
		}
		else
		{
			h_driven[i] = 0;
		}
	}

	for (int t = 0; t < T; t++)
	{
		SpikeTrainYard[t] = new bool[numNeurons];
		VoltageTrace[t] = new float[numNeurons];
	}


	/* Edges */

	std::vector<int> h_source; int *d_source;
	std::vector<int> h_target; int *d_target;
	std::vector<float> h_weight; float *d_weight;
	
	std::mt19937 rd(time(NULL));
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	for (int n = 0; n < numNeurons; n++)
	{
		for (int m = 0; m < numNeurons; m++)
		{
			if (n != m)
			{

				if (dist(rd) < .2)
				{
					h_source.push_back(n);
					h_target.push_back(m);
					if (n < numExcit)
					{
						h_weight.push_back(dist(rd) * 300);
					}
					else
					{
						h_weight.push_back(dist(rd) * -400);
					}
				}

			}
		}
	}

	int numEdges = h_source.size();

	/* CUDA Memory Functions */

	cudaMalloc((void**)&d_v, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_u, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_I, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_driven, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_cf, numNeurons * sizeof(bool));

	
	cudaMalloc((void**)&d_source, numEdges * sizeof(int));
	cudaMalloc((void**)&d_target, numEdges * sizeof(int));
	cudaMalloc((void**)&d_weight, numEdges * sizeof(float));


	cudaMemcpy(d_v, h_v, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, h_u, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_driven, h_driven, numNeurons * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_source, h_source.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_target, h_target.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, h_weight.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);


	/* Run Simulation */

	for (int t = 0; t < equilizationTime; t++)
	{
		/* Run Timesteps, No Communication */
		NeuronTimestep <<<(numNeurons + numThreads - 1) / numThreads , numThreads >>>(
			numNeurons,
			numExcit,
			d_v,
			d_u,
			d_I,
			d_cf,
			d_driven);
	}

	for (int t = 0; t < transientTime; t++)
	{
		/* Run Timesteps, Communication, No Writing */
		NeuronTimestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			numExcit,
			d_v,
			d_u,
			d_I,
			d_cf,
			d_driven);

		CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
			numEdges,
			d_cf,
			d_source,
			d_target,
			d_weight,
			d_I);

	}

	for (int t = 0; t < T; t++)
	{
		/* Run Timesteps, Communication, Write Results*/
		NeuronTimestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			numExcit,
			d_v,
			d_u,
			d_I,
			d_cf,
			d_driven);

		CommunicationPhase<<<(numEdges + numThreads - 1) / numThreads, numThreads>>>(
			numEdges,
			d_cf,
			d_source,
			d_target,
			d_weight,
			d_I);

		cudaMemcpy(SpikeTrainYard[t], d_cf, numNeurons * sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(VoltageTrace[t], d_v, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	}


	/* Analyzing Run */

	std::vector<std::vector<int>> Firings;

	for (int t = 0; t < T; t++)
	{
		for (int n = 0; n < numNeurons; n++)
		{
			if (SpikeTrainYard[t][n] == true)
			{
				std::vector<int> v;
				v.push_back(t);
				v.push_back(n);
				Firings.push_back(v);
			}
		}
	}

	std::cout << "There were " << Firings.size() << " firings." << std::endl;


	/* Clean Up Code */

	cudaDeviceReset();

	for (int t = 0; t < T; t++)
	{
		delete[] SpikeTrainYard[t];
		delete[] VoltageTrace[t];
	}

	delete[] h_v; delete[] h_u; delete[] h_I; delete[] h_cf; delete[] SpikeTrainYard; delete[] h_driven;
	delete[] VoltageTrace;
    return 0;
}
