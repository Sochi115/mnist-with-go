package main

import "gonum.org/v1/gonum/mat"

// Contains all of the information that defines a trained neural network
type neuralNet struct {
	config       neuralNetConfig
	hiddenLayers []*layer
	outputLayer  *layer
}

// defines the architecture and learning params
type neuralNetConfig struct {
	inputNeurons       int
	outputNeurons      int
	hiddenLayerConfigs []layerConfig
	numEpochs          int
	learningRate       float64
}

type layer struct {
	biases  *mat.Dense
	weights *mat.Dense
}

type layerConfig struct {
	inputNeurons int
	neurons      int
}

func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

func newLayer(lcfg layerConfig) *layer {
	randGen := getRandGen()

	weights := mat.NewDense(lcfg.inputNeurons, lcfg.neurons, nil)
	biases := mat.NewDense(1, lcfg.neurons, nil)

	for _, param := range [][]float64{
		weights.RawMatrix().Data, biases.RawMatrix().Data,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	return &layer{
		biases:  biases,
		weights: weights,
	}
}
