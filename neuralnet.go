package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// sigmoid function used for activation
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// derivative of sigmoid function for back propagation
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func getRandGen() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
	layerConfigs := nn.config.hiddenLayerConfigs

	var layers []*layer
	var hiddenLayers []*layer

	for _, config := range layerConfigs {
		hiddenLayers = append(hiddenLayers, newLayer(config))
	}

	lastLayer := layerConfigs[len(layerConfigs)-1]

	outputLayer := newLayer(layerConfig{inputNeurons: lastLayer.neurons, neurons: nn.config.outputNeurons})

	layers = append(layers, hiddenLayers...)
	layers = append(layers, outputLayer)

	if err := nn.backpropagate(x, y, layers); err != nil {
		return err
	}
	nn.hiddenLayers = hiddenLayers
	nn.outputLayer = outputLayer

	return nil
}

// activationLayers[i] stores the output of layers[i]. inputFeatures is the
// input to the first layer. deltaLayers[i] stores the error for layers[i].
func (nn *neuralNet) backpropagate(inputFeatures, labels *mat.Dense, layers []*layer) error {
	activationLayers := make([]*mat.Dense, len(layers))
	deltaLayers := make([]*mat.Dense, len(layers))

	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }

	for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
		// Forward pass: each layer applies weights and bias, then sigmoid.
		for idx, activations := range activationLayers {

			// First layer reads raw input; later layers read previous output.
			var inputs *mat.Dense
			if idx == 0 {
				inputs = inputFeatures
			} else {
				inputs = activationLayers[idx-1]
			}

			hl := layers[idx]
			applyBiases := func(_, col int, v float64) float64 { return v + hl.biases.At(0, col) }

			if activations == nil {
				activationLayers[idx] = new(mat.Dense)
				activations = activationLayers[idx]
			}

			activations.Mul(inputs, hl.weights)
			activations.Apply(applyBiases, activations)
			activations.Apply(applySigmoid, activations)
		}

		// Backward pass: start with output error, then pass each error
		// backward through the next layer's weights.
		// error for layer[x] is stored in the `deltaLayers[x]`
		for j := len(layers) - 1; j >= 0; j-- {
			propogated := new(mat.Dense)

			// Output error starts with label minus prediction.
			if j == len(layers)-1 {
				propogated.Sub(labels, activationLayers[j])
			} else {
				propogated.Mul(deltaLayers[j+1], layers[j+1].weights.T())
			}

			// Sigmoid derivative scales error by how much this output can change.
			slope := new(mat.Dense)
			slope.Apply(applySigmoidPrime, activationLayers[j])

			if deltaLayers[j] == nil {
				deltaLayers[j] = new(mat.Dense)
			}

			deltaLayers[j].MulElem(propogated, slope)
		}

		// Adjust weights using each layer's input and error. Adjust biases
		// using summed errors. These gradients sum across examples, not average.
		for layerIdx, layer := range layers {
			weightAdjustments := new(mat.Dense)
			if layerIdx == 0 {
				weightAdjustments.Mul(inputFeatures.T(), deltaLayers[layerIdx])
			} else {
				weightAdjustments.Mul(activationLayers[layerIdx-1].T(), deltaLayers[layerIdx])
			}
			weightAdjustments.Scale(nn.config.learningRate, weightAdjustments)

			layer.weights.Add(layer.weights, weightAdjustments)

			biasAdjustment, err := sumAlongAxis(0, deltaLayers[layerIdx])
			if err != nil {
				return err
			}
			biasAdjustment.Scale(nn.config.learningRate, biasAdjustment)
			layer.biases.Add(layer.biases, biasAdjustment)

		}

	}
	return nil
}

// sums a matrix along a particular dimension,
// preserving other dimensions
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := range numCols {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := range numRows {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}

// makes a prediction
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	if nn == nil || len(nn.hiddenLayers) == 0 || nn.outputLayer == nil {
		return nil, errors.New("network is not trained")
	}
	if x == nil {
		return nil, errors.New("input is nil")
	}

	_, inputCols := x.Dims()
	expectedCols, _ := nn.hiddenLayers[0].weights.Dims()
	if inputCols != expectedCols {
		return nil, fmt.Errorf(
			"input has %d features; want %d",
			inputCols, expectedCols,
		)
	}
	var layers []*layer
	layers = append(layers, nn.hiddenLayers...)
	layers = append(layers, nn.outputLayer)

	activationLayers := make([]*mat.Dense, len(layers))

	for i, hl := range layers {
		activation := new(mat.Dense)

		if i == 0 {
			activation.Mul(x, hl.weights)
		} else {
			activation.Mul(activationLayers[i-1], hl.weights)
		}
		activation.Apply(func(_ int, col int, v float64) float64 { return sigmoid(v + hl.biases.At(0, col)) }, activation)
		activationLayers[i] = activation
	}

	return activationLayers[len(layers)-1], nil
}
