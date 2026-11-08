package main

import (
	"errors"
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

// TODO: Need to revisit this. im getting lost with the indexes and loopings
// need to fully understand which values need to be combined with what

func (nn *neuralNet) backpropagate(inputFeatures, labels *mat.Dense, layers []*layer) error {
	// Activation for each layer
	activationLayers := make([]*mat.Dense, len(layers)+1)
	// Error for each layer
	deltaLayers := make([]*mat.Dense, len(layers))

	activationLayers[0] = inputFeatures

	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }

	for i := 0; i < nn.config.numEpochs; i++ {
		for idx, activations := range activationLayers {

			if idx == 0 {
				continue
			}

			hl := layers[idx-1]
			applyBiases := func(_, col int, v float64) float64 { return v + hl.biases.At(0, col) }

			if activations == nil {
				activationLayers[idx] = new(mat.Dense)
				activations = activationLayers[idx]
			}

			inputs := activationLayers[idx-1]
			activations.Mul(inputs, hl.weights)
			activations.Apply(applyBiases, activations)
			activations.Apply(applySigmoid, activations)
		}

		// Calculate errors and propogate
		for j := len(layers) - 1; j >= 0; j-- {
			// Specifically to calculate output error

			propogated := new(mat.Dense)

			if j == len(layers)-1 {
				propogated.Sub(labels, activationLayers[j+1])
			} else {
				propogated.Mul(deltaLayers[j+1], layers[j+1].weights.T())
			}

			slope := new(mat.Dense)
			slope.Apply(applySigmoidPrime, activationLayers[j+1])

			if deltaLayers[j] == nil {
				deltaLayers[j] = new(mat.Dense)
			}

			deltaLayers[j].MulElem(propogated, slope)
		}

		for i, layer := range layers {
			// - W[l] += learningRate / batchSize * A[l]ᵀ * D[l+1]
			// - b[l] += learningRate / batchSize * columnSums(D[l+1])
			weightAdjustments := new(mat.Dense)
			weightAdjustments.Mul(activationLayers[i].T(), deltaLayers[i])
			weightAdjustments.Scale(nn.config.learningRate, weightAdjustments)

			layer.weights.Add(layer.weights, weightAdjustments)

			biasAdjustment, err := sumAlongAxis(0, deltaLayers[i])
			if err != nil {
				return err
			}
			biasAdjustment.Scale(nn.config.learningRate, biasAdjustment)
			layer.biases.Add(layer.biases, biasAdjustment)

		}

	}
	return nil
}

//	func (nn *neuralNet) train(x, y *mat.Dense) error {
//		randSource := rand.NewSource(time.Now().UnixNano())
//		randGen := rand.New(randSource)
//
//		wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
//		bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
//		wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
//		bOut := mat.NewDense(1, nn.config.outputNeurons, nil)
//
//		wHiddenRaw := wHidden.RawMatrix().Data
//		bHiddenRaw := bHidden.RawMatrix().Data
//		wOutRaw := wOut.RawMatrix().Data
//		bOutRaw := bOut.RawMatrix().Data
//
//		for _, param := range [][]float64{
//			wHiddenRaw,
//			bHiddenRaw,
//			wOutRaw,
//			bOutRaw,
//		} {
//			for i := range param {
//				param[i] = randGen.Float64()
//			}
//		}
//
//		// Define output
//		output := new(mat.Dense)
//
//		// Use backpropagation
//		//
//		if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
//			return err
//		}
//
//		nn.wHidden = wHidden
//		nn.bHidden = bHidden
//		nn.wOut = wOut
//		nn.bOut = bOut
//		return nil
//	}
//
//	func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
//		// loop over the number of epochs utilizing backpropogation
//		//
//		for i := 0; i < nn.config.numEpochs; i++ {
//
//			hiddenLayerInput := new(mat.Dense)
//			hiddenLayerInput.Mul(x, wHidden)
//			addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
//			hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
//
//			hiddenLayerActivations := new(mat.Dense)
//			applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
//			hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
//
//			outputLayerInput := new(mat.Dense)
//			outputLayerInput.Mul(hiddenLayerActivations, wOut)
//			addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
//			outputLayerInput.Apply(addBOut, outputLayerInput)
//			output.Apply(applySigmoid, outputLayerInput)
//
//			// Complete the backpropagation
//			networkError := new(mat.Dense)
//			networkError.Sub(y, output)
//
//			applySigmoidPrime := func(_, _ int, v float64) float64 {
//				return sigmoidPrime(v)
//			}
//			slopeOutputLayer := new(mat.Dense)
//			slopeOutputLayer.Apply(applySigmoidPrime, output)
//			slopeHiddenLayer := new(mat.Dense)
//			slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)
//
//			dOutput := new(mat.Dense)
//			dOutput.MulElem(networkError, slopeOutputLayer)
//			errorAtHiddenLayer := new(mat.Dense)
//			errorAtHiddenLayer.Mul(dOutput, wOut.T())
//
//			dHiddenLayer := new(mat.Dense)
//			dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)
//
//			wOutAdj := new(mat.Dense)
//			wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
//			wOutAdj.Scale(nn.config.learningRate, wOutAdj)
//			wOut.Add(wOut, wOutAdj)
//
//			bOutAdj, err := sumAlongAxis(0, dOutput)
//			if err != nil {
//				return err
//			}
//			bOutAdj.Scale(nn.config.learningRate, bOutAdj)
//			bOut.Add(bOut, bOutAdj)
//
//			wHiddenAdj := new(mat.Dense)
//			wHiddenAdj.Mul(x.T(), dHiddenLayer)
//			wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
//			wHidden.Add(wHidden, wHiddenAdj)
//
//			bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
//			if err != nil {
//				return err
//			}
//			bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
//			bHidden.Add(bHidden, bHiddenAdj)
//		}
//
//		return nil
//	}
//
// sums a matrix along a particular dimension,
// preserving other dimensions
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}

//
// // makes a prediction
// func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
// 	if nn.wHidden == nil || nn.wOut == nil {
// 		return nil, errors.New("Weights are empty")
// 	}
// 	if nn.bHidden == nil || nn.bOut == nil {
// 		return nil, errors.New("Biases are empty")
// 	}
//
// 	output := new(mat.Dense)
//
// 	// Feed forward process
// 	hiddenLayerInput := new(mat.Dense)
// 	hiddenLayerInput.Mul(x, nn.wHidden)
// 	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
// 	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
//
// 	hiddenLayerActivations := new(mat.Dense)
// 	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
// 	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
//
// 	outputLayerInput := new(mat.Dense)
// 	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
// 	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
// 	outputLayerInput.Apply(addBOut, outputLayerInput)
// 	output.Apply(applySigmoid, outputLayerInput)
//
// 	return output, nil
// }
