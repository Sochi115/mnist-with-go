package main

import (
	// "encoding/csv"
	"fmt"
	// "log"
	// "os"
	// "strconv"
	//
	// "gonum.org/v1/gonum/floats"
	// "gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("Parsing train data...")
	//
	// config := neuralNetConfig{
	// 	inputNeurons:  28 * 28,
	// 	outputNeurons: 10,
	// 	numEpochs:     500,
	// 	learningRate:  0.3,
	// }
	//
	// var layers []*layer
	//
	// layers = append(layers, newLayer(16, 28*28), newLayer(16, 16))
}

// inputs, labels := makeInputsAndLabels("data/train.csv")

// Define config
// config := neuralNetConfig{
// 	inputNeurons:  4,
// 	outputNeurons: 3,
// 	hiddenNeurons: 3,
// 	numEpochs:     500,
// 	learningRate:  0.3,
// }

//		fmt.Println("Training...")
//		// Training
//		network := newNetwork(config)
//		if err := network.train(inputs, labels); err != nil {
//			log.Fatal(err)
//		}
//
//		fmt.Println("Parsing test data...")
//		// Parse test data
//		testInputs, testLabels := makeInputsAndLabels("data/test.csv")
//
//		fmt.Println("Predicting...")
//		// Make predictions
//		predictions, err := network.predict(testInputs)
//		if err != nil {
//			log.Fatal(err)
//		}
//
//		// Calculate accuracy
//		var truePosNeg int
//		numPreds, _ := predictions.Dims()
//		for i := 0; i < numPreds; i++ {
//			labelRow := mat.Row(nil, i, testLabels)
//			var species int
//			for idx, label := range labelRow {
//				if label == 1.0 {
//					species = idx
//					break
//				}
//			}
//
//			if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
//				truePosNeg++
//			}
//		}
//
//		accuracy := float64(truePosNeg) / float64(numPreds)
//
//		fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
//	}
//
//	func makeInputsAndLabels(filename string) (*mat.Dense, *mat.Dense) {
//		f, err := os.Open(filename)
//		if err != nil {
//			log.Fatal(err)
//		}
//		defer f.Close()
//
//		reader := csv.NewReader(f)
//		reader.FieldsPerRecord = 7
//
//		rawCSVData, err := reader.ReadAll()
//		if err != nil {
//			log.Fatal(err)
//		}
//
//		inputsData := make([]float64, 4*len(rawCSVData))
//		labelsData := make([]float64, 3*len(rawCSVData))
//
//		var inputsIndex int
//		var labelsIndex int
//
//		for idx, record := range rawCSVData {
//			// skip header
//			if idx == 0 {
//				continue
//			}
//
//			// Loop over each value in a row
//			for i, val := range record {
//
//				parsedVal, err := strconv.ParseFloat(val, 64)
//				if err != nil {
//					log.Fatal(err)
//				}
//
//				if i == 4 || i == 5 || i == 6 {
//					labelsData[labelsIndex] = parsedVal
//					labelsIndex++
//					continue
//				}
//
//				inputsData[inputsIndex] = parsedVal
//				inputsIndex++
//			}
//		}
//
//		inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
//		labels := mat.NewDense(len(rawCSVData), 3, labelsData)
//		return inputs, labels
// }
