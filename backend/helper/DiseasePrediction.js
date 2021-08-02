import * as tf from "@tensorflow/tfjs";
import fs from "fs";
class DiseasePrediction {
  constructor() {
    // Define a model for linear regression.
    this.model;
  }

  compile = () => {
    this.model = tf.sequential();

    this.model.add(
      tf.layers.dense({ units: 128, inputShape: [445], activation: "relu" })
    );
    this.model.add(tf.layers.dropout({ rate: 0.2 }));

    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));

    this.model.add(tf.layers.dense({ units: 148, activation: "sigmoid" }));

    // Prepare the model for training: Specify the loss and the optimizer.
    this.model.compile({ loss: "binaryCrossentropy", optimizer: "adam" });
  };

  readJsonWeights = async () => {
    let json = null;
    try {
      json = JSON.parse(fs.readFileSync("./outputWeights.json", "UTF-8"));
    } catch (e) {}
    if (json != null) {
      let weights = [];
      for (let i = 0; i < json.length; i++) {
        let tensor = tf.tensor(Object.values(json[i].values), json[i].shape);
        weights.push(tensor);
      }
      this.model.setWeights(weights);
    }
  };

  writeWeights = async () => {
    let json = [];
    for (let i = 0; i < this.model.getWeights().length; i++) {
      json.push({
        values: this.model.getWeights()[i].dataSync(),
        shape: this.model.getWeights()[i].shape,
      });
    }

    let stringInput = JSON.stringify(json, null, 2);

    fs.writeFileSync("outputWeights.json", stringInput);
  };

  train = async (xArray, yArray) => {
    const xInputs = tf.tensor(xArray);
    const yInputs = tf.tensor(yArray);
    await this.model.fit(xInputs, yInputs, {
      batchSize: 1,
      epochs: 1,
    });
    this.writeWeights();
  };

  predict = (xInput) => {
    let tensor = this.model.predict(tf.tensor(xInput));
    const values = tensor.dataSync();
    const arr = Array.from(values);
    return arr;
  };
}

export default DiseasePrediction;
