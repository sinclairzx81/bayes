# bayes-ts

A simple naive bayes classifier written in typescript. This classifier was written specifically to support the streaming of large amounts of dynamic training data
(in the form of plain javascript objects) where training may happen over a long period of time and classification should be quick and efficient.

Useful for enabling classification services in real-time message based systems.

## build
```
npm install typescript -g
tsc bayes.ts 
```

## training

In the following example, we train the classifier with a basic weather dataset. After creating the classifier, we train it with plain javascript objects. The classifier will internally
treat each property on the object as a feature of the training set, and bin the data accordingly.

note: this algorithm only supports discrete values (i.e. fixed options represented as strings, think enums), and not continuous numeric values 
such as integers/floats etc. If using this library, you can encode your numerics as ranges (i.e. use "between 10 and 20" as a string and not the number 19). 
Future work on this library may include some form of linear regression for these types of values.

```javascript
let classifier = new NaiveBayes()
classifier.train({ outlook: "rainy",    temp: "hot",  humidity: "high",   windy: "no",  play_golf: "no" })
classifier.train({ outlook: "rainy",    temp: "hot",  humidity: "high",   windy: "yes", play_golf: "no" })
classifier.train({ outlook: "overcast", temp: "hot",  humidity: "high",   windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "sunny",    temp: "mild", humidity: "high",   windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "sunny",    temp: "cool", humidity: "normal", windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "sunny",    temp: "cool", humidity: "normal", windy: "yes", play_golf: "no" })
classifier.train({ outlook: "overcast", temp: "cool", humidity: "normal", windy: "yes", play_golf: "yes"})
classifier.train({ outlook: "rainy",    temp: "mild", humidity: "high",   windy: "no",  play_golf: "no" })
classifier.train({ outlook: "rainy",    temp: "cool", humidity: "normal", windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "sunny",    temp: "mild", humidity: "normal", windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "rainy",    temp: "mild", humidity: "normal", windy: "yes", play_golf: "yes"})
classifier.train({ outlook: "overcast", temp: "mild", humidity: "high",   windy: "yes", play_golf: "yes"})
classifier.train({ outlook: "overcast", temp: "hot",  humidity: "normal", windy: "no",  play_golf: "yes"})
classifier.train({ outlook: "sunny",    temp: "mild", humidity: "high",   windy: "yes", play_golf: "no" })
```
## classification / prediction

Once the classifier has been trained, it then becomes possible to classify known attributes of the training data. When classifying, 
the caller can expect a unit weighted result for each attribute on the feature being classified. The sum of all weights in the result
will total 1.

This library supports partial classification based on a subset of given feature / attributes. 

Below are some examples.

### playing golf today
```
let p = classifier.classify("play_golf", {
  outlook : "sunny",
  temp    : "cool",
  humidity: "high",
  windy   : "yes",
}) 
// = { no: 0.2285714285714286, 
//     yes: 0.7714285714285715 }
```
### weather outlook based on humidity.
```
let p = classifier.classify("outlook", { 
  humidity: "high" 
}) 
// = { rainy: 0.4285714285714286, 
//     overcast: 0.28571428571428575, 
//     sunny: 0.28571428571428575 }
```
## persisting training data

Its possible to persist training data by JSON serializing the classifiers bin.

``` javascript
  let data = JSON.stringify(classifier.bin)
  // save data somehow.
```