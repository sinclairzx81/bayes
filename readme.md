# bayes-ts

A naive bayes classifier written in typescript. This classifier was written specifically to support streaming large amounts of dynamic training data
(given in the form of plain javascript objects) in real-time. This classifier is able to learn new features at random, and can be trained with
non structured javascript objects of varying properties. The classifier runs with a low memory footprint, and makes persisting training data is 
straight forward.

Useful for enabling classification services in real-time message based systems.

## build
```
npm install typescript -g
tsc bayes.ts 
```

## training

In the following example, we train the classifier with a basic weather dataset. The classifier is passed a series 
of training samples as javascript objects.

note: this algorithm only supports quantized attribute values, and not continuous numeric values such as integers/floats etc. 
If using this library, you can encode your numerical data as strings. For example, instead of passing the number '25', you can quantize
this value as "between 20 and 30". 

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

Once the classifier has been trained, it then becomes possible to classify features.

### playing golf today

asks the likelyhood of playing golf with these conditions.

```
let p = classifier.classify("play_golf", {
  outlook : "sunny",
  temp    : "cool",
  humidity: "high",
  windy   : "yes",
})
// p = { no : 0.2285714285714286, 
//       yes: 0.7714285714285715 }
```
### weather outlook based on humidity

classifies the weather outlook based on only the humidity. other features are ignored in classification.

```
let p = classifier.classify("outlook", { 
  humidity: "high" 
}) 
// p = { rainy    : 0.4285714285714286, 
//       overcast : 0.28571428571428575, 
//       sunny    : 0.28571428571428575 }
```

### probability distribution

obtain a features probability distribution, ignoring all other features.

```
let p = classifier.classify("temp")

// p = { hot  : 0.2857142857142857,
//       mild : 0.42857142857142855,
//       cool : 0.2857142857142857 }
```

## persisting training data

Its possible to persist training data by saving the classifiers state.

``` javascript
  let state = classifier.state

  // save state as json...which can be loaded with...

  let classifier = new NaiveBayes(state)
```