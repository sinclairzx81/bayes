/*--------------------------------------------------------------------------

bayesian-ts - an implementation of a naive bayes classifier in typescript.

The MIT License (MIT)

Copyright (c) 2016 Haydn Paterson (sinclair) <haydn.developer@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---------------------------------------------------------------------------*/

/**
 * classification result.
 */
interface ClassifierResult {
  [attribute: string] : number
}

/**
 * interface for this and other classifiers.
 */
interface Classifier {
  train    (obj: any) : void
  classify (feature: string, obj: any) : ClassifierResult 
}

/**
 * Bayesian:
 * implementation of the naive bayes algorythm for
 * classifying discrete values within a javascript 
 * object. supports training arbituary data
 * with non uniform feature sets.
 */
class Bayesian implements Classifier {

  /**
   * constructs this classifier.
   * @param {any} this classifiers bin data. otherwise will create as empty.
   * @returns {Classifer}
   */
  constructor(private bin?: any) {
    this.bin = bin || {}
  }

  /**
   * trains and encodes this javascript objects properties as features.
   * @param   {any} A javascript object whose properties should be finite strings.
   * @returns {void}
   */
  public train    (obj: any) : void {

    /**
     * detect
     * determimne if the object being passed is a new feature 
     * or attribute. This would require a update to the bin 
     * to apply this feature/attribute across the all features,
     * the application across the board is expensive, so this
     * can be considered a optimization step.
     */
    let needsupdate = false
    let keys = Object.keys(obj)
    for(let i = 0; i < keys.length; i++){
      let feature   = keys[i]
      let attribute = obj[keys[i]]
      if(this.bin[feature] === undefined || 
         this.bin[feature][attribute] === undefined) {
        needsupdate = true
        break;
      }
    }

    /**
     * insert:
     * from the object passed in, we scan both left and right
     * keys and use them to address into the bin. If at any 
     * point we find undefined (as would be the case for new
     * features or attributes), we simply initialize it.
     * 
     * When we get the the count value at the end of the chain,
     * we set its value to 1 if not exists or simply increment
     * the value by one if it does. It is these counts that 
     * are used during classification.
     */
    Object.keys(obj).forEach(lk => {
      Object.keys(obj).forEach(rk => {
        if(lk === rk) return 
        if(this.bin[lk] === undefined) this.bin[lk] = {}
        if(this.bin[lk][obj[lk]] === undefined) this.bin[lk][obj[lk]] = {}
        if(this.bin[lk][obj[lk]][rk] === undefined) this.bin[lk][obj[lk]][rk] = {}
        if(this.bin[lk][obj[lk]][rk][obj[rk]] === undefined) this.bin[lk][obj[lk]][rk][obj[rk]] = 1
        else this.bin[lk][obj[lk]][rk][obj[rk]] += 1
      })
    })

    /**
     * update bin:
     * For consistency (and simplification), we initialize
     * new features / attributes as having 0 counts across
     * existing features. To achieve this, we need to scan
     * quite deep into the bin for a number of iterations,
     * therefore we only do this if we have detected that
     * we need to (see needs update)
     */
    if(needsupdate) {
      Object.keys(this.bin).forEach(lf => {
        Object.keys(this.bin).forEach(rf => {
          if(lf === rf) return;
          Object.keys(this.bin[lf]).forEach(la => {
            Object.keys(this.bin[rf]).forEach(ra => {
              if(this.bin[lf] === undefined) this.bin[lf] = {}
              if(this.bin[lf][la] === undefined) this.bin[lf][la] = {}
              if(this.bin[lf][la][rf] === undefined) this.bin[lf][la][rf] = {}
              if(this.bin[lf][la][rf][ra] === undefined) this.bin[lf][la][rf][ra] = 0
            })
          })
        })
      }) 
    }
  }

  /**
   * classifies this feature with the given object.
   * @param {string} the feature to classify.
   * @param {any} and object that should correlate to the training object data.
   * @returns {any} the bayesian prediction for the given feature.
   */
  public classify (feature: string, obj?: any) : ClassifierResult {

    // no attributes:
    //
    // if the caller attempts to classify with no obj, then
    // we take a different path and try and classify based
    // on the features attributes alone.
    if(obj === undefined || Object.keys(obj).length === 0) {
      let flat_result = {}
      
      /**
       * sum attribute values.
       * here, we sum the occurances of the given features
       * attributes. This gives us a probability of finding
       * this attribute with no correlation to anything 
       * else.
       */      
      Object.keys(this.bin[feature]).forEach(la => {
        Object.keys(this.bin[feature][la]).forEach(rf => {
          let attribute = la
          if(flat_result[la] === undefined) flat_result[la] = 0
          Object.keys(this.bin[feature][la][rf]).forEach(ra => {
           flat_result[la] += this.bin[feature][la][rf][ra]
          })
        })
      })
      /**
       * normalize and return.
       * for the benefit of the caller, we normalize
       * the bayesian result such that all its probabilies
       * total exactly 1.
       */
      let sum = Object.keys(flat_result).reduce((acc, attribute) => acc + flat_result[attribute], 0)
      return Object.keys(flat_result).reduce((acc, attribute) => {
        acc[attribute] = flat_result[attribute] / sum
        return acc
      }, {}) as ClassifierResult

    } else {

      /**
       * totals:
       * The attributes counts keeped under the feature need to be totalled.
       * To do this, we use the input object to address into the bin to 
       * select the values stored there, we then reduce to a object
       * we can use later.
       */
      let totals = Object.keys(obj).reduce((acc, key) => {
        acc[key] = Object.keys(this.bin[feature]).reduce((acc, attribute) => {
          return acc + this.bin[feature][attribute][key][obj[key]]
        }, 0); return acc
      }, {})

      /**
       * bayesian probability:
       * Here, we compute the probability of each attribute, The 
       * results of which are mapped the bayes result object.
       */
      let bayesian_result = Object.keys(this.bin[feature]).reduce((acc, attribute) => {
        /**
         * probability:
         * compute the probability by dividing each attribute bin count 
         * by the total of all attributes.
         */
        let probabilities = Object.keys(obj).reduce((acc, key) => {
          acc[key] = this.bin[feature][attribute][key][obj[key]] / totals[key]
          return acc
        }, {})
          
        /**
         * bayesian rule:
         * using the bayes rule, we multiply each probability to compute the 
         * likelyhood of this attribute.
         */    
        acc[attribute] = Object.keys(probabilities).reduce((acc, feature) => acc * probabilities[feature], 1)
        return acc
      }, {})

      /**
       * normalize and return.
       * for the benefit of the caller, we normalize
       * the bayesian result such that all its probabilies
       * total exactly 1.
       */
      let sum = Object.keys(bayesian_result).reduce((acc, attribute) => acc + bayesian_result[attribute], 0)
      return Object.keys(bayesian_result).reduce((acc, attribute) => {
        acc[attribute] = bayesian_result[attribute] / sum
        return acc
      }, {}) as ClassifierResult
    }
  }
}