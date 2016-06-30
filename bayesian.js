/*--------------------------------------------------------------------------

bayesian-ts - an implementation of naive bayes in typescript.

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
 * Bayesian:
 * implementation of the naive bayes algorythm for
 * classifying discrete values within a javascript
 * object. supports training arbituary data
 * with non uniform feature sets.
 */
var Bayesian = (function () {
    function Bayesian() {
        /**
         * bin encoding for attribute counts.
         * count = bin[feature][attribute][feature][attribute]
         */
        this.bin = {};
    }
    /**
     * trains the classifier with this object.
     * @param   {any} any javascript object
     * @returns {void}
     */
    Bayesian.prototype.train = function (obj) {
        var _this = this;
        //--------------------------------------------------------------
        // detect: determimne if the object being passed is a new feature 
        //         or attribute. This would require a update to the bin 
        //         to apply this feature/attribute across the all features,
        //         the application across the board is expensive, so this
        //         can be considered a optimization step.
        //--------------------------------------------------------------
        var needsupdate = false;
        var keys = Object.keys(obj);
        for (var i = 0; i < keys.length; i++) {
            var feature = keys[i];
            var attribute = obj[keys[i]];
            if (this.bin[feature] === undefined ||
                this.bin[feature][attribute] === undefined) {
                needsupdate = true;
                break;
            }
        }
        //--------------------------------------------------------------
        // insert: increments the count for the feature/attribute pair.
        //--------------------------------------------------------------
        Object.keys(obj).forEach(function (lk) {
            Object.keys(obj).forEach(function (rk) {
                if (lk === rk)
                    return;
                if (_this.bin[lk] === undefined)
                    _this.bin[lk] = {};
                if (_this.bin[lk][obj[lk]] === undefined)
                    _this.bin[lk][obj[lk]] = {};
                if (_this.bin[lk][obj[lk]][rk] === undefined)
                    _this.bin[lk][obj[lk]][rk] = {};
                if (_this.bin[lk][obj[lk]][rk][obj[rk]] === undefined)
                    _this.bin[lk][obj[lk]][rk][obj[rk]] = 1;
                else
                    _this.bin[lk][obj[lk]][rk][obj[rk]] += 1;
            });
        });
        //--------------------------------------------------------------
        // update: ensures all attributes exist for all features.
        //--------------------------------------------------------------
        if (needsupdate) {
            Object.keys(this.bin).forEach(function (lf) {
                Object.keys(_this.bin).forEach(function (rf) {
                    if (lf === rf)
                        return;
                    Object.keys(_this.bin[lf]).forEach(function (la) {
                        Object.keys(_this.bin[rf]).forEach(function (ra) {
                            if (_this.bin[lf] === undefined)
                                _this.bin[lf] = {};
                            if (_this.bin[lf][la] === undefined)
                                _this.bin[lf][la] = {};
                            if (_this.bin[lf][la][rf] === undefined)
                                _this.bin[lf][la][rf] = {};
                            if (_this.bin[lf][la][rf][ra] === undefined)
                                _this.bin[lf][la][rf][ra] = 0;
                        });
                    });
                });
            });
        }
    };
    /**
     * classifies this feature with the given object.
     * @param {string} the feature to classify.
     * @param {any} and object directionary that should correlate to the training object data.
     * @returns {any} the bayesian prediction for the given feature.
     */
    Bayesian.prototype.classify = function (feature, obj) {
        var _this = this;
        //--------------------------------------------------------------
        // total: attributes counts for given feature.
        //--------------------------------------------------------------
        var totals = Object.keys(obj).reduce(function (acc, key) {
            acc[key] = Object.keys(_this.bin[feature]).reduce(function (acc, attribute) {
                return acc + _this.bin[feature][attribute][key][obj[key]];
            }, 0);
            return acc;
        }, {});
        //--------------------------------------------------------------
        // compute bayesian result
        //--------------------------------------------------------------
        var bayesian = Object.keys(this.bin[feature]).reduce(function (acc, attribute) {
            // compute the probability by dividing each attribute count by the total.
            var probabilities = Object.keys(obj).reduce(function (acc, key) {
                acc[key] = _this.bin[feature][attribute][key][obj[key]] / totals[key];
                return acc;
            }, {});
            //--------------------------------------------------------------      
            // multiply the probabilities to give the prediction for this attribute.
            //--------------------------------------------------------------      
            acc[attribute] = Object.keys(probabilities).reduce(function (acc, feature) {
                return acc * probabilities[feature];
            }, 1);
            return acc;
        }, {});
        //--------------------------------------------------------------      
        // normalize the probabilities to sum as 1.
        //-------------------------------------------------------------- 
        var sum = Object.keys(bayesian).reduce(function (acc, attribute) { return acc + bayesian[attribute]; }, 0);
        var result = Object.keys(bayesian).reduce(function (acc, attribute) {
            acc[attribute] = bayesian[attribute] / sum;
            return acc;
        }, {});
        return result;
    };
    return Bayesian;
}());
