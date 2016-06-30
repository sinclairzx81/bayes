/*--------------------------------------------------------------------------

bayes-ts - an implementation of a naive bayes classifier in typescript.

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
 * NaiveBayes:
 * implementation of the naive bayes algorythm for
 * classifying discrete values within a javascript
 * object. supports training arbituary data
 * with non uniform feature sets.
 */
var NaiveBayes = (function () {
    /**
     * constructs this classifier.
     * @param {any} this classifiers bin data. otherwise will create as empty.
     * @returns {Classifer}
     */
    function NaiveBayes(bin) {
        this.bin = bin;
        this.bin = bin || {};
    }
    /**
     * trains and encodes this javascript objects properties as features.
     * @param   {any} A javascript object whose properties should be finite strings.
     * @returns {void}
     */
    NaiveBayes.prototype.train = function (obj) {
        var _this = this;
        /**
         * detect
         * determimne if the object being passed is a new feature
         * or attribute. This would require a update to the bin
         * to apply this feature/attribute across the all features,
         * the application across the board is expensive, so this
         * can be considered a optimization step.
         */
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
        /**
         * update bin:
         * For consistency (and simplification), we initialize
         * new features / attributes as having 0 counts across
         * existing features. To achieve this, we need to scan
         * quite deep into the bin for a number of iterations,
         * therefore we only do this if we have detected that
         * we need to (see needs update)
         */
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
     * @param {any} and object that should correlate to the training object data.
     * @returns {any} the bayes prediction for the given feature.
     */
    NaiveBayes.prototype.classify = function (feature, obj) {
        var _this = this;
        // no attributes:
        //
        // if the caller attempts to classify with no obj, then
        // we take a different path and try and classify based
        // on the features attributes alone.
        if (obj === undefined || Object.keys(obj).length === 0) {
            var flat_result_1 = {};
            /**
             * sum attribute values.
             * here, we sum the occurances of the given features
             * attributes. This gives us a probability of finding
             * this attribute with no correlation to anything
             * else.
             */
            Object.keys(this.bin[feature]).forEach(function (la) {
                Object.keys(_this.bin[feature][la]).forEach(function (rf) {
                    var attribute = la;
                    if (flat_result_1[la] === undefined)
                        flat_result_1[la] = 0;
                    Object.keys(_this.bin[feature][la][rf]).forEach(function (ra) {
                        flat_result_1[la] += _this.bin[feature][la][rf][ra];
                    });
                });
            });
            /**
             * normalize and return.
             * for the benefit of the caller, we normalize
             * the bayes result such that all its probabilies
             * total exactly 1.
             */
            var sum_1 = Object.keys(flat_result_1).reduce(function (acc, attribute) { return acc + flat_result_1[attribute]; }, 0);
            return Object.keys(flat_result_1).reduce(function (acc, attribute) {
                acc[attribute] = flat_result_1[attribute] / sum_1;
                return acc;
            }, {});
        }
        else {
            /**
             * totals:
             * The attributes counts keeped under the feature need to be totalled.
             * To do this, we use the input object to address into the bin to
             * select the values stored there, we then reduce to a object
             * we can use later.
             */
            var totals_1 = Object.keys(obj).reduce(function (acc, key) {
                acc[key] = Object.keys(_this.bin[feature]).reduce(function (acc, attribute) {
                    return acc + _this.bin[feature][attribute][key][obj[key]];
                }, 0);
                return acc;
            }, {});
            /**
             * bayes probability:
             * Here, we compute the probability of each attribute, The
             * results of which are mapped the bayes result object.
             */
            var bayes_result_1 = Object.keys(this.bin[feature]).reduce(function (acc, attribute) {
                /**
                 * probability:
                 * compute the probability by dividing each attribute bin count
                 * by the total of all attributes.
                 */
                var probabilities = Object.keys(obj).reduce(function (acc, key) {
                    acc[key] = _this.bin[feature][attribute][key][obj[key]] / totals_1[key];
                    return acc;
                }, {});
                /**
                 * bayes rule:
                 * using the bayes rule, we multiply each probability to compute the
                 * likelyhood of this attribute.
                 */
                acc[attribute] = Object.keys(probabilities).reduce(function (acc, feature) { return acc * probabilities[feature]; }, 1);
                return acc;
            }, {});
            /**
             * normalize and return.
             * for the benefit of the caller, we normalize
             * the bayes result such that all its probabilies
             * total exactly 1.
             */
            var sum_2 = Object.keys(bayes_result_1).reduce(function (acc, attribute) { return acc + bayes_result_1[attribute]; }, 0);
            return Object.keys(bayes_result_1).reduce(function (acc, attribute) {
                acc[attribute] = bayes_result_1[attribute] / sum_2;
                return acc;
            }, {});
        }
    };
    return NaiveBayes;
}());
