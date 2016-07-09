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
var Parameter = (function () {
    function Parameter() {
    }
    return Parameter;
}());
var NaiveBayes = (function () {
    /**
     * creates a new classifier.
     * @returns {NaiveBayes}
     */
    function NaiveBayes(state) {
        this.state = state;
        this.state = this.state || {
            features: {},
            correlations: {}
        };
    }
    /**
     * trains this classifer with this object.
     * @param {any} the javascript object to train this classifier with.
     * @returns {void}
     */
    NaiveBayes.prototype.train = function (obj) {
        var _this = this;
        var parameters = Object.keys(obj).map(function (key) { return ({ feature: key, attribute: obj[key] }); });
        parameters.forEach(function (parameter) { return _this.insert_feature(parameter); });
        parameters.forEach(function (left) { return parameters.forEach(function (right) {
            if (left.feature === right.feature)
                return;
            _this.insert_correlation(left, right);
        }); });
    };
    /**
     * classifies this feature with the given object.
     * @param {string} the feature to classify.
     * @param {any} an optional feature
     * @returns {any} the bayes prediction for the given feature.
     */
    NaiveBayes.prototype.classify = function (feature, obj) {
        var _this = this;
        if (this.state.features[feature] === undefined) {
            return {};
        }
        else if (obj === undefined || Object.keys(obj).length === 0) {
            var sum_1 = Object.keys(this.state.features[feature])
                .map(function (attribute) { return _this.state.features[feature][attribute]; })
                .reduce(function (acc, count) { return acc + count; }, 0);
            return Object.keys(this.state.features[feature])
                .reduce(function (acc, attribute) {
                acc[attribute] = _this.state.features[feature][attribute] / sum_1;
                return acc;
            }, {});
        }
        else {
            var sums_1 = Object.keys(obj).reduce(function (sums, inner_feature) {
                sums[inner_feature] = Object.keys(_this.state.correlations[feature]).reduce(function (sum, attribute) {
                    if (_this.state.correlations[feature][attribute][inner_feature] !== undefined ||
                        _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] !== undefined) {
                        return sum + _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]];
                    }
                    else
                        return sum;
                }, 0);
                return sums;
            }, {});
            var result_1 = Object.keys(this.state.correlations[feature]).reduce(function (result, attribute) {
                var probabilities = Object.keys(obj).reduce(function (probabilities, inner_feature) {
                    if (_this.state.correlations[feature][attribute][inner_feature] !== undefined
                        || _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] !== undefined) {
                        probabilities[inner_feature] = _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] / sums_1[inner_feature];
                    }
                    else
                        probabilities[inner_feature] = 0;
                    return probabilities;
                }, {});
                result[attribute] = Object.keys(probabilities).reduce(function (acc, feature) { return acc * probabilities[feature]; }, 1);
                return result;
            }, {});
            var sum_2 = Object.keys(result_1).reduce(function (acc, attribute) { return acc + result_1[attribute]; }, 0);
            return Object.keys(result_1).reduce(function (acc, attribute) {
                acc[attribute] = sum_2 > 0 ? result_1[attribute] / sum_2 : 0;
                return acc;
            }, {});
        }
    };
    /**
     * inserts this feature into the feature map, and increments its occurance value +1
     * @param {Parameter} the feature/attribute pair.
     * @returns {void}
     */
    NaiveBayes.prototype.insert_feature = function (parameter) {
        if (this.state.features[parameter.feature] === undefined)
            this.state.features[parameter.feature] = {};
        if (this.state.features[parameter.feature][parameter.attribute] === undefined) {
            this.state.features[parameter.feature][parameter.attribute] = 1;
        }
        else
            this.state.features[parameter.feature][parameter.attribute] += 1;
    };
    /**
     * inserts this correlation in to the correlation map. increments its occurance value +1.
     * This function updates both left and right, feature/attribute pairs, which is a duplication
     * of data, but no more than representing the data in a ND matrix.
     * @param {Parameter} the left feature/attribute pair.
     * @param {Parameter} the right feature/attribute pair.
     * @returns {void}
     */
    NaiveBayes.prototype.insert_correlation = function (left, right) {
        var _this = this;
        var needs_update = false;
        if (this.state.correlations[left.feature] === undefined)
            this.state.correlations[left.feature] = {};
        if (this.state.correlations[left.feature][left.attribute] === undefined)
            this.state.correlations[left.feature][left.attribute] = {};
        if (this.state.correlations[left.feature][left.attribute][right.feature] === undefined)
            this.state.correlations[left.feature][left.attribute][right.feature] = {};
        if (this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] === undefined) {
            this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] = 1;
            needs_update = true;
        }
        else
            this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] += 1;
        if (needs_update === false)
            return;
        Object.keys(this.state.correlations).forEach(function (left_feature) {
            Object.keys(_this.state.correlations).forEach(function (right_feature) {
                if (left_feature === right_feature)
                    return;
                Object.keys(_this.state.correlations[left_feature]).forEach(function (left_attribute) {
                    Object.keys(_this.state.correlations[right_feature]).forEach(function (right_attribute) {
                        if (_this.state.correlations[left_feature] === undefined)
                            _this.state.correlations[left_feature] = {};
                        if (_this.state.correlations[left_feature][left_attribute] === undefined)
                            _this.state.correlations[left_feature][left_attribute] = {};
                        if (_this.state.correlations[left_feature][left_attribute][right_feature] === undefined)
                            _this.state.correlations[left_feature][left_attribute][right_feature] = {};
                        if (_this.state.correlations[left_feature][left_attribute][right_feature][right_attribute] === undefined)
                            _this.state.correlations[left_feature][left_attribute][right_feature][right_attribute] = 0;
                    });
                });
            });
        });
    };
    return NaiveBayes;
}());
