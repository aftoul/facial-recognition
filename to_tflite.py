#!/usr/bin/env python3

#    Copyright 2021 Abdelfattah TOULAOUI
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Foobar is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf

with open('nn/vectorizer.json') as f:
    v = tf.keras.models.model_from_json(f.read())

v.load_weights('nn/vectorizer.h5')
v_converter = tf.lite.TFLiteConverter.from_keras_model(v)

v_converter.optimizations = [tf.lite.Optimize.DEFAULT]
v_converter.target_spec.supported_types = [tf.float16]

with open('vectorizer.tflite', 'wb') as f:
    tflite_model = v_converter.convert()
    f.write(tflite_model)

