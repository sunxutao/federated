# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for computations.py (and __init__.py)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core import api as tff


def tf1_and_tf2_test(test_func):
  """A decorator for testing TFF wrapping of TF.

  Args:
    test_func: A test function to be decorated. It must accept to arguments,
      self (a TestCase), and tf_computation, which is either
      tff.tf_computation or tff.tf2_computation. Optionally,
      the test_func may return something that can be compared using
      self.assertEqual.

  Returns:
    The decorated function, which executes test_func using both wrappers,
    and compares the results.
  """

  def test_tf1_and_tf2(self):
    tf2_result = test_func(self, tff.tf2_computation)
    with tf.Graph().as_default():
      tf1_result = test_func(self, tff.tf_computation)
    self.assertEqual(tf1_result, tf2_result)

  return test_tf1_and_tf2


class TensorFlowComputationsTest(test.TestCase):

  @tf1_and_tf2_test
  def test_tf_comp_first_mode_of_usage_as_non_polymorphic_wrapper(
      self, tf_computation):
    # Wrapping a lambda with a parameter.
    foo = tf_computation(lambda x: x > 10, tf.int32)
    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')
    self.assertEqual(foo(9), False)
    self.assertEqual(foo(11), True)

    # Wrapping an existing Python function with a parameter.
    bar = tf_computation(tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(bar.type_signature), '(<int32,int32> -> int32)')

    # Wrapping a no-parameter lambda.
    baz = tf_computation(lambda: tf.constant(10))
    self.assertEqual(str(baz.type_signature), '( -> int32)')
    self.assertEqual(baz(), 10)

    # Wrapping a no-parameter Python function.
    def bak_fn():
      return tf.constant(10)

    bak = tf_computation(bak_fn)
    self.assertEqual(str(bak.type_signature), '( -> int32)')
    self.assertEqual(bak(), 10)

  def test_tf_fn_with_variable(self):
    # N.B. This does not work with TF 2 style serialization,
    # because a variable is created on a non-first call. See the TF2
    # style example below.

    @tff.tf_computation
    def read_var():
      v = tf.Variable(10, name='test_var')
      return v

    self.assertEqual(read_var(), 10)

  @tf1_and_tf2_test
  def test_tf_comp_second_mode_of_usage_as_non_polymorphic_decorator(
      self, tf_computation):
    # Decorating a Python function with a parameter.
    @tf_computation(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    self.assertEqual(foo(9), False)
    self.assertEqual(foo(10), False)
    self.assertEqual(foo(11), True)

    # Decorating a no-parameter Python function.
    @tf_computation
    def bar():
      return tf.constant(10)

    self.assertEqual(str(bar.type_signature), '( -> int32)')

    self.assertEqual(bar(), 10)

  @tf1_and_tf2_test
  def test_tf_comp_third_mode_of_usage_as_polymorphic_callable(
      self, tf_computation):
    # Wrapping a lambda.
    foo = tf_computation(lambda x: x > 0)

    self.assertEqual(foo(-1), False)
    self.assertEqual(foo(0), False)
    self.assertEqual(foo(1), True)

    # Decorating a Python function.
    @tf_computation
    def bar(x, y):
      return x > y

    self.assertEqual(bar(0, 1), False)
    self.assertEqual(bar(1, 0), True)
    self.assertEqual(bar(0, 0), False)

  @tf1_and_tf2_test
  def test_with_variable(self, tf_computation):

    v_slot = []

    @tf.function(autograph=False)
    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      tf.assign(v, 1)
      return v + x

    tf_comp = tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  @tf1_and_tf2_test
  def test_one_param(self, tf_computation):

    @tf.function
    def foo(x):
      return x + 1

    tf_comp = tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  @tf1_and_tf2_test
  def test_no_params_structured_outputs(self, tf_computation):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo():
      d = collections.OrderedDict([('foo', 3.0), ('bar', 5.0)])
      return (1, 2, d, MyType(True, False))

    tf_comp = tf_computation(foo, None)
    result = tf_comp()
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 2)
    # XXX We get a dict as expected with tf1 serialization, but
    # with tf2 serialization we get an AnonymousTuple.
    # See the note in graph_utils.py on
    # get_tf2_result_dict_and_binding on how to fix.
    # self.assertEqual(result[2], {'foo': 3.0, 'bar': 5.0})
    # self.assertEqual(result[3], MyType(True, False))

  @tf1_and_tf2_test
  def test_polymorphic(self, tf_computation):

    def foo(x, y, z=3):
      # Since we don't wrap this as a tf.function, we need to do some
      # tf.convert_to_tensor(...) in order to ensure we have TensorFlow types.
      x = tf.convert_to_tensor(x)
      y = tf.convert_to_tensor(y)
      return (x + y, tf.convert_to_tensor(z))

    tf_comp = tf_computation(foo)  # A polymorphic TFF function.

    self.assertEqual(tuple(iter(tf_comp(1, 2))), (3, 3))  # With int32
    self.assertEqual(tuple(iter(tf_comp(1.0, 2.0))), (3.0, 3.0))  # With float32
    self.assertEqual(tuple(iter(tf_comp(1, 2, z=3))), (3, 3))  # With z

  @tf1_and_tf2_test
  def test_simple_structured_input(self, tf_computation):

    def foo(t):
      return t[0] + t[1]

    tf_poly = tf_computation(foo)
    self.assertEqual(tf_poly((1, 2)), 3)

  @tf1_and_tf2_test
  def test_more_structured_input(self, tf_computation):

    @tf.function(autograph=False)
    def foo(tuple1, tuple2=(1, 2)):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    tf_poly = tf_computation(foo)
    self.assertEqual(tf_poly((1, (2, 3))), 9)
    self.assertEqual(tf_poly((1, (2, 3)), (0, 0)), 6)

  def test_py_and_tf_args(self):

    @tf.function(autograph=False)
    def foo(x, y, add=True):
      return x + y if add else x - y

      # Note: tf.Functions support mixing tensorflow and Python arguments,
      # usually with the semantics you would expect. Currently, TFF does not
      # support this kind of mixing, even for Polymorphic TFF functions.
      # However, you can work around this by explicitly binding any Python
      # arguments on a tf.Function:

    tf_poly_add = tff.tf_computation(lambda x, y: foo(x, y, True))
    tf_poly_sub = tff.tf_computation(lambda x, y: foo(x, y, False))
    self.assertEqual(tf_poly_add(2, 1), 3)
    self.assertEqual(tf_poly_add(2., 1.), 3.)
    self.assertEqual(tf_poly_sub(2, 1), 1)

  # TODO(b/123193055): Enable this for tff.tf_computation as well
  def test_more_structured_input_explicit_types(self):

    @tf.function(autograph=False)
    def foo(tuple1, tuple2=(1, 2)):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    tff_type1 = [
        (tf.int32, (tf.int32, tf.int32)),
        ('tuple2', (tf.int32, tf.int32))]
    tff_type2 = [
        (tf.int32, (tf.int32, tf.int32)),
        (tf.int32, tf.int32)]

    # Both of the above are valid.
    # XXX Q - Should explicitly naming 'tuple1' also work? (It doesn't)
    for tff_type in [tff_type1, tff_type2]:
      tf_comp = tff.tf2_computation(foo, tff_type)
      self.assertEqual(tf_comp((1, (2, 3)), (0, 0)), 6)

  def test_something_that_only_works_with_tf2(self):
    # These variables will be tracked and serialized automatically.
    v1 = tf.Variable(0.0)
    v2 = tf.Variable(0.0)

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def foo(x):
      tf.assign(v1, 1.0)
      tf.assign(v2, 1.0)
      return (v1 + v2, x)

    foo_cf = foo.get_concrete_function()

    @tf.function
    def bar(x):
      a, b = foo_cf(x)
      return a + b

    # If we had wrapped this test in @graph_mode_test and called
    # tff.tf_computation, this example will not work.
    tf2_comp = tff.tf2_computation(bar, tf.float32)
    self.assertEqual(tf2_comp(1.0), 3.0)


class TensorFlowComputationsWithDatasetsTest(test.TestCase):
   # TODO(b/122081673): Support tf.Dataset serialization in tf2_computation.

  def test_with_tf_datasets(self):

    @tff.tf_computation(tff.SequenceType(tf.int64))
    def foo(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(str(foo.type_signature), '(int64* -> int64)')

    @tff.tf_computation
    def bar():
      return tf.data.Dataset.range(10)

    self.assertEqual(str(bar.type_signature), '( -> int64*)')

    self.assertEqual(foo(bar()), 45)

  def test_with_sequence_of_pairs(self):
    pairs = tf.data.Dataset.from_tensor_slices(
        (list(range(5)), list(range(5, 10))))

    @tff.tf_computation
    def process_pairs(ds):
      return ds.reduce(0, lambda state, pair: state + pair[0] + pair[1])

    self.assertEqual(process_pairs(pairs), 45)

  def test_tf_comp_with_sequence_inputs_and_outputs_does_not_fail(self):

    @tff.tf_computation(tff.SequenceType(tf.int32))
    def _(x):
      return x

  def test_with_four_element_dataset_pipeline(self):

    @tff.tf_computation
    def comp1():
      return tf.data.Dataset.range(5)

    @tff.tf_computation(tff.SequenceType(tf.int64))
    def comp2(ds):
      return ds.map(lambda x: tf.cast(x + 1, tf.float32))

    @tff.tf_computation(tff.SequenceType(tf.float32))
    def comp3(ds):
      return ds.repeat(5)

    @tff.tf_computation(tff.SequenceType(tf.float32))
    def comp4(ds):
      return ds.reduce(0.0, lambda x, y: x + y)

    @tff.tf_computation
    def comp5():
      return comp4(comp3(comp2(comp1())))

    self.assertEqual(comp5(), 75.0)


class FederatedComputationsTest(test.TestCase):

  def test_no_argument_fed_comp(self):

    @tff.federated_computation
    def foo():
      return 10

    self.assertEqual(str(foo.type_signature), '( -> int32)')
    self.assertEqual(foo(), 10)

  def test_fed_comp_typical_usage_as_decorator_with_unlabeled_type(self):

    @tff.federated_computation((tff.FunctionType(tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      assert isinstance(f, tff.Value)
      assert isinstance(x, tff.Value)
      assert str(f.type_signature) == '(int32 -> int32)'
      assert str(x.type_signature) == 'int32'
      result_value = f(f(x))
      assert isinstance(result_value, tff.Value)
      assert str(result_value.type_signature) == 'int32'
      return result_value

    self.assertEqual(
        str(foo.type_signature), '(<(int32 -> int32),int32> -> int32)')

    @tff.tf_computation(tf.int32)
    def third_power(x):
      return x**3

    self.assertEqual(foo(third_power, 10), int(1e9))
    self.assertEqual(foo(third_power, 1), 1)

  def test_fed_comp_typical_usage_as_decorator_with_labeled_type(self):

    @tff.federated_computation((
        ('f', tff.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32),
    ))
    def foo(f, x):
      return f(f(x))

    @tff.tf_computation(tf.int32)
    def square(x):
      return x**2

    @tff.tf_computation(tf.int32, tf.int32)
    def square_drop_y(x, y):  # pylint: disable=unused-argument
      return x * x

    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')

    self.assertEqual(foo(square, 10), int(1e4))
    self.assertEqual(square_drop_y(square_drop_y(10, 5), 100), int(1e4))
    self.assertEqual(square_drop_y(square_drop_y(10, 100), 5), int(1e4))
    with self.assertRaisesRegexp(TypeError,
                                 'is not assignable from source type'):
      self.assertEqual(foo(square_drop_y, 10), 100)


if __name__ == '__main__':
  test.main()
