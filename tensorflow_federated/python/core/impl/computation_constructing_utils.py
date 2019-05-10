# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Library implementing reusable `computation_building_blocks` constructs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils


def construct_federated_getitem_call(arg, idx):
  """Constructs computation building block passing getitem to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.NamedTupleType` underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`, the result
    of applying or mapping the appropriate `__getitem__` function, as defined
    by `idx`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getitem_comp = construct_federated_getitem_comp(arg, idx)
  return construct_map_or_apply(getitem_comp, arg)


def construct_federated_getattr_call(arg, name):
  """Constructs computation building block passing getattr to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `name`.
    name: String name to address the `computation_types.NamedTupleType`
      underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getattr_comp = construct_federated_getattr_comp(arg, name)
  return construct_map_or_apply(getattr_comp, arg)


def construct_federated_setattr_call(federated_comp, name, value_comp):
  """Returns building block for `setattr(name, value_comp)` on `federated_comp`.

  Constructs an appropriate communication intrinsic (either `federated_map` or
  `federated_apply`) as well as a `computation_building_blocks.Lambda`
  representing setting the `name` attribute of `federated_comp`'s `member` to
  `value_comp`, and stitches these together in a call.

  Notice that `federated_comp`'s `member` must actually define a `name`
  attribute; this is enforced to avoid the need to worry about theplacement of a
  previously undefined name.

  Args:
    federated_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock` of type
      `computation_types.FederatedType`, with member of type
      `computation_types.NamedTupleType` whose attribute `name` we wish to set
      to `value_comp`.
    name: String name of the attribute we wish to overwrite in `federated_comp`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to
      assign to `federated_comp`'s `member`'s `name` attribute.

  Returns:
    Instance of `computation_building_blocks.ComputationBuildingBlock`
    representing `federated_comp` with its `member`'s `name` attribute set to
    `value`.
  """
  py_typecheck.check_type(federated_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(federated_comp.type_signature,
                          computation_types.FederatedType)
  py_typecheck.check_type(federated_comp.type_signature.member,
                          computation_types.NamedTupleType)
  named_tuple_type_signature = federated_comp.type_signature.member
  setattr_lambda = construct_named_tuple_setattr_lambda(
      named_tuple_type_signature, name, value_comp)
  return construct_map_or_apply(setattr_lambda, federated_comp)


def construct_named_tuple_setattr_lambda(named_tuple_signature, name,
                                         value_comp):
  """Constructs a building block for replacing one attribute in a named tuple.

  Returns an instance of `computation_building_blocks.Lambda` which takes an
  argument of type `computation_types.NamedTupleType` and returns a
  `computation_building_blocks.Tuple` which contains all the same elements as
  the argument, except the attribute `name` now has value `value_comp`. The
  Lambda constructed is the analogue of Python's `setattr` for the concrete
  type `named_tuple_signature`.

  Args:
    named_tuple_signature: Instance of `computation_types.NamedTupleType`, the
      type of the argument to the constructed
      `computation_building_blocks.Lambda`.
    name: String name of the attribute in the `named_tuple_signature` to replace
      with `value_comp`. Must be present as a name in `named_tuple_signature;
      otherwise we will raise an `AttributeError`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to place
      as attribute `name` in the argument of the returned function.

  Returns:
    An instance of `computation_building_blocks.Block` of functional type
    representing setting attribute `name` to value `value_comp` in its argument
    of type `named_tuple_signature`.

  Raises:
    TypeError: If the types of the arguments don't match the assumptions above.
    AttributeError: If `name` is not present as a named element in
      `named_tuple_signature`
  """
  py_typecheck.check_type(named_tuple_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  value_comp_placeholder = computation_building_blocks.Reference(
      'value_comp_placeholder', value_comp.type_signature)
  lambda_arg = computation_building_blocks.Reference('lambda_arg',
                                                     named_tuple_signature)
  if name not in dir(named_tuple_signature):
    raise AttributeError(
        'There is no such attribute as \'{}\' in this federated tuple. '
        'TFF does not allow for assigning to a nonexistent attribute. '
        'If you want to assign to \'{}\', you must create a new named tuple '
        'containing this attribute.'.format(name, name))
  elements = []
  for idx, (key, element_type) in enumerate(
      anonymous_tuple.to_elements(named_tuple_signature)):
    if key == name:
      if not type_utils.is_assignable_from(element_type,
                                           value_comp.type_signature):
        raise TypeError(
            '`setattr` has attempted to set element {} of type {} with incompatible type {}'
            .format(key, element_type, value_comp.type_signature))
      elements.append((key, value_comp_placeholder))
    else:
      elements.append(
          (key, computation_building_blocks.Selection(lambda_arg, index=idx)))
  return_tuple = computation_building_blocks.Tuple(elements)
  lambda_to_return = computation_building_blocks.Lambda(lambda_arg.name,
                                                        named_tuple_signature,
                                                        return_tuple)
  enclosing_block = computation_building_blocks.Block(
      [(value_comp_placeholder.name, value_comp)], lambda_to_return)
  return enclosing_block


def construct_map_or_apply(fn, arg):
  """Injects intrinsic to allow application of `fn` to federated `arg`.

  Args:
    fn: An instance of `computation_building_blocks.ComputationBuildingBlock` of
      functional type to be wrapped with intrinsic in order to call on `arg`.
    arg: `computation_building_blocks.ComputationBuildingBlock` instance of
      federated type for which to construct intrinsic in order to call `fn` on
      `arg`. `member` of `type_signature` of `arg` must be assignable to
      `parameter` of `type_signature` of `fn`.

  Returns:
    Returns a `computation_building_blocks.Intrinsic` which can call
    `fn` on `arg`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  type_utils.check_assignable_from(fn.type_signature.parameter,
                                   arg.type_signature.member)
  if arg.type_signature.placement == placement_literals.SERVER:
    result_type = computation_types.FederatedType(fn.type_signature.result,
                                                  arg.type_signature.placement,
                                                  arg.type_signature.all_equal)
    intrinsic_type = computation_types.FunctionType(
        [fn.type_signature, arg.type_signature], result_type)
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type)
    tup = computation_building_blocks.Tuple((fn, arg))
    return computation_building_blocks.Call(intrinsic, tup)
  elif arg.type_signature.placement == placement_literals.CLIENTS:
    return create_federated_map(fn, arg)


def construct_federated_getattr_comp(comp, name):
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  element_names = [
      x for x, _ in anonymous_tuple.to_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError('The federated value {} has no element of name {}'.format(
        comp, name))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  selected = computation_building_blocks.Selection(apply_input, name=name)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def construct_federated_getitem_comp(comp, key):
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.NamedTupleType`.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(key, (int, slice))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  if isinstance(key, int):
    selected = computation_building_blocks.Selection(apply_input, index=key)
  else:
    elems = anonymous_tuple.to_elements(comp.type_signature.member)
    index_range = six.moves.range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0],
           computation_building_blocks.Selection(apply_input, index=k)))
    selected = computation_building_blocks.Tuple(elem_list)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def create_computation_appending(comp1, comp2):
  r"""Returns a block appending `comp2` to `comp1`.

                       Block
                      /     \
  [result=Comp(comp1)]       Tuple
                             |
                             [Sel(0), ...,   Sel(n), Comp(comp2)]
                                    \              \
                                     Ref(result)    Ref(result)

  Args:
    comp1: A `computation_building_blocks.ComputationBuildingBlock` the with a
      `type_signature` of type `computation_type.NamedTupleType`.
    comp2: A `computation_building_blocks.ComputationBuildingBlock` or a named
      computation representable as an element of an AnonymousTuple.

  Returns:
    A `computation_building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(comp1,
                          computation_building_blocks.ComputationBuildingBlock)
  type_signatures = anonymous_tuple.to_elements(comp1.type_signature)
  ref_type = [e for _, e in type_signatures]
  ref = computation_building_blocks.Reference('result', ref_type)
  symbols = ((ref.name, comp1),)
  elements = []
  for index, (name, _) in enumerate(type_signatures):
    sel = computation_building_blocks.Selection(ref, index=index)
    elements.append((name, sel))
  elements.append(comp2)
  tup = computation_building_blocks.Tuple(elements)
  return computation_building_blocks.Block(symbols, tup)


def create_federated_apply(fn, arg):
  r"""Creates a called federated apply.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.SERVER, True)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def create_federated_map(fn, arg):
  r"""Creates a called federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A functional `computation_building_blocks.ComputationBuildingBlock` to
      use as the function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  type_utils.check_federated_type(arg.type_signature)
  parameter_type = computation_types.FederatedType(fn.type_signature.parameter,
                                                   placement_literals.CLIENTS,
                                                   False)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.CLIENTS,
                                                False)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri, intrinsic_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def create_federated_zip(value):
  r"""Creates a called federated zip.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` is an empty `computation_building_blocks.Tuple`.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  elements = anonymous_tuple.to_elements(value.type_signature)
  length = len(elements)
  if length == 0:
    raise ValueError('federated_zip is only supported on nonempty tuples.')

  _, first_type_signature = elements[0]
  if first_type_signature.placement == placement_literals.CLIENTS:
    map_fn = create_federated_map
  elif first_type_signature.placement == placement_literals.SERVER:
    map_fn = create_federated_apply

  if length == 1:
    name, type_signature = elements[0]
    ref = computation_building_blocks.Reference('arg', type_signature.member)
    tup = computation_building_blocks.Tuple(((name, ref),))
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, tup)
    sel = computation_building_blocks.Selection(value, index=0)
    return map_fn(fn, sel)
  else:
    type_signatures = anonymous_tuple.to_elements(value.type_signature)

    def _zip_args(arg1, arg2):
      tup = computation_building_blocks.Tuple((arg1, arg2))
      return _create_federated_zip_two_tuple(tup)

    sel_0 = computation_building_blocks.Selection(value, index=0)
    sel_1 = computation_building_blocks.Selection(value, index=1)
    arg = _zip_args(sel_0, sel_1)

    def _append_zipped_args(fn, type_spec):
      """Internal function to append zipped computations."""
      ref_type = computation_types.NamedTupleType((
          fn.type_signature.parameter,
          (type_spec[0], type_spec[1].member),
      ))
      ref = computation_building_blocks.Reference('arg', ref_type)
      sel_0 = computation_building_blocks.Selection(ref, index=0)
      call = computation_building_blocks.Call(fn, sel_0)
      sel_1 = computation_building_blocks.Selection(ref, index=1)
      result = create_computation_appending(call, (type_spec[0], sel_1))
      return computation_building_blocks.Lambda(ref.name, ref.type_signature,
                                                result)

    ref_type = computation_types.NamedTupleType((
        (type_signatures[0][0], type_signatures[0][1].member),
        (type_signatures[1][0], type_signatures[1][1].member),
    ))
    ref = computation_building_blocks.Reference('arg', ref_type)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    for k in range(2, len(type_signatures)):
      sel = computation_building_blocks.Selection(value, index=k)
      arg = _zip_args(arg, sel)
      fn = _append_zipped_args(fn, type_signatures[k])
    return map_fn(fn, arg)


def _create_federated_zip_two_tuple(tup):
  r"""Creates a called federated zip of a 2-tuple.

              Call
             /    \
    Intrinsic      Comp(tup)

  Args:
    tup: A `computation_building_blocks.Tuple` to use as the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `tup` is not a 2-tuple.
  """
  py_typecheck.check_type(tup, computation_building_blocks.Tuple)
  if len(tup) != 2:
    raise ValueError('Expected a 2-tuple, got an {}-tuple.'.format(len(tup)))
  placement = tup[0].type_signature.placement
  if placement is placement_literals.CLIENTS:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri
    all_equal = False
  elif placement is placement_literals.SERVER:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(str(placement)))
  elements = []
  type_signatures = anonymous_tuple.to_elements(tup.type_signature)
  for name, type_spec in type_signatures:
    federated_type = computation_types.FederatedType(type_spec.member,
                                                     placement, all_equal)
    elements.append((name, federated_type))
  parameter_type = computation_types.NamedTupleType(elements)
  result_type = computation_types.FederatedType(
      [(n, e.member) for n, e in type_signatures], placement, all_equal)
  intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
  intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, tup)
