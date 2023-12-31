# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transmission/tenseal/tenseal_shapley_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transmission/tenseal/tenseal_shapley_data.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n/transmission/tenseal/tenseal_shapley_data.proto\"A\n\x12\x63lient_shapley_msg\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x0b\n\x03msg\x18\x03 \x01(\x0c\"-\n\x05top_k\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\x0f\n\x07ranking\x18\x02 \x03(\x05\x32<\n\x0eShapleyService\x12*\n\x0bsum_shapley\x12\x13.client_shapley_msg\x1a\x06.top_kb\x06proto3'
)




_CLIENT_SHAPLEY_MSG = _descriptor.Descriptor(
  name='client_shapley_msg',
  full_name='client_shapley_msg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_rank', full_name='client_shapley_msg.client_rank', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='k', full_name='client_shapley_msg.k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='msg', full_name='client_shapley_msg.msg', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=116,
)


_TOP_K = _descriptor.Descriptor(
  name='top_k',
  full_name='top_k',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_rank', full_name='top_k.client_rank', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ranking', full_name='top_k.ranking', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=118,
  serialized_end=163,
)

DESCRIPTOR.message_types_by_name['client_shapley_msg'] = _CLIENT_SHAPLEY_MSG
DESCRIPTOR.message_types_by_name['top_k'] = _TOP_K
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

client_shapley_msg = _reflection.GeneratedProtocolMessageType('client_shapley_msg', (_message.Message,), {
  'DESCRIPTOR' : _CLIENT_SHAPLEY_MSG,
  '__module__' : 'transmission.tenseal.tenseal_shapley_data_pb2'
  # @@protoc_insertion_point(class_scope:client_shapley_msg)
  })
_sym_db.RegisterMessage(client_shapley_msg)

top_k = _reflection.GeneratedProtocolMessageType('top_k', (_message.Message,), {
  'DESCRIPTOR' : _TOP_K,
  '__module__' : 'transmission.tenseal.tenseal_shapley_data_pb2'
  # @@protoc_insertion_point(class_scope:top_k)
  })
_sym_db.RegisterMessage(top_k)



_SHAPLEYSERVICE = _descriptor.ServiceDescriptor(
  name='ShapleyService',
  full_name='ShapleyService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=165,
  serialized_end=225,
  methods=[
  _descriptor.MethodDescriptor(
    name='sum_shapley',
    full_name='ShapleyService.sum_shapley',
    index=0,
    containing_service=None,
    input_type=_CLIENT_SHAPLEY_MSG,
    output_type=_TOP_K,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SHAPLEYSERVICE)

DESCRIPTOR.services_by_name['ShapleyService'] = _SHAPLEYSERVICE

# @@protoc_insertion_point(module_scope)
