# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tenseal_shapley_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1atenseal_shapley_data.proto\"A\n\x12\x63lient_shapley_msg\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x0b\n\x03msg\x18\x03 \x01(\x0c\"-\n\x05top_k\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\x0f\n\x07ranking\x18\x02 \x03(\x05\x32<\n\x0eShapleyService\x12*\n\x0bsum_shapley\x12\x13.client_shapley_msg\x1a\x06.top_kb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tenseal_shapley_data_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CLIENT_SHAPLEY_MSG']._serialized_start=30
  _globals['_CLIENT_SHAPLEY_MSG']._serialized_end=95
  _globals['_TOP_K']._serialized_start=97
  _globals['_TOP_K']._serialized_end=142
  _globals['_SHAPLEYSERVICE']._serialized_start=144
  _globals['_SHAPLEYSERVICE']._serialized_end=204
# @@protoc_insertion_point(module_scope)