"""Typed ROP surface (facade; implementation lives in rop_* modules)."""

from paomcp.tools.backends.rop_chain import *  # noqa: F401,F403
from paomcp.tools.backends.rop_gadgets import *  # noqa: F401,F403
from paomcp.tools.backends.rop_libc import *  # noqa: F401,F403
from paomcp.tools.backends.rop_schemas import *  # noqa: F401,F403

# Original contract (kept verbatim):
__all__ = [
    "GadgetIndex",
    "GadgetIndexError",
    "GadgetLoader",
    "LibcIdentifier",
    "LibcIdentifierError",
    "ChainBuilder",
    "ChainBuilderError",
    "ChainValidator",
    "ChainValidatorError",
    "SymbolExtractor",
    "OneGadgetRunner",
    "RopTools",
    "GADGET_LIST_SCHEMA",
    "GADGET_SEQUENCE_SCHEMA",
    "GADGET_INSPECTION_SCHEMA",
    "LIBC_IDENTIFICATION_SCHEMA",
    "ONE_GADGET_SCHEMA",
    "ROP_CHAIN_SCHEMA",
    "SIGRETURN_FRAME_SCHEMA",
    "CHAIN_CONSTRAINTS_SCHEMA",
    "CHAIN_VALIDATION_SCHEMA",
    "CHAIN_EXPORT_SCHEMA",
    "CHAIN_EXPORT_FORMATS",
    "DEFAULT_MAX_GADGETS",
    "MAX_MAX_GADGETS",
    "DEFAULT_MAX_INSTRUCTIONS",
    "MAX_SEQUENCE_STEPS",
    "DEFAULT_ONE_GADGET_TIMEOUT_SEC",
    "ropper_gadget_loader",
]
