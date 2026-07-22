# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit instance-local bindings from operations to allocation scopes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import MethodType
from typing import Protocol

from .errors import GMSError


class AllocationScope(Protocol):
    def __call__(self) -> AbstractContextManager[None]:
        pass


def install_scoped_call(
    instance: object,
    method_name: str,
    supported_hook: Callable[..., object],
    scope: AllocationScope,
    predicate: Callable[..., bool],
    *,
    owner: object,
) -> None:
    """Enter ``scope`` only for calls selected by ``predicate``.

    One instance method may have one explicit owner. Reinstalling the same
    owner and scope is a no-op; ambiguous or replaced ownership fails.
    """
    attribute = f"_dynamo_gms_v1_scope_{method_name}"
    existing = vars(instance).get(attribute)
    if existing is not None:
        existing_owner, existing_scope, installed = existing
        if existing_owner is not owner or existing_scope != scope:
            raise GMSError(f"{method_name} already has another allocation scope owner")
        if getattr(instance, method_name) is not installed:
            raise GMSError(f"{method_name} allocation scope binding was replaced")
        return

    original = getattr(instance, method_name)
    if (
        getattr(original, "__self__", None) is not instance
        or getattr(original, "__func__", None) is not supported_hook
    ):
        raise GMSError(f"{method_name} is not the supported allocation hook")

    def scoped(self, *args, **kwargs):
        if not predicate(self, *args, **kwargs):
            return original(*args, **kwargs)
        with scope():
            return original(*args, **kwargs)

    installed = MethodType(scoped, instance)
    setattr(instance, method_name, installed)
    setattr(instance, attribute, (owner, scope, installed))
