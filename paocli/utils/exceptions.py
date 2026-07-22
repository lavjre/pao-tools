#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    : exceptions.py
@Time    : 2021/11/23 23:47:51
@Author  : Roderick Chan
@Email   : roderickchan@foxmail.com
@Desc    : Exception
"""

__all__ = ["PaocliExit"]


class PaocliExit(SystemExit):
    """
    PaocliExit
    """

    pass


class PaocliTodoException(Exception):
    """
    TODO Exception
    """

    pass
