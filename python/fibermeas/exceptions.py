# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class FibermeasError(Exception):
    """A custom core Fibermeas exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(FibermeasError, self).__init__(message)


class FibermeasNotImplemented(FibermeasError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(FibermeasNotImplemented, self).__init__(message)


class FibermeasAPIError(FibermeasError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Fibermeas API'
        else:
            message = 'Http response error from Fibermeas API. {0}'.format(message)

        super(FibermeasAPIError, self).__init__(message)


class FibermeasApiAuthError(FibermeasAPIError):
    """A custom exception for API authentication errors"""
    pass


class FibermeasMissingDependency(FibermeasError):
    """A custom exception for missing dependencies."""
    pass


class FibermeasWarning(Warning):
    """Base warning for Fibermeas."""


class FibermeasUserWarning(UserWarning, FibermeasWarning):
    """The primary warning class."""
    pass


class FibermeasSkippedTestWarning(FibermeasUserWarning):
    """A warning for when a test is skipped."""
    pass


class FibermeasDeprecationWarning(FibermeasUserWarning):
    """A warning for deprecated features."""
    pass
