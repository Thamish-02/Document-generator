## AI Summary

A file named setuptools_frontend.py.


### Function: check_message_extractors(dist, name, value)

**Description:** Validate the ``message_extractors`` keyword argument to ``setup()``.

:param dist: the distutils/setuptools ``Distribution`` object
:param name: the name of the keyword argument (should always be
             "message_extractors")
:param value: the value of the keyword argument
:raise `DistutilsSetupError`: if the value is not valid

## Class: compile_catalog

**Description:** Catalog compilation command for use in ``setup.py`` scripts.

If correctly installed, this command is available to Setuptools-using
setup scripts automatically. For projects using plain old ``distutils``,
the command needs to be registered explicitly in ``setup.py``::

    from babel.messages.setuptools_frontend import compile_catalog

    setup(
        ...
        cmdclass = {'compile_catalog': compile_catalog}
    )

.. versionadded:: 0.9

## Class: extract_messages

**Description:** Message extraction command for use in ``setup.py`` scripts.

If correctly installed, this command is available to Setuptools-using
setup scripts automatically. For projects using plain old ``distutils``,
the command needs to be registered explicitly in ``setup.py``::

    from babel.messages.setuptools_frontend import extract_messages

    setup(
        ...
        cmdclass = {'extract_messages': extract_messages}
    )

## Class: init_catalog

**Description:** New catalog initialization command for use in ``setup.py`` scripts.

If correctly installed, this command is available to Setuptools-using
setup scripts automatically. For projects using plain old ``distutils``,
the command needs to be registered explicitly in ``setup.py``::

    from babel.messages.setuptools_frontend import init_catalog

    setup(
        ...
        cmdclass = {'init_catalog': init_catalog}
    )

## Class: update_catalog

**Description:** Catalog merging command for use in ``setup.py`` scripts.

If correctly installed, this command is available to Setuptools-using
setup scripts automatically. For projects using plain old ``distutils``,
the command needs to be registered explicitly in ``setup.py``::

    from babel.messages.setuptools_frontend import update_catalog

    setup(
        ...
        cmdclass = {'update_catalog': update_catalog}
    )

.. versionadded:: 0.9
