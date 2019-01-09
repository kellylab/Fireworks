from abc import ABCMeta as NativeABCMeta

"""
This is taken from StackOverflow user krassowski
https://stackoverflow.com/questions/23831510/abstract-attribute-not-property/50381071#50381071
This is an implementation of the abstract base class module including an abstractattribute property.

"""

class DummyAttribute:
    pass

def abstractattribute(obj=None):
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstractattribute__ = True
    return obj


class ABCMeta(NativeABCMeta):

    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        abstractattributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstractattribute__', False)
        }
        if abstractattributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(abstractattributes)
                )
            )
        return instance
