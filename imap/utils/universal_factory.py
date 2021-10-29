from inspect import signature

from pytorch_lightning.utilities.parsing import AttributeDict

PARAMETER_ATTRIBUTE_NAMES = ["parameters", "params", "param", "parameter"]
CLASS_NAME_ATTRIBUTES = ["name", "type"]


class UniversalFactory(object):
    def __init__(self, classes):
        self._classes = {x.__name__: x for x in classes}

    def make_from_parameters(self, parameters, **kwargs):
        init_function = None
        if not isinstance(parameters, AttributeDict) and not isinstance(parameters, dict):
            return parameters
        for class_name_attribute in CLASS_NAME_ATTRIBUTES:
            if class_name_attribute in parameters.keys():
                class_name = parameters[class_name_attribute]
                try:
                    init_function = self._classes[class_name]
                except KeyError:
                    raise KeyError(f"Unknown class {class_name}")
        if init_function is None:
            return parameters
        for key, value in parameters.items():
            kwargs[key] = self.make_from_parameters(value)
        init_function_attributes = signature(init_function).parameters.keys()
        for parameter_name in PARAMETER_ATTRIBUTE_NAMES:
            if parameter_name in init_function_attributes:
                kwargs[parameter_name] = parameters
        return self.kwargs_function(init_function)(**kwargs)

    @staticmethod
    def kwargs_function(function):
        def new_function(*args, **kwargs):
            parameter_names = signature(function).parameters.keys()
            right_kwargs = {}
            for parameter_name in parameter_names:
                if parameter_name in kwargs.keys():
                    right_kwargs[parameter_name] = kwargs[parameter_name]
            return function(*args, **right_kwargs)

        return new_function
