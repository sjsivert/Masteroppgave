from keras import optimizers


class KerasOptimizer:
    @staticmethod
    def get(optimizer_name: str, learning_rate: float):
        """Dispatch method"""
        method_name = "_opt_" + str(optimizer_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(KerasOptimizer, method_name, lambda: "Invalid optimizier")
        # Call the method as we return it
        return method(learning_rate)

    @staticmethod
    def _opt_Adam(learning_rate: float):
        return optimizers.adam_v2.Adam(lr=learning_rate)

    @staticmethod
    def _opt_SGD(learning_rate: float):
        return optimizers.adadelta_v2.Adadelta(lr=learning_rate)

    @staticmethod
    def _opt_SGD(learning_rate: float):
        return optimizers.gradient_descent_v2.SGD(lr=learning_rate)

    @staticmethod
    def _opt_RMSprop(learning_rate: float):
        return optimizers.rmsprop_v2.RMSprop(lr=learning_rate)

    @staticmethod
    def opt_Adamax(learning_rate: float):
        return optimizers.adamax_v2.Adamax(lr=learning_rate)

    @staticmethod
    def _opt_Adadelta(learning_rate: float):
        return optimizers.adadelta_v2.Adadelta(lr=learning_rate)

    # and so on for how many cases you would need
