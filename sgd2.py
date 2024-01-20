class StochasticGradientDescent(object):
    def __init__(
        self,
        parameters,
        alpha=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=True,
        maximise=False,
    ):
        self.parameters = parameters
        self.alpha = alpha
        self.betas = betas
        self.eps = eps

        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximise = maximise

    def step(self):
        for parameter in self.parameters:
            gradient = gradient + self.weight_decay * parameter
            gradient = self.gradient_clipping(gradient)

            mean = self.betas[0] * self.mean[param_idx]
            mean += (1 - self.betas[0]) * gradient
            self.mean[param_idx] = mean

            variance = self.betas[1] * self.variance[param_idx]
            variance = (1 - self.betas[1]) * gradient**2
            self.variance[param_idx] = variance

            corrected_mean = mean / (1 - self.betas[0] ** 2)
            corrected_variance = variance / (1 - self.betas[1] ** 2)

            factor = corrected_mean / \
                (np.sqrt(corrected_variance) + self.epsilon)
            parameter = parameter - self.learning_rate * factor

    def zero_grad(self):
        for parameter in parameters:
            parameter.zero_grad()
