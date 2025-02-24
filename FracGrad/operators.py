import torch

class fractional():
    def __init__(self, alpha=0.9) -> None:
        # Validate alpha is numeric and greater than zero
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a numeric value (int or float)")
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero")
        self.alpha = alpha

    def gradient(self, p, pm_1, second_order):
        try:
            # Validate input types
            if not isinstance(p, torch.Tensor) or not isinstance(pm_1, torch.Tensor):
                raise TypeError("Both 'p' and 'pm_1' must be torch.Tensor instances.")
            
            # Check if the gradient for p is available
            if p.grad is None:
                raise ValueError("p.grad is None. Ensure that gradients are computed before calling fractional.gradient.")

            # Compute the constant factor using the gamma function
            factor = 1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha)))
            diff = torch.abs(p.data.detach() - pm_1.data.detach())
            
            result = factor * p.grad.detach() * (diff ** (1 - self.alpha))
            return result

        except Exception as e:
            # Log the error and re-raise to inform the caller
            print(f"Error in fractional.gradient: {e}")
            raise
    
    def __call__(self, p, pm_1, second_order):
        return self.gradient(p, pm_1, second_order)