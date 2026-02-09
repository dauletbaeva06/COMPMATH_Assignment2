import numpy as np

def task_2_taylor():
    x_target = 0.2
    
    y0 = 0
    y1 = 3 
    y2 = 9
    y3 = 21
    y4 = 45
    
    # Taylor series formula
    y_approx = (y0 + 
                (x_target * y1) + 
                (x_target**2 / 2 * y2) + 
                (x_target**3 / 6 * y3) + 
                (x_target**4 / 24 * y4))
    
    y_exact = 3 * np.exp(2 * x_target) - 3 * np.exp(x_target)
    
    print(f"Taylor Approximation at x=0.2: {y_approx:.5f}")
    print(f"Exact Solution at x=0.2:      {y_exact:.5f}")
    print(f"Absolute Error:               {abs(y_exact - y_approx):.6f}")

if __name__ == "__main__":
    task_2_taylor()