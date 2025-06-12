from beecolpy import abc

# Define the objective function
def Sphere(x):
    return sum(i**2 for i in x)

if __name__ == "__main__":
    # Instantiate the ABC optimizer
    optimizer = abc(
        function=Sphere,
        boundaries=[(-5.12, 5.12)] * 30,  # 30 dimensions
        colony_size=20,
        iterations=100
    )

    # Run the optimization
    result = optimizer.fit()

    # Print results
    print("Best solution found:", result)
    print("Best value:", Sphere(result))
    print("Status:", optimizer.get_status())
