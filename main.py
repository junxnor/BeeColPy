
#####################################################################################################
# This code demonstrates how to use the beecolpy library to perform optimization using the Artificial Bee Colony (ABC) algorithm.

# import matplotlib.pyplot as plt

# # from beecolpy import abc

# # # Define the objective function
# # def Sphere(x):
# #     return sum(i**2 for i in x)

# # if __name__ == "__main__":
# #     # Instantiate the ABC optimizer
# #     optimizer = abc(
# #         function=Sphere,
# #         boundaries=[(-5.12, 5.12)] * 30,  # 30 dimensions
# #         colony_size=20,
# #         iterations=100
# #     )

# #     # Run the optimization
# #     result = optimizer.fit()

# #     # Print results
# #     print("Best solution found:", result)
# #     print("Best value:", Sphere(result))
# #     print("Status:", optimizer.get_status())


#####################################################################################################
#showing the graph of the best solution found by the algorithm

# """
# To find the minimum  of sphere function on interval (-10 to 10) with
# 2 dimensions in domain using default parameters:
# """

# from beecolpy import abc

# def sphere(x):
# 	total = 0
# 	for i in range(len(x)):
# 		total += x[i]**2
# 	return total
	
# abc_obj = abc(sphere, [(-10,10), (-10,10)]) #Load data
# abc_obj.fit() #Execute the algorithm

# #If you want to get the obtained solution after execute the fit() method:
# solution = abc_obj.get_solution()

# #If you want to get the number of iterations executed, number of times that
# #scout event occur and number of times that NaN protection actuated:
# iterations = abc_obj.get_status()[0]
# scout = abc_obj.get_status()[1]
# nan_events = abc_obj.get_status()[2]

# #If you want to get a list with position of all points (food sources) used in each iteration:
# food_sources = abc_obj.get_agents()

# print("Best solution found:", solution)
# print("Best value:", sphere(solution))
# print("Iterations:", iterations)
# print("Scout events:", scout)
# print("NaN events:", nan_events)

# # Print food sources from each iteration (optional: limit output)
# print("\nFood sources per iteration (showing first 3 iterations):")
# for i, iteration in enumerate(food_sources[:3]):
#     print(f"Iteration {i + 1}: {iteration}")
# # Note: The above code assumes that the beecolpy library is installed and properly configured.
# # The code above demonstrates how to use the `beecolpy` library to perform optimization using the Artificial Bee Colony (ABC) algorithm.


# # Plot the evolution of food sources
# plt.figure(figsize=(8, 6))

# for i, iteration in enumerate(food_sources[:30]):  # limit to first 30 iterations
#     xs = [point[0] for point in iteration]
#     ys = [point[1] for point in iteration]
#     plt.scatter(xs, ys, alpha=0.4, label=f"Iter {i+1}" if i % 5 == 0 else "", s=20)

# # Mark the final best solution
# plt.scatter(solution[0], solution[1], color='red', marker='*', s=200, label="Best Solution")

# plt.title("Food Source Movement Across Iterations (ABC Optimization)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




#####################################################################################################



import numpy as np  # Add this at the top if not already


# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from beecolpy import abc

# Objective function
def sphere(x):
    return sum(i**2 for i in x)

# def risky_function(x):
#     return np.log(x[0]) + x[1]

def risky_function(x):
    if x[0] <= 0:
        return float('nan')
    return np.log(x[0]) + x[1]
# Initialize ABC optimization
# abc_obj = abc(sphere, [(-10, 10), (-10, 10)], log_agents=True)
# abc_obj = abc(
#     sphere,
#     [(-10, 10), (-10, 10)],
#     colony_size=80,
#     scouts=0.61,
#     iterations=200,
#     seed=50,
#     nan_protection=True,
#     log_agents=True
# )
abc_obj = abc(
    risky_function,         # Objective function choose from [sphere, risky_function]
    [(-10, 10), (-10, 10)], # Search space for the objective function
    colony_size=50,         # Colony size (number of agents)
    scouts=0.5,             # Scouts (the loss of agents)
    iterations=50,          # Number of iterations
    seed=42,                # Seed for reproducibility
    nan_protection=True,    # Enable NaN protection Avoid Crash with a traceback
    log_agents=True,        # Log agents' positions across iterations
    min_max='min'           # if max setting it just maximize the error function
)

abc_obj.fit()

# Get results
solution = abc_obj.get_solution()
iterations, scout, nan_events = abc_obj.get_status()
food_sources = abc_obj.get_agents()

print("Best solution found:", solution)
print("Best value:", sphere(solution))
print("Iterations:", iterations)
print("Scout events:", scout)
print("NaN events:", nan_events)


# --------- Animation Section ---------
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([], [], s=50)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Animated Food Source Movement (ABC Optimization)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Draw the best solution once
best_sol_plot = ax.scatter(solution[0], solution[1], color='red', marker='*', s=200, label="Best Solution")
ax.legend()

# Init function (FIXED)
def init():
    sc.set_offsets(np.empty((0, 2)))
    return sc,

# Update function (already okay)
def update(frame):
    current_sources = food_sources[frame]
    sc.set_offsets(current_sources)
    ax.set_title(f"Iteration {frame + 1} / {len(food_sources)}")
    return sc,

ani = animation.FuncAnimation(
    fig, update, frames=len(food_sources),
    init_func=init, blit=True, repeat=False
)

best_values = [sphere(min(agent, key=sphere)) for agent in food_sources]
plt.plot(best_values)
plt.title("Best Value Over Iterations")
plt.xlabel("Iteration")

plt.tight_layout()
plt.show()

# Optionally: Save as video or GIF
# ani.save("abc_animation.mp4", fps=2)  # Requires ffmpeg
# ani.save("abc_animation.gif", writer='imagemagick')  # Requires ImageMagick





#####################################################################################################
# This code performs a hyperparameter search for the Artificial Bee Colony (ABC) algorithm
# using random combinations of parameters. It evaluates the performance of each configuration
# dont use it unless you want to test the hyperparameters of the algorithm
# and you planned to change the parameter settings for shorter time


# import numpy as np
# import random
# from beecolpy import abc

# def sphere(x):
#     return sum(i**2 for i in x)

# # Define search space
# search_space = {
#     "colony_size": list(range(10, 10001)),                  # 10–10,000
#     "scouts": list(np.linspace(0.1, 1.0, 100)),             # 100 values
#     "iterations": list(range(10, 1001)),                    # 10–1000
#     "seed": list(range(1, 101))                             # 1–100
# }

# best_score = float("inf")
# best_config = None

# # ✅ Try N random configurations
# for i in range(10):  # Change 10 to any number of trials
#     config = {
#         "colony_size": random.choice(search_space["colony_size"]),
#         "scouts": random.choice(search_space["scouts"]),
#         "iterations": random.choice(search_space["iterations"]),
#         "seed": random.choice(search_space["seed"])
#     }

#     print(f"\nRunning Trial {i+1} with config: {config}", flush=True)

#     abc_obj = abc(
#         sphere,
#         [(-10, 10), (-10, 10)],
#         colony_size=config["colony_size"],
#         scouts=config["scouts"],
#         iterations=config["iterations"],
#         seed=config["seed"],
#         nan_protection=True,
#         log_agents=False
#     )

#     abc_obj.fit()
#     solution = abc_obj.get_solution()
#     score = sphere(solution)

#     print(f"Trial {i+1} Score: {score:.6f}")

#     if score < best_score:
#         best_score = score
#         best_config = config

# print("\n✅ Best Config Found:")
# print(best_config)
# print(f"Best Score: {best_score:.6f}")

# # Note: This code performs a simple hyperparameter search for the ABC algorithm
# # using random combinations of parameters. It prints the score for each trial

############################################################################################################################################
# the hyperparameter output is like this: 

# PS C:\APU\AIM\BeeColPy> & C:/Users/USER/AppData/Local/Programs/Python/Python313/python.exe c:/APU/AIM/BeeColPy/main.py

# Running Trial 1 with config: {'colony_size': 1217, 'scouts': np.float64(0.5181818181818182), 'iterations': 280, 'seed': 44}
# Trial 1 Score: 0.000000

# Running Trial 2 with config: {'colony_size': 119, 'scouts': np.float64(0.9363636363636363), 'iterations': 43, 'seed': 53}
# Trial 2 Score: 0.000000

# Running Trial 3 with config: {'colony_size': 1867, 'scouts': np.float64(0.7727272727272727), 'iterations': 283, 'seed': 100}
# Trial 3 Score: 0.000000

# Running Trial 4 with config: {'colony_size': 8803, 'scouts': np.float64(0.28181818181818186), 'iterations': 404, 'seed': 95}
# Trial 4 Score: 0.000000

# Running Trial 5 with config: {'colony_size': 4338, 'scouts': np.float64(0.32727272727272727), 'iterations': 611, 'seed': 63}
# Trial 5 Score: 0.000000

# Running Trial 6 with config: {'colony_size': 2296, 'scouts': np.float64(0.6727272727272727), 'iterations': 551, 'seed': 85}
# Trial 6 Score: 0.000000

# Running Trial 7 with config: {'colony_size': 4996, 'scouts': np.float64(0.1272727272727273), 'iterations': 325, 'seed': 83}
# Trial 7 Score: 0.000000

# Running Trial 8 with config: {'colony_size': 9917, 'scouts': np.float64(0.3090909090909091), 'iterations': 551, 'seed': 94}
# Trial 8 Score: 0.000000

# Running Trial 9 with config: {'colony_size': 9363, 'scouts': np.float64(0.4636363636363636), 'iterations': 583, 'seed': 76}
# Trial 9 Score: 0.000000

# Running Trial 10 with config: {'colony_size': 3141, 'scouts': np.float64(0.24545454545454545), 'iterations': 737, 'seed': 79}
# Trial 10 Score: 0.000000

# ✅ Best Config Found:
# {'colony_size': 1867, 'scouts': np.float64(0.7727272727272727), 'iterations': 283, 'seed': 100}
# Best Score: 0.000000
#############################################################################################################################################
