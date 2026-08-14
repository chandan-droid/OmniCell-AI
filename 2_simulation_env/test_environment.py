# test_environment.py
from bioreactor_gym_env import BioreactorTwinEnv

if __name__ == "__main__":
    # Instantiates the coupled physical/biological twin.
    env = BioreactorTwinEnv()
    obs, info = env.reset()
    print(f"Initial State: Biomass={obs[0]:.2f}, Glucose={obs[1]:.2f}")
    
    # The execution loop: Feeds a static baseline action (e.g., [0.0]) into the 
    # environment over 50 simulation steps. Validates that CobraPy tensors and 
    # Euler physics arrays successfully pass data back and forth without shape 
    # mismatches or infinite calculation loops.
    for _ in range(50): 
        # Hardcode action [0.0] which maps to 0.25 L/h feed rate
        obs, reward, term, trunc, info = env.step([0.0])
        
    print(f"Final State: Biomass={obs[0]:.2f}, Glucose={obs[1]:.2f}, Lactate={obs[2]:.2f}")
    print("Verification Passed: dFBA and Euler Physics successfully integrated.")

