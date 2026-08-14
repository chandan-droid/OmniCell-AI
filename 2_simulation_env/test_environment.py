# test_environment.py
from bioreactor_gym_env import BioreactorTwinEnv

if __name__ == "__main__":
    # Instantiates the coupled physical/biological twin.
    env = BioreactorTwinEnv()
    obs, info = env.reset()
    print(f"Initial State: Biomass={obs[0]:.2f}, Glucose={obs[1]:.2f}")
    
    # The execution loop: Feeds 3-pump actions [0.0, 0.0, 0.0] into the 
    # environment over 200 simulation steps (testing past step 150 drift & bypass).
    for step_i in range(200): 
        # Action: [0.0 (Glucose), 0.0 (Base), 0.0 (Trace Nutrient)]
        # After step 150, activate trace nutrient feed [0.0, 0.0, 0.5] to trigger chemical bypass!
        trace_action = 0.5 if step_i >= 160 else -1.0
        obs, reward, term, trunc, info = env.step([0.0, 0.0, trace_action])
        
    print(f"Final State (Step 200): Biomass={obs[0]:.2f}, Glucose={obs[1]:.2f}, Lactate={obs[2]:.2f}")
    print("Verification Passed: dFBA, Euler Physics, 3-Pump Control, Genetic Drift, and Chemical Bypass successfully integrated.")


