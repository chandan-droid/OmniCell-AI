import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bioreactor_gym_env import BioreactorTwinEnv

# 1. Initialize the Bio-Twin Sandbox
print("[System] Initializing OmniCell-AI Bio-Twin...")
env = BioreactorTwinEnv()
obs, info = env.reset()

# 2. Setup Data Storage Arrays
time_steps = [0.0]
biomass_data = [obs[0]]
glucose_data = [obs[1]]
lactate_data = [obs[2]]

# 3. Configure the Live Matplotlib Figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title('OmniCell-AI: Live Digital Twin')
fig.suptitle("Bioreactor Telemetry (Live)", fontsize=14, fontweight='bold')

# Configure Lines
line_x, = ax1.plot(time_steps, biomass_data, color='green', linewidth=2, label="Biomass (g/L)")
line_s, = ax2.plot(time_steps, glucose_data, color='blue', linewidth=2, label="Glucose (g/L)")
line_l, = ax3.plot(time_steps, lactate_data, color='red', linewidth=2, label="Lactate (mmol/L)")

# Format Axes
ax1.set_ylabel("Cells (g/L)")
ax2.set_ylabel("Sugar (g/L)")
ax3.set_ylabel("Toxin (mmol/L)")
ax3.set_xlabel("Batch Time (Hours)")

for ax in (ax1, ax2, ax3):
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.6)

# 4. The Animation Loop (Executes 10 times a second)
def update_dashboard(frame):
    # 1. Administer the cure! 
    # The mutation hits at step 150. We will turn on the Trace Nutrient pump 
    # at step 152 (Hour 15.2) to mathematically bypass the choked enzyme.
    if env.current_step >= 152:
        trace_pump = 0.5  # Pump ON
    else:
        trace_pump = -1.0 # Pump OFF
        
    # Action Array: [Glucose Feed, Base Buffer, Trace Nutrients]
    action = np.array([0.5, -1.0, trace_pump], dtype=np.float32)
    
    # Step the simulation physics forward
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Append new ground-truth data
    current_time = env.current_step * env.dt
    time_steps.append(current_time)
    biomass_data.append(obs[0])
    glucose_data.append(obs[1])
    lactate_data.append(obs[2])
    
    # Update the lines on the charts
    line_x.set_data(time_steps, biomass_data)
    line_s.set_data(time_steps, glucose_data)
    line_l.set_data(time_steps, lactate_data)
    
    # Dynamically scale the axes as data grows
    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()
        
    # If the tank crashes, pause to let the user see the failure, then restart
    if terminated or truncated:
        print(f"[Warning] Batch Terminated at Hour {current_time:.1f}. Restarting tank...")
        env.reset()
        time_steps.clear(); biomass_data.clear(); glucose_data.clear(); lactate_data.clear()
        
    return line_x, line_s, line_l

# 5. Launch the UI
# interval=100 means the simulation ticks every 100 milliseconds
ani = animation.FuncAnimation(fig, update_dashboard, interval=100, cache_frame_data=False)

plt.tight_layout()
plt.show()


#Visual Behaviors to Observe:
#Hours 0 – 15 (Exponential Growth): Green line (Biomass) curves upwards while Blue line (Glucose) drops rapidly.
#Hour 15 / Step 150 (Genetic Drift Impact): induce_metabolic_drift() triggers synthetic Pyruvate Dehydrogenase (PDH) enzyme suppression.
#Toxin Spike & Crash: Red line (Lactate) spikes past 25.0 mmol/L, triggering a Quality by Design (QbD) failure, episode termination, and auto-restart.