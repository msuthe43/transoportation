import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np

class SumoEnvironment(gym.Env):
    """Custom Environment that interfaces with SUMO using TraCI."""
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path, render_mode='human'):
        super(SumoEnvironment, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(159,), dtype=np.float32)
        self.config_path = config_path
        self.sumoBinary = "sumo-gui"
        self.is_connected = False
        self.initialize_simulation()

    def initialize_simulation(self):
        if self.is_connected:
            traci.close()
        sumo_cmd = [self.sumoBinary, "-c", self.config_path, "--step-length", "1", "--start", "--quit-on-end"]
        traci.start(sumo_cmd)
        self.is_connected = True
        self.last_action = 0
        self.yellowtimer = 0
        self.greentimer = 0
        self.waiting_times = []

    def _start_sumo(self):
        sumo_cmd = [self.sumoBinary, "-c", self.config_path, "--step-length", "1", "--start"]
        traci.start(sumo_cmd)

    def apply_green_phase(self, action):
        """Applies the green light configuration based on the action."""
        traffic_configurations = {
            0: "rrrrGGGrrrrrrrGGGrr",
            1: "rrrrrrrGGrrrrrrrrGG",
            2: "GGGrrrrrrGGGrrrrrrr",
            3: "rrrGrrrrrrrrGGrrrrr"
        }
        traci.trafficlight.setRedYellowGreenState("C2", traffic_configurations[action])


    def step(self, action):
        # Proceed with the simulation step
        traci.simulationStep()

        self.apply_green_phase(action)
        
        lane_ids = traci.lane.getIDList()
        current_state = self._get_state()
        queue_lengths = current_state[:len(lane_ids)]
        average_speeds = current_state[len(lane_ids):2*len(lane_ids)]
        #get the vehicles in the simulation
        num_cars = len(traci.vehicle.getIDList())
        #print(num_cars)
        reward = -num_cars * queue_lengths.sum()  # Reward is the negative of the total number of cars in the simulation

        if action != self.last_action:
            reward -= 5



        #get the time of the simulation
        #get the time of the simulation
        mytime = traci.simulation.getTime()
        done = mytime > 2048  # End the simulation after 7200 seconds


        if mytime % 29 == 0:
            traci.vehicle.add("a"+str(mytime), "r_0", departLane="0")
        if mytime % 18 == 0:
            traci.vehicle.add("b"+str(mytime), "r_1", departLane="0")
        if mytime % 20 == 0:
            traci.vehicle.add("c"+str(mytime), "r_2", departLane="0")
        if mytime % 25 == 0:
            traci.vehicle.add("d"+str(mytime), "r_3", departLane="0")
        if mytime % 3 == 0:
            traci.vehicle.add("e"+str(mytime), "r_4", departLane="0")
        if mytime % 20 == 0:
            traci.vehicle.add("f"+str(mytime), "r_5", departLane="0")
        if mytime % 22 == 0:
            traci.vehicle.add("g"+str(mytime), "r_6", departLane="0")
        if mytime % 3 == 0:
            traci.vehicle.add("h"+str(mytime), "r_7", departLane="0")
        if mytime % 6 == 0:
            traci.vehicle.add("i"+str(mytime), "r_8", departLane="0")
        if mytime % 6 == 0:
            traci.vehicle.add("j"+str(mytime), "r_9", departLane="0")
        if mytime % 23 == 0:
            traci.vehicle.add("k"+str(mytime), "r_10", departLane="0")
        if mytime % 13 == 0:
            traci.vehicle.add("l"+str(mytime), "r_11", departLane="0")

        return current_state, reward, done, False, {}

    def reset(self, **kwargs):
        # Use kwargs to accept any additional unused parameters that might be passed by the environment wrapper
        self.close()
        self.initialize_simulation()
        return self._get_state(), {'a',1}  # Ensure this returns only the state, not a tuple



    def _get_state(self):
        lane_ids = traci.lane.getIDList()
        queue_lengths = np.array([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids], dtype=np.float32)
        average_speeds = np.array([traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in lane_ids], dtype=np.float32)
        num_cars = np.array([traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in lane_ids], dtype=np.float32)
        state = np.concatenate([queue_lengths, average_speeds, num_cars])
        total_waiting_time = np.sum([traci.lane.getWaitingTime(lane_id) for lane_id in lane_ids])
        #get the number of cars in the simulation
        self.waiting_times.append(total_waiting_time)
        #print("Updated waiting times:", self.waiting_times)
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.is_connected:
            traci.close()
            self.is_connected = False

# Create the environment
config_path = r"C:\Program Files (x86)\Eclipse\Sumo\mynet.sumocfg"

from stable_baselines3 import PPO

# Specify the path to the saved model
model_path = r"C:\Users\saver\3010project\model.zip"
model = PPO.load(model_path)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

single_env = SumoEnvironment(config_path)
# Assuming you have a model loaded or trained
obs, _ = single_env.reset()  # Reset the environment and get initial observations

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  # Predict the action
    action = int(action)  # Convert action from NumPy array to integer
    obs, rewards, done, info, _ = single_env.step(action)  # Step the environment with the action
    single_env.render()  # Optional: Render the environment state if needed


waiting_times = single_env.waiting_times  # Access the waiting_times from the single environment instance
#save the model for later use in zip file
model.save("model")

# Plot waiting times
import matplotlib.pyplot as plt
plt.plot(waiting_times)
plt.title('Waiting Time Over Simulation Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Total Waiting Time')
plt.show()
