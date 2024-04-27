import traci
import gym
from gym import spaces
import numpy as np


traci.start(["sumo-gui", "-c", r"C:\Program Files (x86)\Eclipse\Sumo\mynet.sumocfg"])





step = 0
#set the delay of the simulation to 50
#find the names of the phases
print(traci.trafficlight.getCompleteRedYellowGreenDefinition("C2"))




#print the available routes
print(traci.route.getIDList())
while step < 10000000:
    traci.simulationStep()
    if step % 35 == 0:
        traci.vehicle.add("a"+str(step), "r_0", departLane="0")
    if step % 16 == 0:
        traci.vehicle.add("b"+str(step), "r_1", departLane="0")
    if step % 15 == 0:
        traci.vehicle.add("c"+str(step), "r_2", departLane="0")
    if step % 14 == 0:
        traci.vehicle.add("d"+str(step), "r_3", departLane="0")
    if step % 2 == 0:
        traci.vehicle.add("e"+str(step), "r_4", departLane="0")
    if step % 10 == 0:
        traci.vehicle.add("f"+str(step), "r_5", departLane="0")
    if step % 11 == 0:
        traci.vehicle.add("g"+str(step), "r_6", departLane="0")
    if step % 3 == 0:
        traci.vehicle.add("h"+str(step), "r_7", departLane="0")
    if step % 9 == 0:
        traci.vehicle.add("i"+str(step), "r_8", departLane="0")
    if step % 4 == 0:
        traci.vehicle.add("j"+str(step), "r_9", departLane="0")
    if step % 16 == 0:
        traci.vehicle.add("k"+str(step), "r_10", departLane="0")
    if step % 11 == 0:
        traci.vehicle.add("l"+str(step), "r_11", departLane="0")
    
    step += 1

    #print the available lanes
    lane_ids = traci.lane.getIDList()
    #get the max queue length of all lanes from the lane ids
    queue_length = max([traci.lane.getWaitingTime(lane_id) for lane_id in lane_ids])
    print(queue_length)
