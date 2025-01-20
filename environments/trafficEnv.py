import gymnasium as gym
import numpy as np
import traci
from ray.rllib.env.multi_agent_env import MultiAgentEnv
single_light_observation_space = gym.spaces.Dict({
    "local_state": gym.spaces.Dict({
        "queue_length": gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
        "current_phase": gym.spaces.Discrete(3),
        "waiting_time": gym.spaces.Box(low=0, high=300, shape=(4,), dtype=np.float32),
    }),
    "upstream_info": ...,
    "downstream_info": ...,
    "position": ...,
})

# The action space for ONE traffic light:
single_light_action_space = gym.spaces.Dict({
    "phase": gym.spaces.Discrete(3),
    "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
})


class TrafficLightEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.num_lights = 5
        self.sumo_cfg = "data/sumo/simulation.sumocfg"
        self.light_ids = ["A0", "B0", "C0", "D0", "E0"]

        # Simulation parameters
        self.current_step = 0
        self.max_steps = 10000
        
        # --- Define observation space ---
        # Replace infinite bounds with large numbers, e.g. 1e6
        # Also note: sumo sometimes has more than 3 phases, but we set Discrete(3) for demonstration.
        self.observation_spaces = {
            light_id: single_light_observation_space
            for light_id in self.light_ids
        }
        self.action_spaces = {
            light_id: single_light_action_space
            for light_id in self.light_ids
        }

        # --- Define action space ---
        self.action_space = gym.spaces.Dict({
            "phase": gym.spaces.Discrete(3),  # (Green, Yellow, Red)
            "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
        })


    def reset(self, *, seed=None, options=None):
        try:
            traci.close()
        except Exception:
            pass
        traci.start([
            "sumo", "-c", self.sumo_cfg, "--start", "--quit-on-end=true",
            "--random", "--time-to-teleport", "-1", "--waiting-time-memory", "300"
        ])
        self.current_step = 0

        for _ in range(10):
            traci.simulationStep()

        observations = {
            light_id: self._get_observation(light_id) for light_id in self.light_ids
        }

        return observations, {}



    def _get_queue_length(self, light_id):
        """Get queue length for up to 4 lanes. Return a list of size 4."""
        queue_length = [0] * 4
        try:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            for i, lane in enumerate(lanes[:4]):
                queue_length[i] = traci.lane.getLastStepHaltingNumber(lane)
        except traci.exceptions.TraCIException:
            pass
        return queue_length


    def _get_waiting_time(self, light_id):
        """Get waiting times for up to 4 lanes. Return a list of size 4."""
        waiting_time = [0] * 4
        try:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            for i, lane in enumerate(lanes[:4]):
                waiting_time[i] = traci.lane.getWaitingTime(lane)
        except traci.exceptions.TraCIException:
            pass
        return waiting_time

    def _get_upstream_vehicles(self, current_light):
        """Return number of vehicles on edges upstream from current_light."""
        vehicles = [0] * self.num_lights
        edges = ["A0B0", "B0C0", "C0D0", "D0E0"]  # adapt to real edges
        if current_light > 0:
            for i in range(current_light):
                try:
                    vehicles[i] = traci.edge.getLastStepVehicleNumber(edges[i])
                except traci.exceptions.TraCIException:
                    pass
        return vehicles

    def _calculate_arrival_times(self, current_light):
        """Estimate arrival times from upstream lights to current light."""
        edges = ["A0B0", "B0C0", "C0D0", "D0E0"]
        if current_light == 0:
            return [0] * self.num_lights

        # Example: each light is ~400 meters apart
        distances = [(current_light - i) * 400 for i in range(current_light)]
        arrival_times = []
        for i, dist in enumerate(distances):
            try:
                speed = traci.edge.getLastStepMeanSpeed(edges[i])
                if speed > 0:
                    arrival_times.append(dist / speed)
                else:
                    arrival_times.append(1e6)  # no movement => large arrival time
            except traci.exceptions.TraCIException:
                arrival_times.append(1e6)
        return arrival_times

    def _get_upstream_phases(self, current_light):
        phases = [0] * self.num_lights
        for i in range(current_light):
            light_id = self.light_ids[i]
            try:
                phases[i] = traci.trafficlight.getPhase(light_id) % 3
            except traci.exceptions.TraCIException:
                phases[i] = 0
        return phases

    def _get_downstream_queues(self, current_light):
        queues = [0] * self.num_lights
        for i in range(current_light + 1, self.num_lights):
            light_id = self.light_ids[i]
            try:
                queues[i] = sum(self._get_queue_length(light_id))
            except Exception:
                pass
        return queues

    def _get_downstream_phases(self, current_light):
        phases = [0] * self.num_lights
        for i in range(current_light + 1, self.num_lights):
            light_id = self.light_ids[i]
            try:
                # clamp to 3 for (G,Y,R)
                phases[i] = traci.trafficlight.getPhase(light_id) % 3
            except traci.exceptions.TraCIException:
                phases[i] = 0
        return phases

    def step(self, actions):
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                # No more vehicles
                return {}, {}, {"__all__": True}, {}

            self.current_step += 1

            # Apply actions for each traffic light
            for light_id, action in actions.items():
                phase = action.get("phase", 0)
                duration = float(action.get("duration", 30))
                # clamp duration
                duration = max(5, min(60, duration))
                try:
                    traci.trafficlight.setPhase(light_id, phase)
                    traci.trafficlight.setPhaseDuration(light_id, duration)
                except traci.exceptions.TraCIException as e:
                    print(f"Error setting traffic light {light_id}: {e}")

            # Advance the simulation
            traci.simulationStep()

            # Build next observations and rewards
            observations = {
                light_id: self._get_observation(light_id)
                for light_id in self.light_ids
            }
            rewards = {
                light_id: self._compute_reward(light_id)
                for light_id in self.light_ids
            }

            # Check if done
            done = (
                traci.simulation.getMinExpectedNumber() <= 0
                or self.current_step >= self.max_steps
            )
            dones = {light_id: done for light_id in self.light_ids}
            dones["__all__"] = done

            infos = {
                light_id: {
                    "step": self.current_step,
                    "vehicles_in_simulation": traci.vehicle.getIDCount(),
                    "simulation_time": traci.simulation.getTime(),
                }
                for light_id in self.light_ids
            }

            return observations, rewards, dones, infos

        except traci.exceptions.FatalTraCIError as e:
            print(f"Fatal TraCI Error: {e}")
            return {}, {}, {"__all__": True}, {}


    def _compute_reward(self, light_id):
        """Simple reward example using queue, waiting time, etc."""
        queue_lengths = self._get_queue_length(light_id)
        waiting_times = self._get_waiting_time(light_id)

        queue_penalty = -0.05 * sum(queue_lengths)
        waiting_penalty = -0.02 * sum(waiting_times)

        # Throughput = # of vehicles in the controlled lanes
        throughput = 0
        try:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            throughput = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
        except traci.exceptions.TraCIException:
            pass

        throughput_reward = 0.7 * throughput

        # Optional penalty for e.g. being in a "yellow" or "red" for too long
        try:
            current_phase = traci.trafficlight.getPhase(light_id)
            # If phase == 1 is "yellow", apply a small penalty
            phase_penalty = -0.1 if current_phase == 1 else 0
        except traci.exceptions.TraCIException:
            phase_penalty = 0

        return queue_penalty + waiting_penalty + throughput_reward + phase_penalty

    def _get_observation(self, light_id):
        """Build the dictionary observation for one traffic light."""
        light_index = self.light_ids.index(light_id)
        # Current phase in sumo can be 0..N. We clamp to 0..2 for (G,Y,R) demonstration.
        try:
            current_phase = traci.trafficlight.getPhase(light_id)
            if current_phase > 2:
                # Force it into a 0..2 range if your sumo has 3-lights logic
                current_phase = current_phase % 3
        except traci.exceptions.TraCIException:
            current_phase = 0

        return {
            "local_state": {
                "queue_length": np.array(self._get_queue_length(light_id), dtype=np.float32),
                "current_phase": current_phase,
                "waiting_time": np.array(self._get_waiting_time(light_id), dtype=np.float32),
            },
            "upstream_info": {
                "vehicles_approaching": np.array(
                    self._get_upstream_vehicles(light_index), dtype=np.float32
                ),
                "time_to_arrival": np.array(
                    self._calculate_arrival_times(light_index), dtype=np.float32
                ),
                "upstream_phases": np.array(
                    self._get_upstream_phases(light_index), dtype=np.int32
                ),
            },
            "downstream_info": {
                "queue_lengths": np.array(
                    self._get_downstream_queues(light_index), dtype=np.float32
                ),
                "phases": np.array(
                    self._get_downstream_phases(light_index), dtype=np.int32
                ),
            },
            "position": {
                "light_number": light_index,
                "is_first": 1 if light_id == self.light_ids[0] else 0,
                "is_last": 1 if light_id == self.light_ids[-1] else 0,
            },
        }
