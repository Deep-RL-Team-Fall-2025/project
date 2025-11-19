from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

class Overcooked():
    def __init__(self, layout_name):
        """
        Initialize an Overcooked environment with a specified layout.
        
        Args:
            layout_name (str): Name of the layout to use. Options include:
                - "cramped_room"
                - "asymmetric_advantages"
                - "coordination_ring"
                - "forced_coordination"
                - "counter_circuit"
        
        Returns:
            env (OvercookedEnv): The initialized environment
            state (OvercookedState): The initial state
        """
        # Create the base MDP (Markov Decision Process)
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Create the environment wrapper
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=400)

        print("=" * 60)
        print("OVERCOOKED ENVIRONMENT INITIALIZED")
        print("=" * 60)
        
        print(f"\nLayout: {self.mdp.layout_name}")
        print(f"Grid dimensions: {self.mdp.width}x{self.mdp.height}")

        self.action_map = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']

    def _preprocess_state(self, state):
        featurized_state = self.mdp.lossless_state_encoding(state)
        return featurized_state

    def reset(self):
        # Reset the environment (modifies env.state internally)
        self.env.reset()
        # Get the initial state from the environment
        state = self.env.state
        state = self._preprocess_state(state)
        return state

    def print_state_info(self):
        """Print information about the current state."""
        state = self.env.state
        
        print("\n--- Player Information ---")
        for i, player in enumerate(state.players):
            print(f"Player {i}:")
            print(f"  Position: {player.position}")
            print(f"  Orientation: {player.orientation}")
            print(f"  Held object: {player.held_object}")
        
        print("\n--- Objects on Counters ---")
        if state.objects:
            for obj_pos, obj in state.objects.items():
                print(f"  Position {obj_pos}: {obj}")
        else:
            print("  No objects on counters")
        
        print("\n--- Pots State ---")
        if hasattr(state, 'all_orders'):
            print(f"  Current orders: {state.all_orders}")

    def visualize_layout(self):        
        state = self.env.state
        # Visualize the layout (text representation)
        print("\nLayout Visualization:")
        print(self.mdp.state_string(state))
        
    def display_actions(self):
        state = self.env.state
        print("\n" + "=" * 60)
        print("\n--- Available Actions ---")

        actions_per_player = self.mdp.get_actions(state)

        # Get action space info
        print("\n--- Environment Spaces ---")
        for i, actions in enumerate(actions_per_player):
            print(f"Actions for player {i}")
            print(f"- Number of actions: {len(actions)}")
            print(f"- Action indices: {list(range(len(actions)))}")
            print(f"- Action mappings: {actions}")
        
        return
    
    def step(self, action_player_1, action_player_2):
        action_1 = self.action_map[action_player_1]
        action_2 = self.action_map[action_player_2]
        next_state, reward, done, info = self.env.step((action_1, action_2))
        next_state = self._preprocess_state(next_state)
        return next_state, reward, done, False, info

def main():
    # Initialize the environment
    layout_name = "cramped_room"  # You can change this to other layouts
    env = Overcooked(layout_name)

    # Initial state
    state = env.reset()
    
    # Print state information
    print("=========================== STATE INFO ===========================")
    env.print_state_info()

    # Print state information
    print("=========================== STATE AS FEATURES ===========================")
    print(len(state))
    print([s.shape for s in state])
    # print(state)

    # Visualize layout
    print("=========================== LAYOUT ===========================")
    env.visualize_layout()

    # Show actions
    print("=========================== AVAILABLE ACTIONS ===========================")
    env.display_actions()
    
    # Take a step
    print("======================================================")
    print("Taking a step")
    next_state, reward, done, info = env.step(0, 2)

    # Visualize layout
    env.visualize_layout()
    print(reward)
    print(done)

    return


if __name__ == "__main__":
    main()