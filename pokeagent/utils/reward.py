import utils.query_llm as query_llm
from typing import List

class ShapedReward(object):
    """
    Class for generating shaped reward function.
    """

    def __init__(self):
        self.GOAL_PROMPT = """[GOAL] The goal of MountainCar-v0 is to reach the right-most hill by accelerating the car using the provided actions. Assume that the reward for completing the game is 1."""
        self.PARAM_PROMPT = """[PARAMS] x_position is from -INF to INF. velocity is from -INF to INF. action is always either 0 for accelerate left, 1 for don't accelerate, 2 for accelerate right."""
        self.EXAMPLE_TRAJECTORY = """(-100, 1, 0), (-100, 2, 0,) (-50, 2, 0), (0, 2, 0, ), (50, 2, 1), (100, 2, 1) """
        self.PROMPT = """When I ask you a question, only respond with the code which is the answer. No padding words before or after. Code a function in Python called "reward()" for MountainCar-v0 from OpenAI gym classic, where the arguments are x_position, velocity, action.  Do not provide any code other than the function definition for "reward()". The reward function will be dependent on all the arguments in a non-trivial way.

{goal}

{param}

Below is a history of the trajectory in the form (state1, action1, reward1), (state2, action2, reward2.),.. that your reinforcement learning agent took last episode. Modify its reward function to encourage the agent to reach its goal faster. 

{trajectory}
        """

        self.log_of_responses = []

    def build_prompt(self, trajectory : List, failed : bool):
        prompt = self.PROMPT.format(
            goal=self.GOAL_PROMPT,
            param=self.PARAM_PROMPT,
            trajectory=str(trajectory),
        )

        if failed:
            prefix = "Make sure to define a function def reward() in Python that exactly follows the arguments specified and returns one reward value.\n"
            prompt = prefix + prompt
        return prompt

    def generate_default_func(self):
        """
        Default reward function.
        """
        code = "def reward(x_position, velocity, action):\n\treturn 0 "
        exec(code, globals())
        return reward

    def generate_reward_func(self, trajectory : List):
        """
        Current format hard-coded for MountainCar.
        """
        # while True:
        failed = False
        trajectory = trajectory[-50:]
        code = None
        for i in range(2):
            try:
                prompt = self.build_prompt(trajectory, failed=failed)
                print("[P]:", prompt)
                cost, code = query_llm.query_gpt(prompt)
                self.log_of_responses.append({"prompt": prompt, "code": code})
                print('code', code)
                exec(code, globals())
            except:
                if failed:
                    print("Error: failed again!")
                    print(code)
                else:
                    print("Error in trying to define function!")
                failed = True
                continue

            try:
                _ = reward(0, 0, 0)
            except:
                print("Reward arguments are wrong!")
                print(reward.__dict__)
            return reward
        return self.generate_default_func()

    def dump(self):
        print(self.log_of_responses)


if __name__ == "__main__":
    sr = ShapedReward()
    reward = sr.generate_reward_func([0,0,0])
    sr.dump()
    print(reward(1, 2, 3))
