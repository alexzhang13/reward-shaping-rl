import os
import gym


def main():
    env = gym.make("MountainCar-v0")
    env.reset()
    step = 0
    while True:
        step += 1
        obs, reward, done, info, dp = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
            break
        print("[OBS]: ", obs)
        print("[REWARD]: ", reward)
    env.close()


if __name__ == "__main__":
    main()
