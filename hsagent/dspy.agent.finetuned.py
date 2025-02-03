import dspy

#gpt4o_mini = dspy.LM('gpt-4o-mini-2024-07-18')
gpt4o_mini = dspy.LM(model=f"ollama/deepseek-r1:1.5b")
#gpt4o = dspy.LM('openai/gpt-4o')
gpt4o = dspy.LM(model=f"ollama/deepseek-r1:1.5b")
dspy.configure(experimental=True)


from dspy.datasets.alfworld import AlfWorld

alfworld = AlfWorld()
trainset, devset = alfworld.trainset[:200], alfworld.devset[-200:]
len(trainset), len(devset)

example = trainset[0]

with alfworld.POOL.session() as env:
    task, info = env.init(**example.inputs())

print(task)

class Agent(dspy.Module):
    def __init__(self, max_iters=50, verbose=False):
        self.max_iters = max_iters
        self.verbose = verbose
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

    def forward(self, idx):
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            if self.verbose:
                print(f"Task: {task}")

            for _ in range(self.max_iters):
                trajectory_ = "\n".join(trajectory)
                possible_actions = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                prediction = self.react(task=task, trajectory=trajectory_, possible_actions=possible_actions)
                trajectory.append(f"> {prediction.action}")

                if prediction.action.startswith("think:"):
                    trajectory.append("OK.")
                    continue

                obs, reward, done, info = env.step(prediction.action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)

                if self.verbose:
                    print("\n".join(trajectory[-2:]))

                if done:
                    break

        assert reward == int(info["won"][0]), (reward, info["won"][0])
        return dspy.Prediction(trajecotry=trajectory, success=reward)


agent_4o = Agent()
agent_4o.set_lm(gpt4o)
agent_4o.verbose = True

agent_4o(**example.inputs())

metric = lambda x, y, trace=None: y.success
evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, num_threads=16)

agent_4o.verbose = False
evaluate(agent_4o)

agent_4o_mini = Agent()
agent_4o_mini.set_lm(gpt4o_mini)

evaluate(agent_4o_mini)


