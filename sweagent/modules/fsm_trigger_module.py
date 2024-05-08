import dspy
from dspygen.utils.dspy_tools import init_dspy


class FSMTriggerModule(dspy.Module):
    """FSMTriggerModule"""

    def __init__(self, **forward_args):
        super().__init__()
        self.forward_args = forward_args
        self.output = None

    def forward(self, prompt, possible_triggers):
        pred = dspy.Predict("prompt, possible_triggers -> chosen_trigger")
        self.output = pred(prompt=prompt, possible_triggers=possible_triggers).chosen_trigger
        return self.output


def fsm_trigger_call(prompt, possible_triggers):
    fsm_trigger = FSMTriggerModule()
    return fsm_trigger.forward(prompt=prompt, possible_triggers=possible_triggers)


def main():
    init_dspy()
    prompt = ""
    possible_triggers = ""
    result = fsm_trigger_call(prompt=prompt, possible_triggers=possible_triggers)
    print(result)


if __name__ == "__main__":
    main()
