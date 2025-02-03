import dspy
from datasets import load_dataset
import json

# Configure DSPy to use Ollama with the deepseek-r1:8b model
ollama_lm = dspy.OllamaLocal(model="deepseek-r1:8b", base_url="http://localhost:11434")
dspy.configure(lm=ollama_lm)

# Load the dataset
swebench = load_dataset('princeton-nlp/SWE-bench', split='test')

# Filter the specific instance
instance = swebench.filter(lambda x: x['instance_id'] == 'sympy__sympy-20590')

# Define the agent
class CodeFixer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Step 1: Analyze the problem statement
        self.analyze = dspy.Predict("problem_statement -> analysis")
        # Step 2: Generate a code fix based on the analysis
        self.generate = dspy.Predict("analysis -> model_patch")

    def forward(self, problem_statement):
        # Step 1: Analyze the problem
        analysis = self.analyze(problem_statement=problem_statement).analysis
        print("Analysis:")
        print(analysis)

        # Step 2: Generate the code fix
        model_patch = self.generate(analysis=analysis).model_patch
        return model_patch

# Initialize the agent
agent = CodeFixer()

# Generate the solution
problem_statement = instance[0]['problem_statement']
model_patch = agent(problem_statement=problem_statement)

# Print the generated code fix
print("\nGenerated Code Fix:")
print(model_patch)

# Format predictions as a list of dictionaries
predictions = [
    {
        "instance_id": "sympy__sympy-20590",  # Task ID
        "model_patch": model_patch  # Generated code fix
    }
]

# Save predictions to a JSON file
with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

print("\nPredictions saved to predictions.json")
