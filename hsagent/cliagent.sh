#!/bin/bash

# Prompt the user for the command they want to execute
echo -n "Enter the command you want to execute: "
read COMMAND

# Prompt for the custom prompt text
echo -n "Enter the prompt to use with ollama: "
read PROMPT

# Check if inputs are provided
if [ -z "$COMMAND" ] || [ -z "$PROMPT" ]; then
    echo "Both command and prompt are required." >&2
    exit 1
fi

# Run the command through ollama with the prompt, then execute the result and loop through output
echo "$PROMPT$COMMAND" | ollama run smollm2:135m | xargs -I {} sh -c '{}' | while IFS= read -r line; do
    echo "Processing: $line"
    # Add more processing logic here if needed
done

# Check for errors in the execution
if [ $? -ne 0 ]; then
    echo "An error occurred during execution." >&2
    exit 1
fi
