import random
import json

# Function to generate a line in the specified format
def generate_line(line_number, count):
    random_numbers = [str(random.randint(1, 1000)) for _ in range(count)]
    line = f"Line {line_number}: {' '.join(random_numbers)}"
    return line

# Generate 200 lines
lines = [generate_line(line_number, 23) for line_number in range(1, 211)]

# Combine the lines into a single string
result_string = '\n'.join(lines)

# Define a list of dictionaries with the desired format
questions = [
    {"question_id": 81, "category": "extraction", "turns": [result_string, "What is the first number in line 3?"], "reference": [lines[2].split(' ')[2]]},
]

# Convert the list of dictionaries to a JSON-formatted string
json_data = json.dumps(questions, indent=2)

# Write the JSON-formatted string to a file
with open("questions.jsonl", "w") as jsonl_file:
    for question in questions:
        jsonl_file.write(json.dumps(question) + '\n')

# print(lines[0:2])
# print(lines[30].split(' ')[2])
# print(lines[31].split(' ')[2])