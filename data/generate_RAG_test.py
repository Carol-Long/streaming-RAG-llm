import random
import json

# Function to generate a line in the specified format
def generate_line(line_number):
    random_number = str(random.randint(1, 1000000))
    line = f"Line {line_number}: {random_number}"
    return line


result_strings = []
reference_strings = []

for i in range(11):
    # Generate 10 lines in each question
    lines = [generate_line(line_number) for line_number in range(10)]
    random_line = random.randint(1, 10)
    add_question = f"What is the number in Line {str(random_line)}?"
    lines.append(add_question)

    result_strings.append(lines)
    reference_strings.append(lines[random_line].split(' ')[2])

questions = []
for i in range(11):
    questions.append({"question_id": i+81, "category": "extraction", "turns": result_strings[i], "reference": [reference_strings[i]]})
# Convert the list of dictionaries to a JSON-formatted string
json_data = json.dumps(questions, indent=2)


# Write the JSON-formatted string to a file
with open("questions.jsonl", "w") as jsonl_file:
    for question in questions:
        jsonl_file.write(json.dumps(question) + '\n')


# an alternative test, try to give a successful recall
result_strings = []
reference_strings = []

for i in range(11):
    # Generate 10 lines in each question
    lines = [generate_line(line_number) for line_number in range(10)]
    joined_lines = " ".join(lines)
    random_line = random.randint(0, 9)
    add_question = f"What is the number in Line {str(random_line)}?"
    question_list = [joined_lines, add_question]

    result_strings.append(question_list)
    reference_strings.append(lines[random_line].split(' ')[2])

questions = []
for i in range(11):
    questions.append({"question_id": i+81, "category": "extraction", "turns": result_strings[i], "reference": [reference_strings[i]]})
# Convert the list of dictionaries to a JSON-formatted string
json_data = json.dumps(questions, indent=2)


# Write the JSON-formatted string to a file
with open("questions_joined.jsonl", "w") as jsonl_file:
    for question in questions:
        jsonl_file.write(json.dumps(question) + '\n')
