import random
import json
import os

print("Script started.")

# Function to generate a line with unique content
def generate_line(line_number):
    content = [
        "The Eiffel Tower is located in Paris, France.",
        "The Great Wall of China is visible from space.",
        "The Statue of Liberty was a gift to the United States from France.",
        "The Earth orbits the Sun once every 365.25 days.",
        "The Amazon Rainforest is the largest tropical rainforest in the world.",
        "Mount Everest is the highest mountain above sea level.",
        "The Sahara Desert is the largest hot desert in the world.",
        "The city of Venice is famous for its canals and waterways.",
        "The Nile River is the longest river in the world.",
        "Shakespeare wrote a total of 37 plays.",
        "The Great Pyramid of Giza was built by Pharaoh Khufu.",
        "The human body has 206 bones.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The Grand Canyon is located in Arizona, USA.",
        "The Taj Mahal in India is a mausoleum built by Emperor Shah Jahan.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "The Colosseum in Rome could hold up to 50,000 spectators.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "The Leaning Tower of Pisa's tilt was caused by an unstable foundation.",
        "Honey never spoils and can be eaten thousands of years later.",
        "The light bulb was invented by Thomas Edison.",
        "The heart of a blue whale can weigh as much as a small car.",
        "The deepest part of the ocean is the Mariana Trench.",
        "The Louvre in Paris is the world's largest art museum.",
        "The lifespan of a dragonfly is only around 24 hours.",
        "Penguins can't fly, but they are excellent swimmers.",
        "The Dead Sea is so salty that you can easily float in it.",
        "The Great Barrier Reef is the world's largest coral reef system.",
        "The Roman Empire lasted for almost a thousand years.",
        "The human brain is composed of over 100 billion neurons.",
        "The rings of Saturn are made mostly of ice particles.",
        "The first person to walk on the moon was Neil Armstrong.",
        "The lifespan of a housefly is about 28 days.",
        "The city of Timbuktu is famous for its ancient manuscripts.",
        "The highest recorded temperature on Earth was in Furnace Creek Ranch, Death Valley, USA.",
        "Giraffes have the same number of neck vertebrae as humans.",
        "The world's largest desert is Antarctica.",
        "A group of crows is called a murder.",
        "The Great Barrier Reef can be seen from space.",
        "Bananas grow pointing upwards.",
        "The Andes is the longest mountain range in the world.",
        "The kangaroo and emu were chosen as symbols of Australia due to their inability to walk backward.",
        "The human eye can distinguish about 10 million different colors.",
        "The game of chess was invented in India.",
        "The first computer was invented by Charles Babbage.",
        "The deepest natural point on Earth is the Challenger Deep in the Mariana Trench.",
        "The largest mammal on Earth is the blue whale.",
        "The currency of Japan is the Yen.",
        "The first successful airplane was flown by the Wright brothers.",
        "The capital of Egypt is Cairo."
    ]

    random_content = random.choice(content)
    line = f"Line {line_number}: {random_content}"
    return line

# This function generates the questions and reference answers
def generate_qa_pairs(num_pairs):
    result_strings = []
    reference_strings = []

    for i in range(num_pairs):
        # Generate lines with content
        lines = [generate_line(line_number) for line_number in range(1, 11)]
        # Choose a line at random to ask a question about
        random_line_index = random.randint(0, 9)
        random_line = lines[random_line_index]
        add_question = f"In which line can we learn about the origin of the Statue of Liberty?"

        result_strings.append(lines + [add_question])
        # Check if the selected line contains the content about the Statue of Liberty
        if "Statue of Liberty" in random_line:
            reference_strings.append(str(random_line_index + 1))  # Line numbers are 1-indexed
        else:
            reference_strings.append("Not present")

    questions = []
    for i, lines in enumerate(result_strings):
        questions.append({
            "question_id": i + 1,  # Arbitrary question ID
            "category": "extraction",
            "turns": lines,
            "reference": reference_strings[i]  # The line number or "Not present"
        })

    return questions

print("Test script started.")

try:
    # Generate 10 QA pairs
    qa_pairs = generate_qa_pairs(3)

    # Write the QA pairs to a JSON Lines file
    json_filename = "simplified_questions.jsonl"
    with open(json_filename, "w") as jsonl_file:
        for pair in qa_pairs:
            jsonl_file.write(json.dumps(pair) + '\n')

    print(f"QA pairs written to {json_filename}")
    print(f"File is located at: {os.path.abspath(json_filename)}")  # Shows the absolute path of the created file
except Exception as e:
    print(f"An error occurred: {e}")
