import random
import json
import os

# Function to generate a unique fake story with a specified main character
def generate_story(story_number, main_character=None):
    # Define a pool of characters, actions, and settings to construct the stories
    characters = [
        "wizard", "knight", "dragon", "princess", "alien",
        "detective", "robot", "pirate", "ghost", "superhero",
        "enchantress", "spy", "viking", "cyborg", "mermaid",
        "time traveler", "sorcerer", "elf", "bounty hunter", "explorer"
    ]

    actions = [
        "embarked on an adventure", "discovered a secret passage", "saved the village from danger",
        "found a hidden treasure", "solved an ancient riddle", "escaped from a mysterious labyrinth",
        "uncovered an alien conspiracy", "prevented a magical cataclysm", "outwitted a cunning adversary",
        "restored peace to a warring kingdom", "rescued a rare mythical creature", "invented a revolutionary device",
        "translated a cryptic prophecy", "survived a harrowing ordeal", "deciphered a complex code",
        "thwarted a villain’s evil plot", "commandeered a ghost ship", "tamed a wild beast",
        "mastered a forgotten art", "befriended a misunderstood entity"
    ]

    settings = [
        "in the enchanted forest", "on the distant moon", "in the vast desert", "atop the windy mountains",
        "within the ancient ruins", "amidst the bustling cityscape", "deep in the underwater city",
        "across the frozen tundra", "through the time-warped dimension", "inside the labyrinthine caves",
        "beneath the starry sky", "among the floating islands", "along the sunken coral reef",
        "within the walls of the grand castle", "across the sprawling savanna", "under the canopy of the rainforest",
        "on the edge of the known universe", "in the heart of the volcanic crater", "through the neon-lit urban jungle",
        "within the sacred temple grounds"
    ]

    
    # Ensure the main character appears only in story 2
    if main_character is not None and story_number == 2:
        character = main_character
    else:
        character = random.choice([c for c in characters if c != main_character])
    
    action = random.choice(actions)
    setting = random.choice(settings)
    
    story = f"{character} {action} {setting}."
    return f"Story {story_number}: {story}"

# Function to generate the stories and a specific follow-up question
def generate_story_set(num_sets, num_stories=10):
    story_sets = []

    for _ in range(num_sets):
        # Randomly choose a main character for story 2
        main_character = random.choice([
            "rabbit","cat","elephant","wolf","fox","deer","fish","dog",
            "human","tiget","lion","monkey","pigeon","unicorn","bear","horse",
            "zebra","frog","panda"
        ])

        stories = [generate_story(i + 1, main_character if i == 1 else None) for i in range(num_stories)]
        
        # Modified to create a list of turns with each story as a separate turn
        turns = []
        for i, story in enumerate(stories):
            turns.append(f"Line {i + 1}: {story}")
        
        # Adding the follow-up question as a separate turn
        follow_up_question = f"Can you repeat the story about {main_character} I just told you about?"
        turns.append(follow_up_question)

        # Adding the reference story (story 2)
        reference = stories[1].split(": ")[1]  # Extracting only the story part

        story_sets.append({
            "question_id": _ + 1,
            "category": "extraction",
            "turns": turns,
            "reference": reference
        })

    return story_sets

try:
    story_sets = generate_story_set(3)
    json_filename = "unique_story_sets_new.jsonl"
    with open(json_filename, "w") as jsonl_file:
        for story_set in story_sets:
            jsonl_file.write(json.dumps(story_set) + '\n')

    print(f"Story sets written to {json_filename}")
    print(f"File is located at: {os.path.abspath(json_filename)}")
except Exception as e:
    print(f"An error occurred: {e}")