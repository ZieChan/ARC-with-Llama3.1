import json

# Load submission data
with open('submission.json', 'r') as f:
    submission = json.load(f)

# Load evaluation data
with open('arc-agi_evaluation_solutions.json', 'r') as f:
    evaluation = json.load(f)

total_problems = 0
correct_problems = 0

# Iterate over each problem in the submissions
for problem_id in submission:
    total_problems += 1
    correct_solution = evaluation.get(problem_id, [None])[0]
    submissions_list = submission[problem_id]
    problem_correct = False

    # Iterate over each submission entry for the problem
    for submission_entry in submissions_list:
        # Check each attempt
        for attempt in submission_entry.values():
            if attempt == correct_solution:
                correct_problems += 2
                problem_correct = True
                break
        if problem_correct:
            break


accuracy = correct_problems / total_problems if total_problems > 0 else 0

print(f'Number of correct solutions: {correct_problems}')
print(f'Total number of problems: {total_problems}')
print(f'Accuracy: {accuracy * 100:.2f}%')