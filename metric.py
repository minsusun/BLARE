import argparse
from src.datasets import *
from src.templates import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, choices=['strategyqa', '2wikihop', 'wikiasp', 'asqa'])
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)

args = parser.parse_args()

print(f"INFO:: arguments - {vars(args)}")
print()

if any(v is None for _,v in vars(args).items()):
    raise Exception("Error:: You should provide all the arguments")

print(f"Loading:: DataSet({args.dataset}) - '{args.input}'")

# load data
if args.dataset == 'strategyqa':
    data = StrategyQA(args.input)
elif args.dataset == '2wikihop':
    data = WikiMultiHopQA(args.input)
elif args.dataset == 'asqa':
    data = ASQA(args.input)
elif args.dataset == 'wikiasp':
    data = WikiAsp(args.input)
else:
    raise NotImplementedError

data.format()

print(f"Loading:: Done")
print()

result = []

print(f"Extracting:: {args.output}")
 
with open(args.output, 'r') as f:
    for line in f:
        result.append(json.loads(line))

EM_data = {
    'correct': 0,
    'incorrect': 0
}
F1_data = {
    'f1': 0,
    'precision': 0,
    'recall': 0
}

for row in result:
    answer = row['output'].split('So the answer is')[-1]
    gt = row['answer']
    gt_id = row['answer_id']
    EM = data.exact_match_score(answer, gt, gt_id)
    F1 = data.f1_score(answer, gt, gt_id)
    EM_data['correct'] += EM['correct']
    EM_data['incorrect'] += EM['incorrect']
    F1_data['f1'] += F1['f1']
    F1_data['precision'] += F1['precision']
    F1_data['recall'] += F1['recall']

print("Extracting:: Done")
print()

print("METRIC:: EM")
print(f"EM:: Correct {EM_data['correct']/len(result):.2f} Incorrect {EM_data['incorrect']/len(result):.2f}")
print()

print("METRIC:: F1")
print(f"F1:: F1 {F1_data['f1']/len(result):.2f} Precision {F1_data['precision']/len(result):.2f} Recall {F1_data['recall']/len(result):.2f}")
print()