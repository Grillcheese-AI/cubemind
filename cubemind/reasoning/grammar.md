# CubeMind Reasoning Grammar — Natural Language → Symbolic Operations

## Problem-Solving Pipeline

```
1. Understanding the Problem
   a) Read problem
   b) Define problem context
   c) Split problem into chunks
   d) Define variables for the problem
   e) Define variables contexts
   f) Read the chunks in order
   g) Evaluate possible answers based on current knowledge + future predictions
   h) Get the ones that make sense
   i) Debate them and pick one right answer

2. Define causal links between variables
3. Create a global rule for that type of problem
4. Store it in the global world of knowledge for future usage if same problem signature is encountered
```

## Grammar → Math Mapping

| Natural Language | Symbolic Operation |
|---|---|
| When / While | Loop / temporal binding |
| subject | variable |
| is | equals (=) |
| from | container / array lookup |
| subject gets | subject.count += 1; source.count -= 1 |
| subject gets X | X = type of object |
| subject gets X from Y | X = type, Y = source container |
| subject A X from Y | A = action (gets/puts/subtract/add/destroys), X = object type, Y = container |
| subject has Z and gives X many to Y number of subjectsB | division: answer = total_left (context-dependent) |
| how much / how many | final variable / problem declaration (triggers solve) |

## Formal Grammar

```
while X -> action -> variable:
    X->fetch=Y["Z"]
    B = ?
```

## VSA Encoding Strategy

Each grammar element maps to a block-code operation:
- **subject** → VSA variable (block-code hypervector)
- **action** → VSA role vector (bind with subject + object)
- **container** → VSA set (bundle of items)
- **causal link** → VSA binding chain: bind(cause, effect)
- **rule** → stored as bound pair: bind(problem_signature, solution_template)
